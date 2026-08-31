/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "virtual_swapchain_runtime.hpp"
#include "d3b3_production_seams.hpp"
#include "d3b1_present_path.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>
#include <array>
#include <cstring>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

namespace {
class RuntimeRetirementWakeTarget final : public DeviceRetirementWakeTarget {
public:
    explicit RuntimeRetirementWakeTarget(VirtualSwapchainState& value) : state(&value) {}
    void notifyDeviceRetirement() noexcept override {
        const std::scoped_lock lock(mutex);
        if (state) state->wake();
    }
    void detach() noexcept {
        const std::scoped_lock lock(mutex);
        state = nullptr;
    }
private:
    std::mutex mutex;
    VirtualSwapchainState* state{};
};
}

VirtualImageBacking::VirtualImageBacking(const vk::Vulkan& vk,
        const VirtualSwapchainImageSpec& spec,
        const vk::ImageCreateOptions& options) :
    image(vk, spec.extent, spec.format, spec.usage, options),
    ready(vk), originalReady(vk) {}

PresentExecutionResult PresentExecutionResult::completed(
        VkResult result, PresentResultOrigin origin) noexcept {
    return {PresentExecutionStatus::COMPLETED, result,
        origin, {}};
}

PresentExecutionResult PresentExecutionResult::pendingCompletion(VkResult result,
        std::unique_ptr<VirtualPresentPendingOperation> operation,
        PresentResultOrigin origin) noexcept {
    return {PresentExecutionStatus::PENDING, result,
        origin, std::move(operation)};
}

PresentExecutionResult PresentExecutionResult::failed(
        VkResult result, PresentResultOrigin origin) noexcept {
    return {PresentExecutionStatus::FAILED, result,
        origin, {}};
}

VkResult lsfgvk::layer::publicPresentResult(VkResult result,
        PresentResultOrigin origin, PresentTransactionPhase phase) noexcept {
    // Once the transaction crossed C1, waits may already have been consumed
    // by a bridge submit. An OOM result can no longer preserve Vulkan's
    // unchanged-state contract, regardless of downstream origin.
    if (phase == PresentTransactionPhase::POST_COMMIT
            && (result == VK_ERROR_OUT_OF_HOST_MEMORY
                || result == VK_ERROR_OUT_OF_DEVICE_MEMORY))
        return VK_ERROR_DEVICE_LOST;
    if (origin == PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT)
        return allowedPublicPresentResult(result) ? result : VK_ERROR_DEVICE_LOST;
    if (result == VK_SUCCESS)
        return VK_SUCCESS;
    if (result == VK_ERROR_DEVICE_LOST)
        return result;
    if (phase == PresentTransactionPhase::PRE_COMMIT
            && (result == VK_ERROR_OUT_OF_HOST_MEMORY
                || result == VK_ERROR_OUT_OF_DEVICE_MEMORY))
        return result;
    return VK_ERROR_DEVICE_LOST;
}

bool lsfgvk::layer::allowedPublicPresentResult(VkResult result) noexcept {
    return result != VK_NOT_READY && result != VK_TIMEOUT
        && result != VK_ERROR_INITIALIZATION_FAILED
        && result != VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult lsfgvk::layer::publicPrePresentGateResult(
        PrePresentGateResult result) noexcept {
    return result == PrePresentGateResult::READY
        ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

VkResult lsfgvk::layer::publicBridgePresentResult(VkResult result) noexcept {
    return result == VK_SUCCESS ? VK_SUCCESS
        : publicPresentResult(result, PresentResultOrigin::INTERNAL,
            PresentTransactionPhase::PRE_COMMIT);
}

bool VirtualPresentPendingSet::install(uint32_t imageIndex,
        uint64_t lifecycleIdentity,
        std::unique_ptr<VirtualPresentPendingOperation> operation,
        const std::shared_ptr<DeviceRetirementWakeTarget>& wakeTarget) noexcept {
    if (!operation || lifecycleIdentity == 0 || contains(imageIndex))
        return false;
    try {
        operation->activateCompletionWake(wakeTarget);
        entries.push_back(Entry{
            imageIndex, lifecycleIdentity, std::move(operation)});
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<VirtualPresentPendingSet::ReservedInstall>
VirtualPresentPendingSet::reserveInstall(
        uint32_t imageIndex, uint64_t lifecycleIdentity) noexcept {
    if (!lifecycleIdentity || contains(imageIndex)) return std::nullopt;
    try {
        ReservedInstall result;
        result.node.push_back(Entry{imageIndex, lifecycleIdentity, nullptr});
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

bool VirtualPresentPendingSet::installReserved(ReservedInstall& reserved,
        std::unique_ptr<VirtualPresentPendingOperation> operation,
        const std::shared_ptr<DeviceRetirementWakeTarget>& wakeTarget) noexcept {
    if (!operation || reserved.node.size() != 1) return false;
    auto& entry = reserved.node.front();
    if (!entry.lifecycleIdentity || contains(entry.imageIndex)) return false;
    operation->activateCompletionWake(wakeTarget);
    entry.operation = std::move(operation);
    entries.splice(entries.end(), reserved.node);
    return true;
}

bool VirtualPresentPendingSet::contains(uint32_t imageIndex) const noexcept {
    return std::ranges::any_of(entries, [imageIndex](const Entry& entry) {
        return entry.imageIndex == imageIndex;
    });
}

VirtualSwapchainRuntime::BatchPresentReservation::BatchPresentReservation(
        BatchPresentReservation&& other) noexcept
    : owner(other.owner), stateReservation(other.stateReservation),
      pendingReservation(std::move(other.pendingReservation)),
      backing(std::move(other.backing)), publication(std::move(other.publication)),
      bridgeCommitted(other.bridgeCommitted), terminal(other.terminal) {
    other.owner = nullptr;
    other.terminal = true;
}

VirtualSwapchainRuntime::BatchPresentReservation&
VirtualSwapchainRuntime::BatchPresentReservation::operator=(
        BatchPresentReservation&& other) noexcept {
    if (this == &other) return *this;
    if (valid()) static_cast<void>(abortCleanly());
    owner = other.owner;
    stateReservation = other.stateReservation;
    pendingReservation = std::move(other.pendingReservation);
    backing = std::move(other.backing);
    publication = std::move(other.publication);
    bridgeCommitted = other.bridgeCommitted;
    terminal = other.terminal;
    other.owner = nullptr;
    other.terminal = true;
    return *this;
}

VirtualSwapchainRuntime::BatchPresentReservation::~BatchPresentReservation() noexcept {
    if (valid()) static_cast<void>(abortCleanly());
}

bool VirtualSwapchainRuntime::BatchPresentReservation::valid() const noexcept {
    return owner && stateReservation.valid && pendingReservation.valid()
        && backing && publication && !terminal;
}

bool VirtualSwapchainRuntime::BatchPresentReservation::commitAfterBridge() noexcept {
    if (!valid() || bridgeCommitted
            || !owner->state.commitBatchPresent(stateReservation)) return false;
    bridgeCommitted = true;
    return true;
}

bool VirtualSwapchainRuntime::BatchPresentReservation::abortCleanly() noexcept {
    if (!valid()) return false;
    if (!owner->state.abortBatchPresent(stateReservation, bridgeCommitted))
        return false;
    terminal = true;
    backing.reset();
    publication.reset();
    return true;
}

bool VirtualSwapchainRuntime::BatchPresentReservation::
installPublishedCompletion() noexcept {
    if (!valid() || !bridgeCommitted || !publication || !*publication)
        return false;
    const std::scoped_lock lock(owner->pendingPresentsMutex);
    if (!owner->pendingPresents.installReserved(pendingReservation,
            std::move(*publication), owner->retirementWakeTarget)) return false;
    terminal = true;
    backing.reset();
    publication.reset();
    owner->state.wake();
    return true;
}

std::optional<VirtualSwapchainRuntime::BatchPresentReservation>
VirtualSwapchainRuntime::reserveBatchPresent(
        uint32_t imageIndex, VkResult* failure) noexcept {
    const auto fail = [&](VkResult result) {
        if (failure) *failure = result;
        return std::optional<BatchPresentReservation>{};
    };
    const auto pending = asyncResult.load();
    if (pending != VK_SUCCESS && pending != VK_SUBOPTIMAL_KHR)
        return fail(pending);
    if (prePresentGate) {
        try {
            if (prePresentGate() != PrePresentGateResult::READY)
                return fail(VK_ERROR_DEVICE_LOST);
        } catch (const std::bad_alloc&) {
            return fail(VK_ERROR_OUT_OF_HOST_MEMORY);
        } catch (...) {
            return fail(VK_ERROR_DEVICE_LOST);
        }
    }
    try {
        const auto serial = presentSerial.fetch_add(1);
        auto backing = imageBackings.at(imageIndex);
        auto publication = std::make_shared<
            std::unique_ptr<VirtualPresentPendingOperation>>();
        std::optional<VirtualPresentPendingSet::ReservedInstall> reserved;
        {
            const std::scoped_lock lock(pendingPresentsMutex);
            reserved = pendingPresents.reserveInstall(imageIndex, serial);
        }
        if (!reserved) return fail(VK_ERROR_DEVICE_LOST);
        auto stateToken = state.prepareBatchPresent(imageIndex, serial);
        if (!stateToken) return fail(VK_ERROR_DEVICE_LOST);
        BatchPresentReservation result;
        result.owner = this;
        result.stateReservation = *stateToken;
        result.backing = std::move(backing);
        result.publication = std::move(publication);
        result.pendingReservation = std::move(*reserved);
        if (failure) *failure = VK_SUCCESS;
        return result;
    } catch (const std::bad_alloc&) {
        return fail(VK_ERROR_OUT_OF_HOST_MEMORY);
    } catch (...) {
        return fail(VK_ERROR_DEVICE_LOST);
    }
}

void DeferredVirtualRetirementOwner::adopt(
        VirtualPresentPendingSet& source) noexcept {
    {
        const std::scoped_lock lock(mutex);
        const auto self = weak_from_this().lock();
        if (!self) std::terminate();
        for (auto& entry : source.entries)
            entry.operation->redirectCompletionWake(self);
        operations.splice(operations.end(), source.entries);
    }
    // Closes the completion-before-redirect race without polling any Vulkan
    // object; tryComplete observes only already-owned reactor tickets.
    notifyDeviceRetirement();
}

void DeferredVirtualRetirementOwner::notifyDeviceRetirement() noexcept {
    const std::scoped_lock lock(mutex);
    for (auto it = operations.begin(); it != operations.end();) {
        const auto status = it->operation->tryComplete();
        if (status == VirtualPresentCompletionStatus::RETIRED)
            it = operations.erase(it);
        else
            ++it;
    }
}

size_t DeferredVirtualRetirementOwner::pendingCount() const noexcept {
    const std::scoped_lock lock(mutex);
    return operations.size();
}

VirtualPresentCompletionStatus VirtualPresentPendingSet::tryCompleteOnce(
        VirtualSwapchainState& state) noexcept {
    bool retiredAny{};
    for (auto iterator = entries.begin(); iterator != entries.end();) {
        const auto status = iterator->operation->tryComplete();
        if (status == VirtualPresentCompletionStatus::NOT_READY) {
            ++iterator;
            continue;
        }
        if (status != VirtualPresentCompletionStatus::RETIRED)
            return status;
        if (!state.complete(iterator->imageIndex))
            return VirtualPresentCompletionStatus::FAILED;
        iterator = entries.erase(iterator);
        retiredAny = true;
    }
    return retiredAny ? VirtualPresentCompletionStatus::RETIRED
                      : VirtualPresentCompletionStatus::NOT_READY;
}

VirtualSwapchainRuntime::VirtualSwapchainRuntime(const vk::Vulkan& vk,
        VkQueue offloadQueue,
        std::shared_ptr<std::mutex> offloadMutex,
        size_t imageCount,
        const VirtualSwapchainImageSpec& spec,
        bool d3b2ReleaseCapable) :
        vk(std::cref(vk)),
        offloadQueue(offloadQueue),
        offloadMutex(std::move(offloadMutex)),
        state(imageCount),
        d3b2ReleaseCapable(d3b2ReleaseCapable) {
    if (this->offloadQueue == VK_NULL_HANDLE || !this->offloadMutex)
        throw ls::error("virtual swapchain requires a dedicated graphics queue");
    if (!spec.supported())
        throw ls::error("virtual swapchain image specification is not supported");
    this->retirementWakeTarget =
        std::make_shared<RuntimeRetirementWakeTarget>(this->state);

    this->imageBackings.reserve(imageCount);
    for (size_t i = 0; i < imageCount; ++i) {
        auto formatList = spec.makeFormatListInfo();
        const void* pNext = spec.hasFormatList ? &formatList : nullptr;

        const auto options = spec.imageOptions(pNext);
        this->imageBackings.push_back(
            std::make_shared<VirtualImageBacking>(vk, spec, options));
    }
}

VirtualSwapchainRuntime::~VirtualSwapchainRuntime() {
    this->stop();
}

VkResult VirtualSwapchainRuntime::getImages(uint32_t* count, VkImage* images) const noexcept {
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!images) {
        *count = static_cast<uint32_t>(this->imageBackings.size());
        return VK_SUCCESS;
    }

    const auto requested = *count;
    const auto available = static_cast<uint32_t>(this->imageBackings.size());
    const auto copied = std::min(requested, available);

    for (uint32_t i = 0; i < copied; ++i)
        images[i] = this->imageBackings.at(i)->image.handle();

    *count = copied;
    return requested < available ? VK_INCOMPLETE : VK_SUCCESS;
}

VkResult VirtualSwapchainRuntime::signalAcquire(
        VkSemaphore semaphore, VkFence fence) const noexcept {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = semaphore == VK_NULL_HANDLE ? 0U : 1U;
    submitInfo.pSignalSemaphores = semaphore == VK_NULL_HANDLE ? nullptr : &semaphore;

    std::scoped_lock lock(*this->offloadMutex);
    return this->vk.get().df().QueueSubmit(
        this->offloadQueue, 1, &submitInfo, fence);
}

VkResult VirtualSwapchainRuntime::acquire(uint64_t timeout,
        VkSemaphore semaphore, VkFence fence, uint32_t* imageIndex) noexcept {
    if (!imageIndex)
        return VK_ERROR_INITIALIZATION_FAILED;

    const auto pending = this->asyncResult.load();
    if (pending != VK_SUCCESS && pending != VK_SUBOPTIMAL_KHR)
        return pending;

    std::optional<uint32_t> acquired;
    if (timeout == 0) {
        acquired = this->state.tryAcquire();
    } else if (timeout == UINT64_MAX) {
        acquired = this->state.waitAcquire(VirtualSwapchainState::Duration::max());
    } else {
        using Rep = VirtualSwapchainState::Duration::rep;
        constexpr auto maxRep = static_cast<uint64_t>(std::numeric_limits<Rep>::max());
        const auto clamped = std::min(timeout, maxRep);
        acquired = this->state.waitAcquire(
            VirtualSwapchainState::Duration(static_cast<Rep>(clamped)));
    }

    if (!acquired.has_value()) {
        const auto result = this->asyncResult.load();
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            return result;
        if (this->state.stopped())
            return VK_ERROR_OUT_OF_DATE_KHR;
        return timeout == 0 ? VK_NOT_READY : VK_TIMEOUT;
    }

    const auto res = this->signalAcquire(semaphore, fence);
    if (res != VK_SUCCESS) {
        (void)this->state.release(*acquired);
        return res;
    }

    *imageIndex = *acquired;
    return pending == VK_SUBOPTIMAL_KHR ? VK_SUBOPTIMAL_KHR : VK_SUCCESS;
}

void VirtualSwapchainRuntime::startWorker(Presenter presenter) {
    if (!presenter)
        throw ls::error("virtual swapchain worker requires a presenter");
    if (this->worker.joinable())
        throw ls::error("virtual swapchain worker already started");

    this->presenter = std::move(presenter);
    this->stopping.store(false);
    this->worker = std::jthread([this](std::stop_token stopToken) {
        this->workerLoop(stopToken);
    });
}

void VirtualSwapchainRuntime::setPrePresentGate(PrePresentGate gate) {
    if (this->worker.joinable())
        throw ls::error("virtual swapchain pre-present gate must be installed before worker start");
    this->prePresentGate = std::move(gate);
}

VkResult VirtualSwapchainRuntime::bridgePresentWaits(VkQueue sourceQueue,
        uint32_t imageIndex, uint64_t epoch,
        const std::vector<VkSemaphore>& waitSemaphores) const noexcept {
    if (sourceQueue == VK_NULL_HANDLE || imageIndex >= this->imageBackings.size())
        return VK_ERROR_OUT_OF_DATE_KHR;

    try {
        const auto& backing = this->imageBackings.at(imageIndex);
        const auto ready = backing->ready.handle();
        const auto originalReady = backing->originalReady.handle();
        const bool d3b2 = [] {
            const char* value = std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC");
            return value && std::strcmp(value, "1") == 0;
        }();
        const std::vector<VkSemaphore> signals = d3b2
            ? std::vector<VkSemaphore>{ready, originalReady}
            : std::vector<VkSemaphore>{ready};

        // C0: all allocations above occur before QueueSubmit acceptance.
        ApplicationPresentWaitAuthority input(waitSemaphores, epoch,
            reinterpret_cast<uintptr_t>(this));
        return bridgeApplicationPresentWaits(input, sourceQueue, signals,
            d3b2 ? originalReady : VK_NULL_HANDLE,
            [this](VkQueue queue, const VkSubmitInfo& submit) {
                return this->vk.get().df().QueueSubmit(
                    queue, 1, &submit, VK_NULL_HANDLE);
            }).result;
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

VkResult VirtualSwapchainRuntime::queuePresent(VkQueue sourceQueue,
        uint32_t sourceQueueFamily,
        uint32_t sourceQueueIndex,
        VkQueueFlags sourceQueueFlags,
        bool surfacePresentSupported,
        uint32_t imageIndex,
        const std::vector<VkSemaphore>& waitSemaphores,
        void* nextChain,
        bool synchronous,
        PresentedPhysicalImageIdentity* presentedIdentity) noexcept {
    if (!this->worker.joinable() || !this->presenter || this->stopping.load()) {
        const auto result = this->asyncResult.load();
        return result == VK_SUCCESS ? VK_ERROR_DEVICE_LOST : result;
    }

    const auto pending = this->asyncResult.load();
    if (pending != VK_SUCCESS && pending != VK_SUBOPTIMAL_KHR)
        return pending;

    // This is deliberately before both lease construction and bridge submission.
    // D3B2/D3B1 leave the gate empty, preserving their existing ordering.
    if (this->prePresentGate) {
        PrePresentGateResult gate{};
        try {
            gate = this->prePresentGate();
        } catch (const std::bad_alloc&) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        } catch (...) {
            return VK_ERROR_DEVICE_LOST;
        }
        if (gate != PrePresentGateResult::READY)
            return publicPrePresentGateResult(gate);
    }

    const auto serial = this->presentSerial.fetch_add(1);
    const bool d3bSingleSwapchainEligible = !synchronous;
    const char* d3b2 = std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC");
    if (d3bSingleSwapchainEligible && d3b2 && std::strcmp(d3b2, "1") == 0
            && !this->d3b2ReleaseCapable)
        return VK_ERROR_DEVICE_LOST;
    // A borrowed application queue is valid only while the intercepted public
    // present remains open. Every borrowed graphics-final operation is thus
    // synchronous, including ordinary virtual-final blits.
    const bool borrowedSynchronous = true;
    const GraphicsFinalQueueInfo graphicsFinalQueue{
        .queue = sourceQueue,
        .family = sourceQueueFamily,
        .index = sourceQueueIndex,
        .flags = sourceQueueFlags,
        .surfacePresentSupported = surfacePresentSupported,
        .borrowed = true};
    if (graphicsFinalExecutionMode(graphicsFinalQueue, borrowedSynchronous)
            != GraphicsFinalExecutionMode::BORROWED_SYNCHRONOUS)
        return VK_ERROR_DEVICE_LOST;

    std::optional<VirtualSwapchainState::PreparedPresent> prepared;
    std::shared_ptr<Completion> completion;
    std::shared_ptr<BorrowedGraphicsQueueLease> lease;
    std::shared_ptr<PresentedPhysicalImageIdentity> installedIdentity;
    try {
        prepared = this->state.preparePresent(imageIndex, serial);
        if (!prepared)
            return VK_ERROR_DEVICE_LOST;
        completion = std::make_shared<Completion>();
        lease = std::make_shared<BorrowedGraphicsQueueLease>(
            sourceQueue, sourceQueueFamily, sourceQueueIndex, serial);
        installedIdentity = std::make_shared<PresentedPhysicalImageIdentity>();
        const auto imageBacking = this->imageBackings.at(imageIndex);
        const std::scoped_lock lock(this->jobsMutex);
        const auto [it, inserted] = this->jobs.emplace(serial, Job {
            .nextChain = nextChain,
            .sourcePresentTime = std::chrono::steady_clock::now(),
            .completion = completion,
            .graphicsFinalQueue = graphicsFinalQueue,
            .borrowedLease = lease,
            .presentedIdentity = installedIdentity,
            .imageBacking = imageBacking,
            .d3bSingleSwapchainEligible = d3bSingleSwapchainEligible
        });
        (void)it;
        if (!inserted)
            return VK_ERROR_DEVICE_LOST;
    } catch (const std::bad_alloc&) {
        if (lease) (void)lease->release();
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
        if (lease) (void)lease->release();
        return VK_ERROR_DEVICE_LOST;
    }

    // Queue the application's wait semaphores before publishing the job. This
    // preserves vkQueuePresentKHR semaphore-consumption semantics while still
    // allowing the CPU call to return before WSI presentation is performed.
    const auto bridge = this->bridgePresentWaits(
        sourceQueue, imageIndex, serial, waitSemaphores);
    if (bridge != VK_SUCCESS) {
        {
            const std::scoped_lock lock(this->jobsMutex);
            this->jobs.erase(serial);
        }
        (void)this->state.release(imageIndex);
        (void)lease->release();
        return publicBridgePresentResult(bridge);
    }

    // C1: QueueSubmit accepted the application waits. Everything needed for
    // worker publication is already owned, and splice cannot allocate.
    if (!this->state.commitPreparedPresent(*prepared)) {
        {
            const std::scoped_lock lock(this->jobsMutex);
            this->jobs.erase(serial);
        }
        this->asyncResult.store(VK_ERROR_DEVICE_LOST);
        this->stopping.store(true);
        this->state.stop();
        (void)lease->release();
        this->finishCompletion(completion, VK_ERROR_DEVICE_LOST);
        return VK_ERROR_DEVICE_LOST;
    }

    std::unique_lock lock(completion->mutex);
    completion->cv.wait(lock, [&]() {
        return completion->done || this->stopping.load();
    });
    if (!completion->done) {
        const auto result = this->asyncResult.load();
        return result == VK_SUCCESS ? VK_ERROR_DEVICE_LOST : result;
    }
    if (presentedIdentity) *presentedIdentity = *installedIdentity;
    return completion->result;
}

void VirtualSwapchainRuntime::finishCompletion(
        const std::shared_ptr<Completion>& completion,
        VkResult result) noexcept {
    if (!completion)
        return;

    {
        const std::scoped_lock lock(completion->mutex);
        completion->result = result;
        completion->done = true;
    }
    completion->cv.notify_all();
}

void VirtualSwapchainRuntime::workerLoop(std::stop_token stopToken) noexcept {
    while (!stopToken.stop_requested()) {
        // Capture the wake generation before testing pending tickets so a
        // reactor notification racing with the transition into sleep cannot be
        // lost.  Completion wakeups are strictly event driven; no timer polls
        // GPU fences in the qualified worker path.
        const auto wakeGeneration = this->state.wakeGeneration();
        const auto pendingStatus = this->tryCompletePendingPresents();
        if (pendingStatus == VirtualPresentCompletionStatus::DEVICE_LOST
                || pendingStatus == VirtualPresentCompletionStatus::FAILED) {
            const auto failure = pendingStatus == VirtualPresentCompletionStatus::DEVICE_LOST
                ? VK_ERROR_DEVICE_LOST : VK_ERROR_UNKNOWN;
            this->asyncResult.store(failure);
            this->stopping.store(true);
            this->state.stop();
            this->failPending(failure);
            break;
        }

        bool hasPending{};
        {
            const std::scoped_lock lock(this->pendingPresentsMutex);
            hasPending = !this->pendingPresents.empty();
        }
        if (hasPending) {
            this->state.waitForWake(
                wakeGeneration, VirtualSwapchainState::Duration::max());
            continue;
        }
        const auto present = this->state.waitPresent(
            VirtualSwapchainState::Duration::max());
        if (!present.has_value()) {
            if (this->stopping.load() || stopToken.stop_requested()) break;
            continue;
        }

        Job job{};
        {
            const std::scoped_lock lock(this->jobsMutex);
            const auto it = this->jobs.find(present->serial);
            if (it == this->jobs.end()) {
                this->asyncResult.store(VK_ERROR_DEVICE_LOST);
                this->stopping.store(true);
                this->state.stop();
                break;
            }
            job = std::move(it->second);
            this->jobs.erase(it);
        }

        PresentExecutionResult execution = PresentExecutionResult::failed(VK_ERROR_UNKNOWN);
        bool stopAfterCompletion{};
        try {
            if (!job.borrowedLease
                    || !job.borrowedLease->validFor(
                        job.graphicsFinalQueue.queue, job.graphicsFinalQueue.family))
                throw ls::error("borrowed graphics queue lease is not active");
            execution = this->presenter(
                present->imageIndex,
                job.imageBacking->ready.handle(),
                job.imageBacking->originalReady.handle(),
                job.nextChain,
                stopToken,
                job.sourcePresentTime,
                job.d3bSingleSwapchainEligible,
                job.graphicsFinalQueue,
                *job.borrowedLease,
                stopAfterCompletion,
                job.presentedIdentity.get(), job.imageBacking);
        } catch (...) {
            execution = PresentExecutionResult::failed(VK_ERROR_UNKNOWN);
        }

        auto result = publicPresentResult(execution.result, execution.origin,
            PresentTransactionPhase::POST_COMMIT);
        const bool apiAccepted = result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
        if (execution.status == PresentExecutionStatus::COMPLETED && apiAccepted) {
            if (!this->state.complete(present->imageIndex))
                result = VK_ERROR_DEVICE_LOST;
            else if (result == VK_SUBOPTIMAL_KHR)
                this->asyncResult.store(result);
        } else if (execution.status == PresentExecutionStatus::PENDING
                && apiAccepted && execution.pending
                && [&] {
                    const std::scoped_lock lock(this->pendingPresentsMutex);
                    return this->pendingPresents.install(present->imageIndex,
                        present->serial, std::move(execution.pending),
                        this->retirementWakeTarget);
                }()) {
        } else {
            if (apiAccepted) result = VK_ERROR_DEVICE_LOST;
            this->asyncResult.store(result);
            this->stopping.store(true);
            this->state.stop();
        }

        if (!job.borrowedLease || !job.borrowedLease->release())
            result = VK_ERROR_DEVICE_LOST;
        this->finishCompletion(job.completion, result);
        if ((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
                && stopAfterCompletion) {
            this->stopping.store(true);
            this->state.stop();
            this->failPending(VK_ERROR_OUT_OF_DATE_KHR);
            break;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            break;
    }
}

VirtualPresentCompletionStatus
VirtualSwapchainRuntime::tryCompletePendingPresents() noexcept {
    const std::scoped_lock lock(this->pendingPresentsMutex);
    return pendingPresents.tryCompleteOnce(state);
}

void VirtualSwapchainRuntime::failPending(VkResult result) noexcept {
    std::vector<std::shared_ptr<Completion>> completions;
    {
        const std::scoped_lock lock(this->jobsMutex);
        completions.reserve(this->jobs.size());
        for (auto& [serial, job] : this->jobs) {
            (void)serial;
            if (job.completion)
                completions.push_back(std::move(job.completion));
        }
        this->jobs.clear();
    }

    for (const auto& completion : completions)
        this->finishCompletion(completion, result);
}

void VirtualSwapchainRuntime::stop() noexcept {
    this->stopping.store(true);
    this->state.stop();
    if (this->worker.joinable()) {
        this->worker.request_stop();
        if (this->worker.get_id() != std::this_thread::get_id())
            this->worker.join();
    }

    if (const auto target = std::dynamic_pointer_cast<RuntimeRetirementWakeTarget>(
            this->retirementWakeTarget))
        target->detach();

    auto result = this->asyncResult.load();
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
        result = VK_ERROR_OUT_OF_DATE_KHR;
    this->failPending(result);
}

bool VirtualSwapchainRuntime::stopAndDetach(
        const std::shared_ptr<DeferredVirtualRetirementOwner>& owner) noexcept {
    this->stop();
    const std::scoped_lock lock(this->pendingPresentsMutex);
    if (this->pendingPresents.empty()) return true;
    if (!owner) return false;
    owner->adopt(this->pendingPresents);
    return true;
}

bool VirtualSwapchainRuntime::installPendingForTesting(
    uint32_t imageIndex, uint64_t lifecycleIdentity,
        std::unique_ptr<VirtualPresentPendingOperation> operation) noexcept {
    const std::scoped_lock lock(this->pendingPresentsMutex);
    return this->pendingPresents.install(
        imageIndex, lifecycleIdentity, std::move(operation));
}

std::vector<VkImage> VirtualSwapchainRuntime::imageHandles() const {
    std::vector<VkImage> handles;
    handles.reserve(this->imageBackings.size());
    for (const auto& backing : this->imageBackings)
        handles.push_back(backing->image.handle());
    return handles;
}

bool VirtualSwapchainRuntime::workerRunning() const noexcept {
    return this->worker.joinable() && !this->stopping.load();
}

VkResult VirtualSwapchainRuntime::workerResult() const noexcept {
    return this->asyncResult.load();
}
