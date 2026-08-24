/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "virtual_swapchain_runtime.hpp"
#include "d3b1_present_path.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include <array>
#include <cstring>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

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

    this->images.reserve(imageCount);
    this->readySemaphores.reserve(imageCount);
    this->originalReadySemaphores.reserve(imageCount);
    for (size_t i = 0; i < imageCount; ++i) {
        auto formatList = spec.makeFormatListInfo();
        const void* pNext = spec.hasFormatList ? &formatList : nullptr;

        this->images.emplace_back(vk,
            spec.extent,
            spec.format,
            spec.usage,
            spec.imageOptions(pNext));
        this->readySemaphores.emplace_back(vk);
        this->originalReadySemaphores.emplace_back(vk);
    }
}

VirtualSwapchainRuntime::~VirtualSwapchainRuntime() {
    this->stop();
}

VkResult VirtualSwapchainRuntime::getImages(uint32_t* count, VkImage* images) const noexcept {
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!images) {
        *count = static_cast<uint32_t>(this->images.size());
        return VK_SUCCESS;
    }

    const auto requested = *count;
    const auto available = static_cast<uint32_t>(this->images.size());
    const auto copied = std::min(requested, available);

    for (uint32_t i = 0; i < copied; ++i)
        images[i] = this->images.at(i).handle();

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
        uint32_t imageIndex,
        const std::vector<VkSemaphore>& waitSemaphores) const noexcept {
    if (sourceQueue == VK_NULL_HANDLE || imageIndex >= this->readySemaphores.size())
        return VK_ERROR_OUT_OF_DATE_KHR;

    std::vector<VkPipelineStageFlags> stages(
        waitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    const auto ready = this->readySemaphores.at(imageIndex).handle();
    const auto originalReady = this->originalReadySemaphores.at(imageIndex).handle();
    const bool d3b2 = [] {
        const char* value = std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
    }();
    const std::array<VkSemaphore, 2> signals{ready, originalReady};
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.empty() ? nullptr : waitSemaphores.data(),
        .pWaitDstStageMask = stages.empty() ? nullptr : stages.data(),
        .commandBufferCount = 0,
        .pCommandBuffers = nullptr,
        .signalSemaphoreCount = d3b2 ? 2U : 1U,
        .pSignalSemaphores = signals.data()
    };

    // vkQueuePresentKHR requires host access to its queue to be externally
    // synchronized by the caller. The layer is executing inside that call, so
    // this bridge submit can safely consume the same present waits on the
    // application's source queue before returning control to the application.
    return this->vk.get().df().QueueSubmit(
        sourceQueue, 1, &submitInfo, VK_NULL_HANDLE);
}

VkResult VirtualSwapchainRuntime::queuePresent(VkQueue sourceQueue,
        uint32_t sourceQueueFamily,
        uint32_t sourceQueueIndex,
        VkQueueFlags sourceQueueFlags,
        bool surfacePresentSupported,
        uint32_t imageIndex,
        const std::vector<VkSemaphore>& waitSemaphores,
        void* nextChain,
        bool synchronous) noexcept {
    if (!this->worker.joinable() || !this->presenter || this->stopping.load()) {
        const auto result = this->asyncResult.load();
        return result == VK_SUCCESS ? VK_ERROR_OUT_OF_DATE_KHR : result;
    }

    const auto pending = this->asyncResult.load();
    if (pending != VK_SUCCESS && pending != VK_SUBOPTIMAL_KHR)
        return pending;

    // This is deliberately before both lease construction and bridge submission.
    // D3B2/D3B1 leave the gate empty, preserving their existing ordering.
    if (this->prePresentGate) {
        const auto gate = this->prePresentGate();
        if (gate != PrePresentGateResult::READY) {
            switch (gate) {
            case PrePresentGateResult::TIMEOUT: return VK_TIMEOUT;
            case PrePresentGateResult::DEVICE_LOST: return VK_ERROR_DEVICE_LOST;
            case PrePresentGateResult::FAILED: return VK_ERROR_INITIALIZATION_FAILED;
            case PrePresentGateResult::READY: break;
            }
        }
    }

    const auto serial = this->presentSerial.fetch_add(1);
    const bool d3bSingleSwapchainEligible = !synchronous;
    const char* d3b2 = std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC");
    if (d3bSingleSwapchainEligible && d3b2 && std::strcmp(d3b2, "1") == 0
            && !this->d3b2ReleaseCapable)
        return VK_ERROR_FEATURE_NOT_PRESENT;
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
        return VK_ERROR_FEATURE_NOT_PRESENT;
    auto completion = std::make_shared<Completion>();
    auto lease = std::make_shared<BorrowedGraphicsQueueLease>(
        sourceQueue, sourceQueueFamily, sourceQueueIndex, serial);
    {
        const std::scoped_lock lock(this->jobsMutex);
        this->jobs.emplace(serial, Job {
            .nextChain = nextChain,
            .sourcePresentTime = std::chrono::steady_clock::now(),
            .completion = completion,
            .graphicsFinalQueue = graphicsFinalQueue,
            .borrowedLease = lease,
            .d3bSingleSwapchainEligible = d3bSingleSwapchainEligible
        });
    }

    // Queue the application's wait semaphores before publishing the job. This
    // preserves vkQueuePresentKHR semaphore-consumption semantics while still
    // allowing the CPU call to return before WSI presentation is performed.
    const auto bridge = this->bridgePresentWaits(
        sourceQueue, imageIndex, waitSemaphores);
    if (bridge != VK_SUCCESS) {
        {
            const std::scoped_lock lock(this->jobsMutex);
            this->jobs.erase(serial);
        }
        (void)this->state.release(imageIndex);
        (void)lease->release();
        return bridge;
    }

    if (!this->state.queuePresent(imageIndex, serial)) {
        {
            const std::scoped_lock lock(this->jobsMutex);
            this->jobs.erase(serial);
        }
        this->asyncResult.store(VK_ERROR_OUT_OF_DATE_KHR);
        this->stopping.store(true);
        this->state.stop();
        (void)lease->release();
        this->finishCompletion(completion, VK_ERROR_OUT_OF_DATE_KHR);
        return VK_ERROR_OUT_OF_DATE_KHR;
    }

    std::unique_lock lock(completion->mutex);
    completion->cv.wait(lock, [&]() {
        return completion->done || this->stopping.load();
    });
    if (!completion->done) {
        const auto result = this->asyncResult.load();
        return result == VK_SUCCESS ? VK_ERROR_OUT_OF_DATE_KHR : result;
    }
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
        const auto present = this->state.waitPresent(
            VirtualSwapchainState::Duration::max());
        if (!present.has_value())
            break;

        Job job{};
        {
            const std::scoped_lock lock(this->jobsMutex);
            const auto it = this->jobs.find(present->serial);
            if (it == this->jobs.end()) {
                this->asyncResult.store(VK_ERROR_UNKNOWN);
                this->stopping.store(true);
                this->state.stop();
                break;
            }
            job = std::move(it->second);
            this->jobs.erase(it);
        }

        VkResult result{VK_ERROR_UNKNOWN};
        bool stopAfterCompletion{};
        try {
            if (!job.borrowedLease
                    || !job.borrowedLease->validFor(
                        job.graphicsFinalQueue.queue, job.graphicsFinalQueue.family))
                throw ls::error("borrowed graphics queue lease is not active");
            result = this->presenter(
                present->imageIndex,
                this->readySemaphores.at(present->imageIndex).handle(),
                this->originalReadySemaphores.at(present->imageIndex).handle(),
                job.nextChain,
                stopToken,
                job.sourcePresentTime,
                job.d3bSingleSwapchainEligible,
                job.graphicsFinalQueue,
                *job.borrowedLease,
                stopAfterCompletion);
        } catch (...) {
            result = VK_ERROR_UNKNOWN;
        }

        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            (void)this->state.complete(present->imageIndex);
            if (result == VK_SUBOPTIMAL_KHR)
                this->asyncResult.store(result);
        } else {
            this->asyncResult.store(result);
            this->stopping.store(true);
            this->state.stop();
        }

        if (!job.borrowedLease || !job.borrowedLease->release())
            result = VK_ERROR_UNKNOWN;
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

    auto result = this->asyncResult.load();
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
        result = VK_ERROR_OUT_OF_DATE_KHR;
    this->failPending(result);
}

std::vector<VkImage> VirtualSwapchainRuntime::imageHandles() const {
    std::vector<VkImage> handles;
    handles.reserve(this->images.size());
    for (const auto& image : this->images)
        handles.push_back(image.handle());
    return handles;
}

bool VirtualSwapchainRuntime::workerRunning() const noexcept {
    return this->worker.joinable() && !this->stopping.load();
}

VkResult VirtualSwapchainRuntime::workerResult() const noexcept {
    return this->asyncResult.load();
}
