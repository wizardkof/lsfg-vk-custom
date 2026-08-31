/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "swapchain.hpp"
#include "aborted_present_semantics.hpp"
#include "adaptive_1x_prepared_logical_final.hpp"
#include "adaptive_1x_preparation_reservation.hpp"
#include "adaptive_1x_recovery_authority.hpp"
#include "borrowed_present_fence_registry.hpp"
#include "present_batch_projection.hpp"
#include "device_retirement_reactor.hpp"
#include "sync_file_completion.hpp"
#include "owned_present_sync_fd.hpp"
#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "terminal_pending_gpu_operation.hpp"
#include "virtual_swapchain_runtime.hpp"
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
#include "entrypoint_test_seam.hpp"
#endif
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>
#include <poll.h>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk;
using namespace lsfgvk::layer;

namespace lsfgvk::layer {
namespace {
std::atomic<uint64_t> nextProductionSwapchainLifecycleIdentity{1};
std::atomic<uint64_t> nextRetirementOperationIdentity{1};
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
std::atomic<void (*)()> adaptiveReservationBeforeExecute{};
std::atomic<void (*)()> adaptiveReservationAfterExecute{};
std::atomic<test::AdaptiveBuilderFailurePoint> adaptiveBuilderFailurePoint{};
std::atomic<test::AdaptiveExecuteFailurePoint> adaptiveExecuteFailurePoint{};
std::atomic_uint32_t adaptiveCommandPoolCreates{};
std::atomic_uint32_t adaptiveCommandPoolDestroys{};
std::mutex adaptiveCommandPoolLifecycleMutex;
std::condition_variable adaptiveCommandPoolLifecycleCv;

bool consumeAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint point) noexcept {
    auto expected = point;
    return adaptiveBuilderFailurePoint.compare_exchange_strong(expected,
        test::AdaptiveBuilderFailurePoint::None, std::memory_order_acq_rel);
}

void failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint point) {
    if (consumeAdaptiveBuilderAt(point)) throw std::bad_alloc();
}

bool failAdaptiveExecuteAt(test::AdaptiveExecuteFailurePoint point) noexcept {
    auto expected = point;
    return adaptiveExecuteFailurePoint.compare_exchange_strong(expected,
        test::AdaptiveExecuteFailurePoint::None, std::memory_order_acq_rel);
}
#endif

struct PresentedWsiSemaphoreBacking {
    enum class CompletionAuthority : uint8_t {
        UnexportedFence,
        ExportedSyncFd,
        ReactorOwnedSyncFd,
        Completed,
        Terminal
    };

    explicit PresentedWsiSemaphoreBacking(const vk::Vulkan& vk)
        : ready(vk), internalPresentFence(
            vk, vk::Fence::ExternalHandle::SyncFd) {}
    void acceptExportedFd(int fd) noexcept {
        const std::scoped_lock lock(completionMutex);
        teardownFd.accept(fd);
        completionAuthority = fd == -1
            ? CompletionAuthority::Completed
            : CompletionAuthority::ExportedSyncFd;
    }

    void markReactorOwned() noexcept {
        const std::scoped_lock lock(completionMutex);
        if (completionAuthority == CompletionAuthority::ExportedSyncFd)
            completionAuthority = CompletionAuthority::ReactorOwnedSyncFd;
    }

    void markCompleted() noexcept {
        const std::scoped_lock lock(completionMutex);
        completionAuthority = CompletionAuthority::Completed;
    }

    [[nodiscard]] VkResult waitForCompletion(
            VkDevice device, PFN_vkWaitForFences waitForFences) noexcept {
        int fd{-1};
        CompletionAuthority authority{};
        {
            const std::scoped_lock lock(completionMutex);
            authority = completionAuthority;
            fd = teardownFd.snapshot();
        }
        if (authority == CompletionAuthority::Completed) return VK_SUCCESS;
        if (authority == CompletionAuthority::ExportedSyncFd
                || authority == CompletionAuthority::ReactorOwnedSyncFd) {
            if (fd < 0) return VK_ERROR_UNKNOWN;
            pollfd descriptor{.fd = fd, .events = POLLIN};
            int result{};
            do { result = ::poll(&descriptor, 1, -1); }
            while (result < 0 && errno == EINTR);
            if (result <= 0 || (descriptor.revents & POLLNVAL))
                return VK_ERROR_UNKNOWN;
            switch (querySyncFileCompletion(fd)) {
            case SyncFileCompletionState::Completed:
                markCompleted();
                return VK_SUCCESS;
            case SyncFileCompletionState::Pending:
                return VK_NOT_READY;
            case SyncFileCompletionState::GpuError:
                return VK_ERROR_DEVICE_LOST;
            case SyncFileCompletionState::FailedPermanent:
                return VK_ERROR_UNKNOWN;
            }
            return VK_ERROR_UNKNOWN;
        }
        if (authority != CompletionAuthority::UnexportedFence)
            return VK_ERROR_DEVICE_LOST;
        if (!waitForFences) return VK_ERROR_EXTENSION_NOT_PRESENT;
        const auto fence = internalPresentFence.handle();
        return waitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    [[nodiscard]] VkResult waitForCompletion(const vk::Vulkan& vk) noexcept {
        return waitForCompletion(vk.dev(), vk.df().WaitForFences);
    }

    vk::Semaphore ready;
    vk::Fence internalPresentFence;
    std::mutex completionMutex;
    OwnedPresentSyncFd teardownFd;
    CompletionAuthority completionAuthority{CompletionAuthority::UnexportedFence};
};

struct AdaptiveAcquireSyncBacking final {
    explicit AdaptiveAcquireSyncBacking(const vk::Vulkan& vk)
        : semaphore(vk), fence(vk) {}
    vk::Semaphore semaphore;
    vk::Fence fence;
};

struct AdaptiveProducerSyncBacking final {
    explicit AdaptiveProducerSyncBacking(const vk::Vulkan& vk)
        : fence(vk, vk::Fence::ExternalHandle::SyncFd) {}
    vk::Fence fence;
};

struct AdaptiveProducerCommandBacking final {
    AdaptiveProducerCommandBacking(const vk::Vulkan& vk, uint32_t queueFamily) {
        const VkCommandPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = queueFamily};
        VkCommandPool handle{};
        const auto result = vk.df().CreateCommandPool(
            vk.dev(), &info, nullptr, &handle);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result,
                "Adaptive producer command-pool creation failed");
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        adaptiveCommandPoolCreates.fetch_add(1, std::memory_order_relaxed);
#endif
        pool = ls::owned_ptr<VkCommandPool>(new VkCommandPool(handle),
            [device = vk.dev(), destroy = vk.df().DestroyCommandPool](
                    VkCommandPool& value) {
                destroy(device, value, nullptr);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
                adaptiveCommandPoolDestroys.fetch_add(1,
                    std::memory_order_relaxed);
                adaptiveCommandPoolLifecycleCv.notify_all();
#endif
            });
        commandBuffer.emplace(vk, pool.get());
    }

    [[nodiscard]] VkCommandPool commandPool() const noexcept {
        return pool.get();
    }

    // Reverse member destruction frees the command buffer before its pool.
    ls::owned_ptr<VkCommandPool> pool;
    ls::lazy<vk::CommandBuffer> commandBuffer;
};

struct OwnedInternalPresentFenceProjection {
    VkFence fence{VK_NULL_HANDLE};
    VkSwapchainPresentFenceInfoKHR injected{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR};
    std::optional<PresentPNextProjection> copied;
    const void* head{};

    [[nodiscard]] bool prepare(const void* applicationChain,
            VkFence internalFence) noexcept {
        fence = internalFence;
        auto* current = reinterpret_cast<const VkBaseInStructure*>(applicationChain);
        bool hasFenceNode{};
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR) {
                hasFenceNode = true;
                break;
            }
            current = current->pNext;
        }
        if (!hasFenceNode) {
            injected.pNext = applicationChain;
            injected.swapchainCount = 1;
            injected.pFences = &fence;
            head = &injected;
            return true;
        }

        copied.emplace(PresentPNextProjection::build(applicationChain, 1, 0));
        if (!copied->supported()) return false;
        auto* owned = reinterpret_cast<VkBaseOutStructure*>(copied->head());
        while (owned) {
            if (owned->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR) {
                auto* info = reinterpret_cast<VkSwapchainPresentFenceInfoKHR*>(owned);
                info->swapchainCount = 1;
                info->pFences = &fence;
                head = copied->head();
                return true;
            }
            owned = owned->pNext;
        }
        return false;
    }
};

struct CompositeGpuRetirementBacking final {
    std::shared_ptr<void> operation;
    std::shared_ptr<void> dependency;
};

std::shared_ptr<void> combineGpuRetirementBacking(
        std::shared_ptr<void> operation, std::shared_ptr<void> dependency) {
    if (!dependency) return operation;
    return std::make_shared<CompositeGpuRetirementBacking>(
        CompositeGpuRetirementBacking{std::move(operation), std::move(dependency)});
}

[[nodiscard]] bool armInternalPresentFence(
        VkDevice device, PFN_vkGetFenceFdKHR getFenceFd,
        const std::shared_ptr<PresentedWsiSemaphoreBacking>& backing,
        const std::shared_ptr<PresentedPhysicalImageLeaseRegistry>& leases,
        const PresentedPhysicalImageIdentity& identity,
        const std::shared_ptr<DeviceRetirementReactor>& reactor) noexcept {
    if (!backing || !leases || !identity.valid() || !reactor) return false;
    int reactorFd{-1};
    try {
        if (!getFenceFd) return false;
        const VkFenceGetFdInfoKHR exportInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
            .fence = backing->internalPresentFence.handle(),
            .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
        int exportedFd{-1};
        if (getFenceFd(device, &exportInfo, &exportedFd) != VK_SUCCESS)
            return false;
        backing->acceptExportedFd(exportedFd);
        if (exportedFd >= 0) {
            reactorFd = ::dup(exportedFd);
            if (reactorFd < 0) return false;
        }
        auto ticket = std::make_shared<DeviceRetirementTicket>();
        DeviceRetirementJob job{
            .deviceIdentity = identity.deviceLifetimeIdentity,
            .swapchainLifecycleIdentity = identity.swapchainLifecycleIdentity,
            .operationIdentity = identity.presentOperationIdentity,
            .completionFd = reactorFd,
            .retainedBacking = backing,
            .ticket = std::move(ticket),
            .retireBacking = [leases, identity, backing](
                    DeviceRetirementTicketState state) {
                if (state == DeviceRetirementTicketState::Completed) {
                    backing->markCompleted();
                    static_cast<void>(leases->retireAfterPresentFence(identity));
                }
            }};
        const bool accepted = reactorFd == -1
            ? reactor->completeImmediately(std::move(job))
            : reactor->registerJob(std::move(job));
        if (accepted) {
            backing->markReactorOwned();
            return true;
        }
    } catch (...) {
    }
    if (reactorFd >= 0) static_cast<void>(::close(reactorFd));
    // The committed lease remains the terminal owner. Teardown must observe
    // the layer-owned fence directly before destroying it.
    return false;
}

[[nodiscard]] bool armInternalPresentFence(
        const vk::Vulkan& vk,
        const std::shared_ptr<PresentedWsiSemaphoreBacking>& backing,
        const std::shared_ptr<PresentedPhysicalImageLeaseRegistry>& leases,
        const PresentedPhysicalImageIdentity& identity,
        const std::shared_ptr<DeviceRetirementReactor>& reactor) noexcept {
    return armInternalPresentFence(vk.dev(), vk.df().GetFenceFdKHR,
        backing, leases, identity, reactor);
}
}

class D3B2VirtualPresentPendingOperation final :
        public VirtualPresentPendingOperation {
public:
    D3B2VirtualPresentPendingOperation(std::shared_ptr<Swapchain> value,
            D3B2InsertionPath&& insertion, D3B2PendingInsertion&& pendingValue,
            std::shared_ptr<DeviceRetirementReactor> retirement,
            std::shared_ptr<void> backingValue,
            std::vector<int> exportedFds = {}) :
        owner(std::move(value)), path(std::move(insertion)),
        pending(std::move(pendingValue)), reactor(std::move(retirement)),
        backing(std::move(backingValue)), fds(std::move(exportedFds)) {
        path.state = &owner->d3b2State;
        tickets.reserve(fds.size());
        for (size_t i = 0; i < fds.size(); ++i)
            tickets.push_back(std::make_shared<DeviceRetirementTicket>());
    }

    ~D3B2VirtualPresentPendingOperation() override {
        for (auto& fd : fds) if (fd >= 0) { static_cast<void>(::close(fd)); fd = -1; }
    }

    void activateCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        activate(wake);
    }
    void redirectCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        for (const auto& ticket : tickets) ticket->setWakeTarget(wake);
    }
    uint64_t swapchainLifecycleIdentity() const noexcept override {
        return owner ? owner->physicalLifecycleIdentity() : 0;
    }

    VirtualPresentCompletionStatus tryComplete() noexcept override {
        if (activationFailed.load()) return VirtualPresentCompletionStatus::FAILED;
        if (!activated.load() || tickets.empty())
            return VirtualPresentCompletionStatus::NOT_READY;

        // The terminal QueueSubmit fence is a different completion domain from
        // WSI presentation.  A5G requires returnedForGraphics to retire as soon
        // as the terminal submit is complete, even while presentation fences
        // remain pending and keep their own resources alive.
        const auto gateStatus = completionGate.observe(tickets);
        if (gateStatus == TerminalWsiCompletionGateStatus::Pending)
            return VirtualPresentCompletionStatus::NOT_READY;
        if (gateStatus == TerminalWsiCompletionGateStatus::DeviceLost) {
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        }
        if (gateStatus == TerminalWsiCompletionGateStatus::Failed)
            return VirtualPresentCompletionStatus::FAILED;

        if (gateStatus == TerminalWsiCompletionGateStatus::TerminalReady) {
            path.tryRetireGraphicsFence = [] { return VK_SUCCESS; };
            const auto result = tryRetireD3B2Insertion(path, pending);
            switch (result) {
            case D3B2RetirementResult::RETIRED:
                completionGate.markTerminalRetired();
                break;
            case D3B2RetirementResult::NOT_READY:
                return VirtualPresentCompletionStatus::NOT_READY;
            case D3B2RetirementResult::DEVICE_LOST:
                if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
                return VirtualPresentCompletionStatus::DEVICE_LOST;
            case D3B2RetirementResult::FAILED:
                return VirtualPresentCompletionStatus::FAILED;
            }
        }

        // Remaining tickets protect WSI-present resources only.  They may
        // outlive logical terminal retirement but still gate virtual-image
        // reuse until the current presentation path is fully safe.
        const auto wsiStatus = completionGate.observe(tickets);
        if (wsiStatus == TerminalWsiCompletionGateStatus::Pending)
            return VirtualPresentCompletionStatus::NOT_READY;
        if (wsiStatus == TerminalWsiCompletionGateStatus::DeviceLost) {
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        }
        if (wsiStatus != TerminalWsiCompletionGateStatus::WsiReady)
            return VirtualPresentCompletionStatus::FAILED;
        owner->captureOnlyPhase = 6;
        return VirtualPresentCompletionStatus::RETIRED;
    }

private:
    void activate(const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept {
        if (activated.exchange(true) || !reactor || !backing || fds.empty()) {
            activationFailed.store(true);
            return;
        }
        for (size_t index = 0; index < fds.size(); ++index) {
            tickets[index]->setWakeTarget(wake);
            DeviceRetirementJob job{
                .deviceIdentity = owner->terminalContext
                    ? owner->terminalContext->deviceIdentity() : 0,
                .swapchainLifecycleIdentity = owner->terminalContext
                    ? owner->terminalContext->lifecycleGeneration() : 0,
                .operationIdentity = pending.submitAccepted
                    ? nextRetirementOperationIdentity.fetch_add(
                        1, std::memory_order_relaxed) : 0,
                .completionFd = fds[index],
                .retainedBacking = backing,
                .ticket = tickets[index],
                .wakeTarget = wake
            };
            const bool accepted = fds[index] == -1
                ? reactor->completeImmediately(std::move(job))
                : reactor->registerJob(std::move(job));
            if (!accepted) {
                if (fds[index] >= 0) {
                    static_cast<void>(::close(fds[index]));
                    fds[index] = -1;
                }
                if (!reactor->retainEventSourceLost(backing))
                    std::cerr << "lsfg-vk: failed to retain D3B2 event-source-lost backing\n";
                activationFailed.store(true);
                return;
            }
            fds[index] = -1;
        }
    }

    [[nodiscard]] VirtualPresentCompletionStatus eventStatus() const noexcept {
        if (activationFailed.load()) return VirtualPresentCompletionStatus::FAILED;
        if (!activated.load()) return VirtualPresentCompletionStatus::NOT_READY;
        for (const auto& ticket : tickets) {
            switch (ticket->state()) {
            case DeviceRetirementTicketState::Pending:
                return VirtualPresentCompletionStatus::NOT_READY;
            case DeviceRetirementTicketState::DeviceLost:
                return VirtualPresentCompletionStatus::DEVICE_LOST;
            case DeviceRetirementTicketState::GpuError:
            case DeviceRetirementTicketState::FailedPermanent:
                return VirtualPresentCompletionStatus::FAILED;
            case DeviceRetirementTicketState::Completed:
                break;
            }
        }
        return VirtualPresentCompletionStatus::RETIRED;
    }

    std::shared_ptr<Swapchain> owner;
    D3B2InsertionPath path;
    D3B2PendingInsertion pending;
    std::shared_ptr<DeviceRetirementReactor> reactor;
    std::shared_ptr<void> backing;
    std::vector<int> fds;
    std::vector<std::shared_ptr<DeviceRetirementTicket>> tickets;
    std::atomic_bool activated{false};
    std::atomic_bool activationFailed{false};
    TerminalWsiCompletionGate completionGate;
};

class D3B1VirtualPresentPendingOperation final :
        public VirtualPresentPendingOperation {
public:
    D3B1VirtualPresentPendingOperation(std::shared_ptr<Swapchain> value,
            D3B1PresentPath&& presentPath, D3B1PendingPresent&& pendingValue,
            std::shared_ptr<DeviceRetirementReactor> retirement,
            std::shared_ptr<void> backingValue,
            std::vector<int> exportedFds = {}) :
        owner(std::move(value)), path(std::move(presentPath)),
        pending(std::move(pendingValue)), reactor(std::move(retirement)),
        backing(std::move(backingValue)), fds(std::move(exportedFds)) {
        path.state = &owner->d3b1State;
        tickets.reserve(fds.size());
        for (size_t i = 0; i < fds.size(); ++i)
            tickets.push_back(std::make_shared<DeviceRetirementTicket>());
    }

    ~D3B1VirtualPresentPendingOperation() override {
        for (auto& fd : fds) if (fd >= 0) { static_cast<void>(::close(fd)); fd = -1; }
    }

    void activateCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        if (activated.exchange(true) || !reactor || !backing || fds.empty()) {
            activationFailed.store(true);
            return;
        }
        for (size_t index = 0; index < fds.size(); ++index) {
            tickets[index]->setWakeTarget(wake);
            DeviceRetirementJob job{
                .deviceIdentity = owner->terminalContext
                    ? owner->terminalContext->deviceIdentity() : 0,
                .swapchainLifecycleIdentity = owner->terminalContext
                    ? owner->terminalContext->lifecycleGeneration() : 0,
                .operationIdentity = pending.submitAccepted
                    ? nextRetirementOperationIdentity.fetch_add(
                        1, std::memory_order_relaxed) : 0,
                .completionFd = fds[index], .retainedBacking = backing,
                .ticket = tickets[index], .wakeTarget = wake};
            const bool accepted = fds[index] == -1
                ? reactor->completeImmediately(std::move(job))
                : reactor->registerJob(std::move(job));
            if (!accepted) {
                if (fds[index] >= 0) {
                    static_cast<void>(::close(fds[index]));
                    fds[index] = -1;
                }
                if (!reactor->retainEventSourceLost(backing))
                    std::cerr << "lsfg-vk: failed to retain D3B1 event-source-lost backing\n";
                activationFailed.store(true);
                return;
            }
            fds[index] = -1;
        }
    }
    void redirectCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        for (const auto& ticket : tickets) ticket->setWakeTarget(wake);
    }
    uint64_t swapchainLifecycleIdentity() const noexcept override {
        return owner ? owner->physicalLifecycleIdentity() : 0;
    }

    VirtualPresentCompletionStatus tryComplete() noexcept override {
        if (activationFailed.load()) return VirtualPresentCompletionStatus::FAILED;
        if (!activated.load() || tickets.empty())
            return VirtualPresentCompletionStatus::NOT_READY;

        const auto gateStatus = completionGate.observe(tickets);
        if (gateStatus == TerminalWsiCompletionGateStatus::Pending)
            return VirtualPresentCompletionStatus::NOT_READY;
        if (gateStatus == TerminalWsiCompletionGateStatus::DeviceLost) {
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        }
        if (gateStatus == TerminalWsiCompletionGateStatus::Failed)
            return VirtualPresentCompletionStatus::FAILED;

        if (gateStatus == TerminalWsiCompletionGateStatus::TerminalReady) {
            path.tryRetireRenderFence = [] { return VK_SUCCESS; };
            const auto result = tryRetireD3B1Present(path, pending);
            switch (result) {
            case D3B1RetirementResult::RETIRED:
                completionGate.markTerminalRetired();
                break;
            case D3B1RetirementResult::NOT_READY:
                return VirtualPresentCompletionStatus::NOT_READY;
            case D3B1RetirementResult::DEVICE_LOST:
                if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
                return VirtualPresentCompletionStatus::DEVICE_LOST;
            case D3B1RetirementResult::FAILED:
                return VirtualPresentCompletionStatus::FAILED;
            }
        }

        const auto wsiStatus = completionGate.observe(tickets);
        if (wsiStatus == TerminalWsiCompletionGateStatus::Pending)
            return VirtualPresentCompletionStatus::NOT_READY;
        if (wsiStatus == TerminalWsiCompletionGateStatus::DeviceLost) {
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        }
        if (wsiStatus != TerminalWsiCompletionGateStatus::WsiReady)
            return VirtualPresentCompletionStatus::FAILED;
        owner->captureOnlyPhase = 6;
        return VirtualPresentCompletionStatus::RETIRED;
    }

private:
    std::shared_ptr<Swapchain> owner;
    D3B1PresentPath path;
    D3B1PendingPresent pending;
    std::shared_ptr<DeviceRetirementReactor> reactor;
    std::shared_ptr<void> backing;
    std::vector<int> fds;
    std::vector<std::shared_ptr<DeviceRetirementTicket>> tickets;
    std::atomic_bool activated{false};
    std::atomic_bool activationFailed{false};
    TerminalWsiCompletionGate completionGate;
};

class SwapchainFencePendingOperation final :
        public VirtualPresentPendingOperation {
public:
    SwapchainFencePendingOperation(std::shared_ptr<Swapchain> value,
            const vk::Vulkan& vulkan, VkFence valueFence,
            bool advanceFrameValue,
            std::vector<PresentedPhysicalImageAcquireToken> consumed = {}) :
        owner(std::move(value)), vk(&vulkan), fence(valueFence),
        advanceFrame(advanceFrameValue), consumedAcquires(std::move(consumed)) {}

    VirtualPresentCompletionStatus tryComplete() noexcept override {
        const auto result = vk->df().GetFenceStatus(vk->dev(), fence);
        if (result == VK_NOT_READY || result == VK_TIMEOUT)
            return VirtualPresentCompletionStatus::NOT_READY;
        if (result == VK_ERROR_DEVICE_LOST) {
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        }
        if (result != VK_SUCCESS)
            return VirtualPresentCompletionStatus::FAILED;
        for (const auto& token : consumedAcquires) {
            if (!owner->presentedPhysicalImages
                    || !owner->presentedPhysicalImages->retireAfterSafeCompletion(token))
                return VirtualPresentCompletionStatus::FAILED;
        }
        consumedAcquires.clear();
        if (advanceFrame) ++owner->fidx;
        return VirtualPresentCompletionStatus::RETIRED;
    }
    uint64_t swapchainLifecycleIdentity() const noexcept override {
        return owner ? owner->physicalLifecycleIdentity() : 0;
    }

private:
    std::shared_ptr<Swapchain> owner;
    const vk::Vulkan* vk{};
    VkFence fence{};
    bool advanceFrame{};
    std::vector<PresentedPhysicalImageAcquireToken> consumedAcquires;
};

class ReactorFencePendingOperation final : public VirtualPresentPendingOperation {
public:
    ReactorFencePendingOperation(std::shared_ptr<Swapchain> value,
            std::shared_ptr<DeviceRetirementReactor> retirement,
            int completionFd,
            std::vector<PresentedPhysicalImageAcquireToken> consumed = {},
            bool advanceFrameValue = false,
            std::shared_ptr<void> retained = {},
            std::shared_ptr<DeviceRetirementReactor::ReservedJob> reservedJobValue = {},
            std::shared_ptr<DeviceRetirementTicket> ticketValue = {}) : owner(std::move(value)),
        reactor(std::move(retirement)), fd(completionFd),
        consumedAcquires(std::move(consumed)), advanceFrame(advanceFrameValue),
        ticket(ticketValue ? std::move(ticketValue)
                           : std::make_shared<DeviceRetirementTicket>()),
        reservedJob(std::move(reservedJobValue)),
        retainedBacking(std::move(retained)) {
        if (!retainedBacking) retainedBacking = owner;
    }

    [[nodiscard]] bool bindCompletion(int completionFd,
            std::vector<PresentedPhysicalImageAcquireToken> consumed) noexcept {
        if (fd != -2) return false;
        fd = completionFd;
        consumedAcquires = std::move(consumed);
        return true;
    }

    ~ReactorFencePendingOperation() override {
        if (fd >= 0) static_cast<void>(::close(fd));
    }

    void activateCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        if (activated.exchange(true) || !owner || !reactor || !ticket || fd == -2) {
            activationFailed.store(true);
            return;
        }
        DeviceRetirementJob job{
            .deviceIdentity = owner->terminalContext
                ? owner->terminalContext->deviceIdentity() : 0,
            .swapchainLifecycleIdentity = owner->physicalSwapchainLifecycleIdentity,
            .operationIdentity = nextRetirementOperationIdentity.fetch_add(
                1, std::memory_order_relaxed),
            .completionFd = fd,
            .retainedBacking = retainedBacking,
            .ticket = ticket,
            .wakeTarget = wake};
        ticket->setWakeTarget(wake);
        const bool accepted = fd == -1
            ? reactor->completeImmediately(std::move(job))
            : (reservedJob && reservedJob->valid()
                ? reactor->registerReservedJob(*reservedJob, fd, retainedBacking)
                : reactor->registerJob(std::move(job)));
        if (!accepted) {
            if (fd >= 0) { static_cast<void>(::close(fd)); fd = -1; }
            if (!reactor->retainEventSourceLost(retainedBacking))
                std::cerr << "lsfg-vk: failed to retain virtual-final event-source-lost backing\n";
            activationFailed.store(true);
            return;
        }
        fd = -1;
    }
    void redirectCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>& wake) noexcept override {
        ticket->setWakeTarget(wake);
    }
    uint64_t swapchainLifecycleIdentity() const noexcept override {
        return owner ? owner->physicalLifecycleIdentity() : 0;
    }

    VirtualPresentCompletionStatus tryComplete() noexcept override {
        if (activationFailed.load()) return VirtualPresentCompletionStatus::FAILED;
        if (!activated.load()) return VirtualPresentCompletionStatus::NOT_READY;
        switch (ticket->state()) {
        case DeviceRetirementTicketState::Pending:
            return VirtualPresentCompletionStatus::NOT_READY;
        case DeviceRetirementTicketState::DeviceLost:
            if (owner->terminalContext) owner->terminalContext->quarantineDeviceLost();
            return VirtualPresentCompletionStatus::DEVICE_LOST;
        case DeviceRetirementTicketState::GpuError:
        case DeviceRetirementTicketState::FailedPermanent:
            return VirtualPresentCompletionStatus::FAILED;
        case DeviceRetirementTicketState::Completed:
            break;
        }
        for (const auto& token : consumedAcquires) {
            if (!owner->presentedPhysicalImages
                    || !owner->presentedPhysicalImages->retireAfterSafeCompletion(token))
                return VirtualPresentCompletionStatus::FAILED;
        }
        consumedAcquires.clear();
        if (advanceFrame) ++owner->fidx;
        return VirtualPresentCompletionStatus::RETIRED;
    }

private:
    std::shared_ptr<Swapchain> owner;
    std::shared_ptr<DeviceRetirementReactor> reactor;
    int fd{-1};
    std::vector<PresentedPhysicalImageAcquireToken> consumedAcquires;
    bool advanceFrame{};
    std::shared_ptr<DeviceRetirementTicket> ticket;
    std::shared_ptr<DeviceRetirementReactor::ReservedJob> reservedJob;
    std::shared_ptr<void> retainedBacking;
    std::atomic_bool activated{false};
    std::atomic_bool activationFailed{false};
};

}

namespace {
    struct CaptureResources {
        VkDevice device{};
        PFN_vkDestroyBuffer destroyBuffer{};
        PFN_vkFreeMemory freeMemory{};
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        ~CaptureResources() {
            if (buffer) destroyBuffer(device, buffer, nullptr);
            if (memory) freeMemory(device, memory, nullptr);
        }
    };

    [[nodiscard]] bool isFixedMode(const ls::GameConf& profile) {
        return profile.frame_generation_mode == ls::FrameGenerationMode::Fixed;
    }

    [[nodiscard]] bool isAdaptiveBypass(const ls::GameConf& profile) {
        return !isFixedMode(profile) && profile.multiplier == 1;
    }

    [[nodiscard]] bool hasPresentModeInfo(const void* nextChain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(nextChain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR)
                return true;
            current = current->pNext;
        }
        return false;
    }

    [[nodiscard]] VkFence applicationPresentFence(const void* nextChain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(nextChain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR) {
                const auto* info =
                    reinterpret_cast<const VkSwapchainPresentFenceInfoKHR*>(current);
                // Swapchain::present() handles one real WSI swapchain. Do not
                // guess the mapping of a batched application's fence array.
                if (info->swapchainCount == 1 && info->pFences)
                    return info->pFences[0];
                return VK_NULL_HANDLE;
            }
            current = current->pNext;
        }
        return VK_NULL_HANDLE;
    }

    [[nodiscard]] size_t generatedFrameCapacity(const ls::GameConf& profile) {
        if (isFixedMode(profile))
            return FixedFrameScheduler::DEFAULT_MAX_GENERATED_FRAMES;
        return profile.multiplier > 1 ? profile.multiplier - 1 : 0;
    }

    VkImageMemoryBarrier barrierHelper(VkImage handle,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            VkImageSubresourceRange range = vk::exchangeImageSubresourceRange()) {
        return vk::makeImageBarrier(handle,
            srcAccessMask, dstAccessMask, oldLayout, newLayout,
            srcQueueFamilyIndex, dstQueueFamilyIndex, range);
    }

    [[nodiscard]] VkImageSubresourceRange fullSwapchainRange(uint32_t arrayLayers) {
        auto range = vk::exchangeImageSubresourceRange();
        range.layerCount = arrayLayers;
        return range;
    }

    bool modifierSupportsBlitSource(const vk::Vulkan& vk, VkFormat format,
            uint64_t modifier) {
        VkDrmFormatModifierPropertiesListEXT list{
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
        VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &list};
        vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), format, &properties);
        std::vector<VkDrmFormatModifierPropertiesEXT> modifiers(
            list.drmFormatModifierCount);
        list.pDrmFormatModifierProperties = modifiers.data();
        vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), format, &properties);
        return std::ranges::any_of(modifiers, [modifier](const auto& item) {
            return item.drmFormatModifier == modifier
                && (item.drmFormatModifierTilingFeatures
                    & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
        });
    }

    VkResult acquireRealSwapchainImage(const vk::Vulkan& vk,
            VkSwapchainKHR swapchain, VkSemaphore semaphore,
            uint32_t* imageIndex, std::stop_token stopToken,
            VkFence fence = VK_NULL_HANDLE) {
        constexpr uint64_t WORKER_ACQUIRE_SLICE_NS = 50ULL * 1000ULL * 1000ULL;
        while (true) {
            if (stopToken.stop_possible() && stopToken.stop_requested())
                return VK_ERROR_OUT_OF_DATE_KHR;

            const uint64_t timeout = stopToken.stop_possible()
                ? WORKER_ACQUIRE_SLICE_NS
                : UINT64_MAX;
            const auto res = vk.df().AcquireNextImageKHR(vk.dev(), swapchain,
                timeout, semaphore, fence, imageIndex);
            if (res == VK_TIMEOUT && stopToken.stop_possible())
                continue;
            return res;
        }
    }
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace lsfgvk::layer::test {
__attribute__((visibility("default")))
void setAdaptiveReservationExecuteHooks(
        void (*before)(), void (*after)()) noexcept {
    adaptiveReservationBeforeExecute.store(before, std::memory_order_release);
    adaptiveReservationAfterExecute.store(after, std::memory_order_release);
}

__attribute__((visibility("default")))
void setAdaptiveBuilderFailurePoint(AdaptiveBuilderFailurePoint point) noexcept {
    adaptiveBuilderFailurePoint.store(point, std::memory_order_release);
}

__attribute__((visibility("default")))
void setAdaptiveExecuteFailurePoint(AdaptiveExecuteFailurePoint point) noexcept {
    adaptiveExecuteFailurePoint.store(point, std::memory_order_release);
}

__attribute__((visibility("default")))
void resetAdaptiveCommandPoolLifecycleCounts() noexcept {
    adaptiveCommandPoolCreates.store(0, std::memory_order_relaxed);
    adaptiveCommandPoolDestroys.store(0, std::memory_order_relaxed);
}

__attribute__((visibility("default")))
uint32_t adaptiveCommandPoolCreateCount() noexcept {
    return adaptiveCommandPoolCreates.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
uint32_t adaptiveCommandPoolDestroyCount() noexcept {
    return adaptiveCommandPoolDestroys.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
bool waitForAdaptiveCommandPoolDestroyCount(uint32_t count,
        uint32_t timeoutMilliseconds) noexcept {
    try {
        std::unique_lock lock(adaptiveCommandPoolLifecycleMutex);
        return adaptiveCommandPoolLifecycleCv.wait_for(lock,
            std::chrono::milliseconds(timeoutMilliseconds), [count] {
                return adaptiveCommandPoolDestroys.load(
                    std::memory_order_relaxed) >= count;
            });
    } catch (...) { return false; }
}
}
#endif

PrePresentGateResult Swapchain::prePresentGate() noexcept {
    if (!d3b3ProductionState)
        return PrePresentGateResult::READY;
    return d3b3ProductionState->prePresentGate();
}

void Swapchain::captureRealFrameOnce(const vk::Vulkan& vk, VkImage sourceImage,
        uint32_t imageIndex, const std::vector<VkSemaphore>& bridgeSemaphores) {
    if (!captureDiagnosticFormatSupported(this->info.format))
        throw ls::error("P4C-B unsupported capture source format");
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(this->info.extent.width)
        * this->info.extent.height * 4;
    CaptureResources resources{vk.dev(), vk.df().DestroyBuffer, vk.df().FreeMemory};
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byteSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    auto result = vk.df().CreateBuffer(vk.dev(), &bufferInfo, nullptr, &resources.buffer);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture buffer creation");
    VkMemoryRequirements requirements{};
    vk.df().GetBufferMemoryRequirements(vk.dev(), resources.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vk.fi().GetPhysicalDeviceMemoryProperties(vk.physdev(), &memoryProperties);
    std::optional<uint32_t> memoryType;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryType = i; break;
        }
        if (!memoryType && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) memoryType = i;
    }
    if (!memoryType) throw ls::error("P4C-B capture staging has no host-visible memory type");
    const auto memoryFlags = memoryProperties.memoryTypes[*memoryType].propertyFlags;
    const bool coherent = (memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    const VkMemoryAllocateInfo allocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = *memoryType
    };
    result = vk.df().AllocateMemory(vk.dev(), &allocation, nullptr, &resources.memory);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture memory allocation");
    result = vk.df().BindBufferMemory(vk.dev(), resources.buffer, resources.memory, 0);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture buffer bind");

    vk::CommandBuffer command(vk);
    command.begin(vk);
    const auto range = vk::exchangeImageSubresourceRange();
    const auto toTransfer = vk::makeImageBarrier(sourceImage, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    command.insertBarriers(vk, {toTransfer}, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {this->info.extent.width, this->info.extent.height, 1}
    };
    vk.df().CmdCopyImageToBuffer(command.handle(), sourceImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, resources.buffer, 1, &copy);
    const auto restore = vk::makeImageBarrier(sourceImage, VK_ACCESS_TRANSFER_READ_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    command.insertBarriers(vk, {restore}, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    command.end(vk);
    vk::Fence fence(vk);
    command.submit(vk, vk.queue(), bridgeSemaphores, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE, 0,
        fence.handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (!fence.wait(vk)) throw ls::error("P4C-B capture fence wait failed");

    void* mapped{};
    result = vk.df().MapMemory(vk.dev(), resources.memory, 0, byteSize, 0, &mapped);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture readback map");
    if (!coherent) {
        const VkMappedMemoryRange invalidate{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = resources.memory,
            .size = byteSize
        };
        result = vk.df().InvalidateMappedMemoryRanges(vk.dev(), 1, &invalidate);
        if (result != VK_SUCCESS) {
            vk.df().UnmapMemory(vk.dev(), resources.memory);
            throw ls::vulkan_error(result, "P4C-B capture readback invalidate");
        }
    }
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    uint64_t checksum = ::lsfgvk::common::fnv1a64(bytes, byteSize);
    VkDeviceSize nonZero{};
    for (VkDeviceSize i = 0; i < byteSize; ++i) {
        nonZero += bytes[i] != 0;
    }
    vk.df().UnmapMemory(vk.dev(), resources.memory);
    if (nonZero == 0) throw ls::error("P4C-B capture readback is entirely zero");
    std::cerr << "[DG2X-P4C-B] Real frame capture on application GPU\n"
        << "  Runtime mode: CAPTURE_ONLY\n"
        << "  Source image index: " << imageIndex << "\n"
        << "  Source VkImage: " << sourceImage << "\n"
        << "  Source format: " << static_cast<int>(this->info.format) << "\n"
        << "  Source extent: " << this->info.extent.width << 'x' << this->info.extent.height << "\n"
        << "  Present waits at boundary: " << bridgeSemaphores.size() << "\n"
        << "  bridgePresentWaits: PASS\n"
        << "  Capture queue family: " << vk.queueFamilyIndex() << "\n"
        << "  Capture command pool: PASS\n"
        << "  Capture staging allocation: PASS\n"
        << "  Capture byte size: " << byteSize << "\n"
        << "  Capture memoryType: " << *memoryType << "\n"
        << "  Capture host coherent: " << (coherent ? "YES" : "NO") << "\n"
        << "  Source state acquisition: PASS\n"
        << "  Source layout transition: PASS\n"
        << "  vkCmdCopyImageToBuffer: PASS\n"
        << "  Source state restore: PASS\n"
        << "  Capture submit A: PASS\n"
        << "  Capture fence completion: PASS\n"
        << "  Capture readback: PASS\n"
        << "  Captured byte count: " << byteSize << "\n"
        << "  Captured non-zero bytes: " << nonZero << "\n"
        << "  Captured checksum: 0x" << std::hex << checksum << std::dec << "\n"
        << "  Backend submission: NONE\n"
        << "  Frame DMA-BUF: NONE\n"
        << "  Frame FOREIGN ownership: NONE\n"
        << "  Frame SYNC_FD: NONE\n"
        << "  LSFG backend execution: NONE\n"
        << "  Frame transport connected: NO\n"
        << "DG2X_P4C_B_REAL_FRAME_CAPTURE_A_PASS\n"
        << "cross-device real frame captured on application GPU, but frame transport is not connected yet\n";
}

void layer::context_ModifySwapchainCreateInfo(const ls::GameConf& profile, uint32_t maxImages,
        VkSwapchainCreateInfoKHR& createInfo) {
    createInfo.imageUsage |=
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    switch (profile.pacing) {
        case ls::Pacing::None:
            // Preserve the exact Adaptive expansion. Fixed reserves enough real
            // swapchain images for its maximum dynamic generation capacity.
            createInfo.minImageCount += isFixedMode(profile)
                ? FixedFrameScheduler::DEFAULT_MAX_GENERATED_FRAMES + 1
                : profile.multiplier;
            if (maxImages && createInfo.minImageCount > maxImages)
                createInfo.minImageCount = maxImages;

            createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            break;
    }
}

bool layer::fgTransformationEligible(
        const ls::GameConf& profile, const SwapchainInfo& info) noexcept {
    constexpr VkImageUsageFlags2KHR requiredTransferUsage =
        VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT_KHR
        | VK_IMAGE_USAGE_2_TRANSFER_DST_BIT_KHR;
    return !info.d2Foundation
        && info.arrayLayers == 1
        && info.timelineSemaphoreAvailable
        && info.presentFenceAvailable
        && (info.usage & requiredTransferUsage) == requiredTransferUsage
        && !isAdaptiveBypass(profile);
}

Swapchain::Swapchain(const vk::Vulkan& vk, backend::Instance& backend,
            vk::RuntimeDevicePair devicePair,
            ls::GameConf profile, SwapchainInfo info,
            uint32_t offloadQueueFamily) :
        instance(backend),
        devicePair(std::move(devicePair)),
        fixedScheduler(profile.target_fps),
        fixedOutputPacer(profile.target_fps),
        profile(std::move(profile)), info(std::move(info)) {
    this->physicalSwapchainLifecycleIdentity =
        nextProductionSwapchainLifecycleIdentity.fetch_add(
            1, std::memory_order_relaxed);
    this->offloadQueueFamily = offloadQueueFamily;
    if (this->devicePair.crossDevice())
        this->crossDeviceMode = CrossDeviceRuntimeMode::CAPTURE_ONLY;

    // A virtual Adaptive 1x swapchain still needs the final virtual->real copy
    // resources even though it must not create an LSFG generation context.
    if (this->info.virtualized) {
        if (offloadQueueFamily == VK_QUEUE_FAMILY_IGNORED)
            throw ls::error("virtual presentation requires an auxiliary queue family");
    }

    if (!fgTransformationEligible(this->profile, this->info))
        return;

    // P4C-B0 deliberately exposes only the application-side capture boundary.
    // Do not construct backend/LSFG resources until real-frame transport exists.
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY) {
        auto backing = RuntimeDmaBufBacking::createImage(
            this->devicePair.render.identity, this->info.extent);
        auto endpointA = vk::makeRuntimeExchangeEndpoint(vk);
        auto endpointB = backend.runtimeExchangeEndpoint();
        const vk::RuntimeImageBackingInfo backingInfo{
            .extent = this->info.extent,
            .backingSize = backing.size(),
            .fourcc = backing.fourcc(),
            .modifier = backing.modifier(),
            .planeCount = backing.planeCount(),
            .plane = {backing.offset(), 0, backing.stride(), 0, 0}
        };
        this->frameTransportA = vk::createRuntimeImageEndpoint(
            endpointA, backing.duplicatePlaneFd(0), backingInfo);
        this->frameTransportB = vk::createRuntimeImageEndpoint(
            endpointB, backing.duplicatePlaneFd(0), backingInfo);
        this->frameTransportA = vk::RuntimeImageEndpoint::createExecutionResources(
            std::move(this->frameTransportA), 1);
        this->frameTransportB = vk::RuntimeImageEndpoint::createExecutionResources(
            std::move(this->frameTransportB), 1);
        this->frameTransportBacking = std::move(backing);
        this->frameTransportReady = true;
        return;
    }

    const VkExtent2D extent = this->info.extent;
    const bool hdr = this->info.format > 57;

    const auto format = hdr
        ? VK_FORMAT_R16G16B16A16_SFLOAT
        : VK_FORMAT_R8G8B8A8_UNORM;
    const auto sourceDescriptor = vk::makeSourceExchangeImageDescriptor(extent, format);
    const auto destinationDescriptor = vk::makeDestinationExchangeImageDescriptor(extent, format);
    std::pair<vk::ExternalImage, vk::ExternalImage> externalSourceImages{};
    std::vector<vk::ExternalImage> externalDestinationImages(
        generatedFrameCapacity(this->profile));

    this->sourceImages.reserve(2);
    this->sourceImages.emplace_back(vk, sourceDescriptor, externalSourceImages.first);
    this->sourceImages.emplace_back(vk, sourceDescriptor, externalSourceImages.second);

    this->destinationImages.reserve(externalDestinationImages.size());
    for (auto& image : externalDestinationImages)
        this->destinationImages.emplace_back(vk, destinationDescriptor, image);

    // A owns every exported destination initially, while B is its first
    // writer. Complete the one-time A -> EXTERNAL releases before importing
    // the same allocations into the backend context.
    if (!this->destinationImages.empty()) {
        std::vector<vk::Barrier> initialDestinationReleases;
        initialDestinationReleases.reserve(this->destinationImages.size());
        for (const auto& image : this->destinationImages) {
            initialDestinationReleases.push_back(
                vk::destinationInitialReleaseToBackend(
                    image.handle(), vk.queueFamilyIndex()));
        }
        const vk::CommandBuffer initialRelease{vk};
        initialRelease.begin(vk);
        initialRelease.insertBarriers(vk, initialDestinationReleases);
        initialRelease.end(vk);
        initialRelease.submit(vk);
    }

    int syncFd{};
    this->syncSemaphore.emplace(vk, 0, std::nullopt, &syncFd);

    std::vector<ls::OwnedFd> ownedDestinationReturnFds;
    ownedDestinationReturnFds.reserve(this->destinationImages.size());
    this->destinationReturnSemaphores.reserve(this->destinationImages.size());
    for (size_t i = 0; i < this->destinationImages.size(); ++i) {
        int fd{-1};
        this->destinationReturnSemaphores.emplace_back(
            vk, 0, std::nullopt, &fd);
        ownedDestinationReturnFds.emplace_back(fd);
    }
    this->destinationReturnStates.resize(this->destinationImages.size());

    std::vector<int> destinationReturnFds;
    destinationReturnFds.reserve(ownedDestinationReturnFds.size());
    for (auto& fd : ownedDestinationReturnFds)
        destinationReturnFds.push_back(fd.release());

    try {
        this->ctx = ls::owned_ptr<ls::R<backend::Context>>(
            new ls::R<backend::Context>(backend.openContext(
                std::move(externalSourceImages),
                std::move(externalDestinationImages), syncFd,
                std::move(destinationReturnFds),
                1.0F / this->profile.flow_scale, this->profile.performance_mode
            )),
            [backend = &backend](ls::R<backend::Context>& ctx) {
                backend->closeContext(ctx);
            }
        );

        backend::makeLeaking(); // don't worry about it :3
    } catch (const std::exception& e) {
        throw ls::error("failed to create swapchain context", e);
    }

    this->renderCommandBuffer.emplace(vk);
    if (!this->renderFence.has_value())
        this->renderFence.emplace(vk, vk::Fence::ExternalHandle::SyncFd);
    for (size_t i = 0; i < this->destinationImages.size(); i++) {
        this->passes.emplace_back(RenderPass {
            .commandBuffer = vk::CommandBuffer(vk),
            .acquireSemaphore = vk::Semaphore(vk)
        });
    }

    const size_t frames = std::max(this->info.images.size(), this->destinationImages.size() + 2);
    for (size_t i = 0; i < frames; i++) {
        this->postCopySemaphores.emplace_back(
            vk::Semaphore(vk),
            vk::Semaphore(vk)
        );
    }
}

void Swapchain::ensureGraphicsFinalResources(const vk::Vulkan& vk, uint32_t family,
        std::shared_ptr<D3B3DeviceLifetimeQuarantine> deviceQuarantine) {
    if (family == VK_QUEUE_FAMILY_IGNORED)
        throw ls::error("graphics-final queue family is unavailable");
    if (this->virtualFinalCommandBuffer.has_value()) {
        if (this->virtualFinalCommandFamily != family)
            throw ls::error("graphics-final command resources changed queue family");
        return;
    }
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = family;
    VkCommandPool pool{};
    const auto result = vk.df().CreateCommandPool(vk.dev(), &poolInfo, nullptr, &pool);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "graphics-final command pool creation failed");
    this->virtualFinalCommandPool = ls::owned_ptr<VkCommandPool>(
        new VkCommandPool(pool),
        [dev = vk.dev(), destroy = vk.df().DestroyCommandPool](VkCommandPool& value) {
            destroy(dev, value, nullptr);
        });
    this->virtualFinalCommandFamily = family;
    this->virtualFinalCommandBuffer.emplace(vk, pool);
    this->virtualFinalAcquireSemaphore.emplace(vk);
    this->virtualFinalPresentSemaphore.emplace(vk);
    this->terminalContext = std::make_shared<D3B3PerSwapchainTerminalContext>(
        vk, family, nextProductionSwapchainLifecycleIdentity.fetch_add(
            1, std::memory_order_relaxed), std::move(deviceQuarantine));
    this->terminalHandoffBinding = std::make_unique<D3B3AReturnHandoffBinding>(
        this->terminalContext);
    if (d3b2InsertionDiagnosticEnabled()) {
        this->d3b2OriginalAcquireSemaphore.emplace(vk);
        this->d3b2GeneratedPresentSemaphore.emplace(vk);
        this->d3b2OriginalPresentSemaphore.emplace(vk);
    }
    if (!this->renderFence.has_value()) {
      this->renderFence.emplace(vk, vk::Fence::ExternalHandle::SyncFd);
    }
}

struct AdaptivePreparationWaitBackingSlot final {
    std::shared_ptr<void> value;
};

struct Adaptive1xPreparationContext final {
    ~Adaptive1xPreparationContext() noexcept {
        if (consumed) return;
        if (reservedBorrowedFence && borrowedFences)
            borrowedFences->abort(*reservedBorrowedFence);
        if (reservedLease.valid() && leases)
            leases->abort(reservedLease);
    }
    std::shared_ptr<Swapchain> owner;
    VkDevice device{VK_NULL_HANDLE};
    vk::VulkanDeviceFuncs dispatch{};
    VkQueue queue{VK_NULL_HANDLE};
    std::shared_ptr<std::mutex> queueMutex;
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    uint32_t virtualImageIndex{};
    VkFence applicationPresentFence{VK_NULL_HANDLE};
    std::stop_token stopToken;
    std::shared_ptr<void> runtimeGpuLifetime;
    std::shared_ptr<DeviceRetirementReactor> reactor;
    std::shared_ptr<BorrowedPresentFenceRegistry> borrowedFences;
    std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases;
    std::shared_ptr<std::unique_ptr<VirtualPresentPendingOperation>> publication;
    std::shared_ptr<AdaptiveAcquireSyncBacking> acquireBacking;
    std::shared_ptr<AdaptiveProducerSyncBacking> producerBacking;
    std::shared_ptr<AdaptiveProducerCommandBacking> commandBacking;
    std::shared_ptr<PresentedWsiSemaphoreBacking> presentBacking;
    std::shared_ptr<AdaptivePreparationWaitBackingSlot> waitBackingSlot;
    std::shared_ptr<void> recoveryBacking;
    std::shared_ptr<std::unique_ptr<ReactorFencePendingOperation>> pendingHolder;
    std::shared_ptr<PresentedPhysicalImageIdentity> presentFenceIdentity;
    std::shared_ptr<DeviceRetirementReactor::ReservedJob> presentFenceJob;
    Adaptive1xPreparedLogicalFinal::Operations preparedOperations;
    std::optional<Adaptive1xRecoveryAuthority> recovery;
    PresentedPhysicalImageLeaseRegistry::ReservedPresentedLease reservedLease;
    std::optional<BorrowedPresentFenceRegistry::ReservedBorrowedPresentFence>
        reservedBorrowedFence;
    std::vector<PresentedPhysicalImageAcquireToken> consumedAcquires;
    uint64_t presentOperationIdentity{};
    bool consumed{};
};

Adaptive1xPreparationReservation Swapchain::reserveAdaptive1xPreparation(
        const vk::Vulkan& vk, VkQueue queue,
        std::shared_ptr<std::mutex> queueMutex, VkSwapchainKHR swapchain,
        uint32_t virtualImageIndex, VkFence applicationPresentFence,
        std::stop_token stopToken, std::shared_ptr<void> runtimeGpuLifetime,
        std::shared_ptr<DeviceRetirementReactor> reactor,
        std::shared_ptr<BorrowedPresentFenceRegistry> borrowedFences,
        std::shared_ptr<std::unique_ptr<VirtualPresentPendingOperation>> publication) {
    if (!queueMutex || queue == VK_NULL_HANDLE || swapchain == VK_NULL_HANDLE
            || !reactor || !publication
            || this->info.releaseBackend == SwapchainReleaseBackend::None)
        throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
            "virtual Adaptive reservation inputs are unavailable");
    auto context = std::make_shared<Adaptive1xPreparationContext>();
    context->owner = shared_from_this();
    context->device = vk.dev();
    context->dispatch = vk.df();
    context->queue = queue;
    context->queueMutex = std::move(queueMutex);
    context->swapchain = swapchain;
    context->virtualImageIndex = virtualImageIndex;
    context->applicationPresentFence = applicationPresentFence;
    context->stopToken = stopToken;
    context->runtimeGpuLifetime = std::move(runtimeGpuLifetime);
    context->reactor = std::move(reactor);
    context->borrowedFences = std::move(borrowedFences);
    context->leases = this->presentedPhysicalImages;
    context->publication = std::move(publication);
    if (virtualImageIndex >= this->info.images.size()
            || !this->presentedPhysicalImages || !this->terminalContext)
        throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
            "virtual Adaptive reservation identity is unavailable");
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::AcquireBacking);
#endif
    context->acquireBacking = std::make_shared<AdaptiveAcquireSyncBacking>(vk);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::ProducerBacking);
#endif
    context->producerBacking = std::make_shared<AdaptiveProducerSyncBacking>(vk);
    context->commandBacking = std::make_shared<AdaptiveProducerCommandBacking>(
        vk, this->virtualFinalCommandFamily);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::PresentBacking);
#endif
    context->presentBacking = std::make_shared<PresentedWsiSemaphoreBacking>(vk);
    context->waitBackingSlot =
        std::make_shared<AdaptivePreparationWaitBackingSlot>();
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(
        test::AdaptiveBuilderFailurePoint::CompositeRecoveryBacking);
#endif
    context->recoveryBacking = combineGpuRetirementBacking(
        combineGpuRetirementBacking(
            combineGpuRetirementBacking(context->acquireBacking,
                context->producerBacking), context->presentBacking),
        context->runtimeGpuLifetime);
    context->recoveryBacking = combineGpuRetirementBacking(
        std::move(context->recoveryBacking), context->commandBacking);
    context->recoveryBacking = combineGpuRetirementBacking(
        std::move(context->recoveryBacking), context->waitBackingSlot);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::ConsumedAcquireStorage);
#endif
    context->consumedAcquires.reserve(1);
    context->presentOperationIdentity =
        nextRetirementOperationIdentity.fetch_add(1, std::memory_order_relaxed);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::ReactorPendingOperation);
#endif
    auto pendingTicket = std::make_shared<DeviceRetirementTicket>();
    auto pendingReservation = context->reactor->reserveJob(DeviceRetirementJob{
        .deviceIdentity = this->terminalContext->deviceIdentity(),
        .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
        .operationIdentity = context->presentOperationIdentity,
        .ticket = pendingTicket});
    if (!pendingReservation)
        throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
            "Adaptive producer reactor reservation failed");
    auto pendingJob = std::make_shared<DeviceRetirementReactor::ReservedJob>(
        std::move(*pendingReservation));
    auto pendingOperation = std::make_unique<ReactorFencePendingOperation>(
        shared_from_this(), context->reactor, -2,
        std::vector<PresentedPhysicalImageAcquireToken>{}, true,
        context->recoveryBacking, std::move(pendingJob),
        std::move(pendingTicket));
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::PendingHolder);
#endif
    context->pendingHolder =
        std::make_shared<std::unique_ptr<ReactorFencePendingOperation>>(
            std::move(pendingOperation));
    context->presentFenceIdentity =
        std::make_shared<PresentedPhysicalImageIdentity>();
    context->presentFenceJob =
        std::make_shared<DeviceRetirementReactor::ReservedJob>();
    auto presentFenceTicket = std::make_shared<DeviceRetirementTicket>();
    auto presentFenceJob = context->reactor->reserveJob(DeviceRetirementJob{
        .deviceIdentity = this->terminalContext->deviceIdentity(),
        .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
        .operationIdentity = context->presentOperationIdentity,
        .ticket = std::move(presentFenceTicket),
        .retireBacking = [leases = context->leases,
                identity = context->presentFenceIdentity,
                backing = context->presentBacking](
                DeviceRetirementTicketState state) {
            if (state == DeviceRetirementTicketState::Completed) {
                backing->markCompleted();
                static_cast<void>(leases->retireAfterPresentFence(*identity));
            }
        }});
    if (!presentFenceJob)
        throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
            "Adaptive present-fence reactor reservation failed");
    *context->presentFenceJob = std::move(*presentFenceJob);
    const auto device = vk.dev();
    const auto waitForFences = vk.df().WaitForFences;
    const auto getFenceFd = vk.df().GetFenceFdKHR;
    const auto releaseKhr = vk.df().ReleaseSwapchainImagesKHR;
    const auto releaseExt = vk.df().ReleaseSwapchainImagesEXT;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::PreparedCallbacks);
#endif
    context->preparedOperations = {
        .armInternalPresentFence = [device, getFenceFd,
                presentBacking = context->presentBacking,
                leases = this->presentedPhysicalImages,
                reactor = context->reactor,
                reserved = context->presentFenceJob,
                reservedIdentity = context->presentFenceIdentity](
                const PresentedPhysicalImageIdentity& identity) {
            if (!getFenceFd || !reactor || !reserved || !reserved->valid()
                    || !identity.valid()) return false;
            int fd{-1};
            const VkFenceGetFdInfoKHR exportInfo{
                .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
                .fence = presentBacking->internalPresentFence.handle(),
                .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
            if (getFenceFd(device, &exportInfo, &fd) != VK_SUCCESS)
                return false;
            presentBacking->acceptExportedFd(fd);
            *reservedIdentity = identity;
            if (fd == -1) {
                presentBacking->markCompleted();
                static_cast<void>(leases->retireAfterPresentFence(identity));
                return true;
            }
            const auto reactorFd = ::dup(fd);
            if (reactorFd < 0) return false;
            if (!reactor->registerReservedJob(
                    *reserved, reactorFd, presentBacking)) {
                static_cast<void>(::close(reactorFd));
                return false;
            }
            presentBacking->markReactorOwned();
            return true;
        },
        .exportProducerCompletion = [device, getFenceFd,
                producerBacking = context->producerBacking](int* fd) {
            if (!fd || !getFenceFd) return VK_ERROR_EXTENSION_NOT_PRESENT;
            const VkFenceGetFdInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
                .fence = producerBacking->fence.handle(),
                .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
            return getFenceFd(device, &info, fd);
        },
        .publishCompletion = [pendingHolder = context->pendingHolder,
                publication = context->publication](int fd,
                std::vector<PresentedPhysicalImageAcquireToken>&& tokens) {
            if (!publication || !*pendingHolder
                    || !(*pendingHolder)->bindCompletion(fd, std::move(tokens)))
                return VK_ERROR_DEVICE_LOST;
            *publication = std::move(*pendingHolder);
            return VK_SUCCESS;
        },
        .retainConservatively = [reactor = context->reactor](
                std::shared_ptr<void> value) {
            return reactor && reactor->retainEventSourceLost(std::move(value));
        }};
    const auto releaseBackend = this->info.releaseBackend;
    const auto releasePhysicalImage = [device, swapchain, releaseBackend,
            releaseKhr, releaseExt](uint32_t index) {
        const VkReleaseSwapchainImagesInfoKHR releaseInfo{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = swapchain, .imageIndexCount = 1,
            .pImageIndices = &index};
        if (releaseBackend == SwapchainReleaseBackend::Khr && releaseKhr)
            return releaseKhr(device, &releaseInfo);
        if (releaseBackend == SwapchainReleaseBackend::Ext && releaseExt)
            return releaseExt(device, &releaseInfo);
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    };
    const auto recoveryGeneration = nextRetirementOperationIdentity.fetch_add(
        1, std::memory_order_relaxed);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(test::AdaptiveBuilderFailurePoint::RecoveryCallbacks);
#endif
    context->recovery.emplace(this->terminalContext->deviceIdentity(),
        this->physicalSwapchainLifecycleIdentity, recoveryGeneration,
        swapchain, releaseBackend, context->acquireBacking->semaphore.handle(),
        context->acquireBacking->fence.handle(),
        context->producerBacking->fence.handle(), context->recoveryBacking,
        Adaptive1xRecoveryAuthority::Operations{
            .waitAcquireCompletion = [device, waitForFences,
                    backing = context->acquireBacking] {
                const auto fence = backing->fence.handle();
                return waitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            },
            .waitProducerCompletion = [device, waitForFences,
                    backing = context->producerBacking] {
                const auto fence = backing->fence.handle();
                return waitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            },
            .releasePhysicalImage = releasePhysicalImage,
            .retainTerminal = [reactor = context->reactor](
                    std::shared_ptr<void> value) {
                return reactor && reactor->retainEventSourceLost(std::move(value));
            }});
    if (!context->recovery->eligible())
        throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
            "virtual Adaptive recovery reservation is unavailable");
    const PresentedPhysicalImageIdentity unboundIdentity{
        .deviceLifetimeIdentity = this->terminalContext->deviceIdentity(),
        .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
        .physicalSwapchainIdentity = reinterpret_cast<uintptr_t>(swapchain),
        .physicalImageIndex = 0,
        .presentOperationIdentity = context->presentOperationIdentity};
    PresentedPhysicalImageLeaseRegistry::PrepareFailure leaseFailure{};
    Adaptive1xPreparationReservation::Operations reservationOperations{
        .execute = [context](VkSemaphore preparationWaitSemaphore)
                -> Adaptive1xPreparedLogicalFinal {
            auto& recovery = *context->recovery;
            using PreparedLease = std::optional<
                PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease>;
            using PreparedFence = std::optional<
                BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>;
            auto cleanReservations = [&](PreparedLease* preparedLease,
                    PreparedFence* preparedFence) noexcept {
                if (preparedFence && *preparedFence && context->borrowedFences)
                    context->borrowedFences->abort(**preparedFence);
                else if (context->reservedBorrowedFence
                        && context->borrowedFences)
                    context->borrowedFences->abort(
                        *context->reservedBorrowedFence);
                if (preparedLease && *preparedLease)
                    context->leases->abort(**preparedLease);
                else if (context->reservedLease.valid())
                    context->leases->abort(context->reservedLease);
            };
            const auto recoverAcquire = [&]() noexcept {
                const auto result = recovery.recoverAcquireFailureOnly();
                if (result == VK_SUCCESS) {
                    for (const auto& token : context->consumedAcquires)
                        if (!context->leases->retireAfterAcquireCompletion(token))
                            return VK_ERROR_DEVICE_LOST;
                    context->consumedAcquires.clear();
                }
                return result;
            };
            if (!recovery.markAcquireCalled()) {
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                    "Adaptive reservation lost Acquire authority");
            }
            uint32_t realImageIdx{};
            VkResult acquireResult{};
            do {
                if (context->stopToken.stop_possible()
                        && context->stopToken.stop_requested()) {
                    acquireResult = VK_ERROR_OUT_OF_DATE_KHR;
                    break;
                }
                acquireResult = context->dispatch.AcquireNextImageKHR(
                    context->device, context->swapchain,
                    context->stopToken.stop_possible()
                        ? 50ULL * 1000ULL * 1000ULL : UINT64_MAX,
                    context->acquireBacking->semaphore.handle(),
                    context->acquireBacking->fence.handle(), &realImageIdx);
            } while (acquireResult == VK_TIMEOUT
                && context->stopToken.stop_possible());
            if (!recovery.bindAcquireResult(acquireResult, realImageIdx)) {
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "Adaptive reservation lost Acquire result");
            }
            if (acquireResult != VK_SUCCESS
                    && acquireResult != VK_SUBOPTIMAL_KHR) {
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(acquireResult,
                    "reserved Adaptive physical Acquire failed");
            }
            if (const auto token = context->leases->reacquired(
                    context->owner->terminalContext->deviceIdentity(),
                    context->owner->physicalSwapchainLifecycleIdentity,
                    reinterpret_cast<uintptr_t>(context->swapchain), realImageIdx,
                    nextRetirementOperationIdentity.fetch_add(
                        1, std::memory_order_relaxed)))
                context->consumedAcquires.push_back(*token);
            if (realImageIdx >= context->owner->info.realImages.size()) {
                static_cast<void>(recoverAcquire());
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "reserved Adaptive Acquire image is out of range");
            }
            const auto command = context->commandBacking->commandBuffer->handle();
            const VkCommandBufferBeginInfo beginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            auto result = context->dispatch.BeginCommandBuffer(command, &beginInfo);
            if (result == VK_SUCCESS) {
                const VkImageMemoryBarrier pre[]{
                    barrierHelper(context->owner->info.images[
                            context->virtualImageIndex], VK_ACCESS_NONE,
                        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        fullSwapchainRange(context->owner->info.arrayLayers)),
                    barrierHelper(context->owner->info.realImages[realImageIdx],
                        VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        fullSwapchainRange(context->owner->info.arrayLayers))};
                context->dispatch.CmdPipelineBarrier(command,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                    2, pre);
                const auto region = vk::makeImageBlitRegion(
                    {context->owner->info.extent, context->owner->info.extent},
                    context->owner->info.arrayLayers);
                context->dispatch.CmdBlitImage(command,
                    context->owner->info.images[context->virtualImageIndex],
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    context->owner->info.realImages[realImageIdx],
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                    VK_FILTER_NEAREST);
                const VkImageMemoryBarrier post[]{
                    barrierHelper(context->owner->info.images[
                            context->virtualImageIndex],
                        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        fullSwapchainRange(context->owner->info.arrayLayers)),
                    barrierHelper(context->owner->info.realImages[realImageIdx],
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        fullSwapchainRange(context->owner->info.arrayLayers))};
                context->dispatch.CmdPipelineBarrier(command,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                    nullptr, 2, post);
                result = context->dispatch.EndCommandBuffer(command);
            }
            if (result != VK_SUCCESS) {
                static_cast<void>(recoverAcquire());
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(result,
                    "reserved Adaptive command recording failed");
            }
            const PresentedPhysicalImageIdentity presented{
                .deviceLifetimeIdentity =
                    context->owner->terminalContext->deviceIdentity(),
                .swapchainLifecycleIdentity =
                    context->owner->physicalSwapchainLifecycleIdentity,
                .physicalSwapchainIdentity =
                    reinterpret_cast<uintptr_t>(context->swapchain),
                .physicalImageIndex = realImageIdx,
                .presentOperationIdentity = context->presentOperationIdentity};
            PresentedPhysicalImageLeaseRegistry::PrepareFailure leaseFailure{};
            auto preparedLease =
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
                failAdaptiveExecuteAt(test::AdaptiveExecuteFailurePoint::PhysicalLeaseBind)
                    ? std::optional<PresentedPhysicalImageLeaseRegistry::
                        PreparedPresentedLease>{}
                    :
#endif
                context->leases->bindPhysicalImage(
                    context->reservedLease, presented, &leaseFailure);
            if (!preparedLease) {
                static_cast<void>(recoverAcquire());
                cleanReservations(nullptr, nullptr);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "reserved Adaptive physical lease bind failed");
            }
            std::optional<
                BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
                preparedFence;
            if (context->reservedBorrowedFence) {
                BorrowedPresentFenceRegistry::PrepareFailure fenceFailure{};
                preparedFence =
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
                    failAdaptiveExecuteAt(
                        test::AdaptiveExecuteFailurePoint::BorrowedFenceBind)
                        ? std::optional<BorrowedPresentFenceRegistry::
                            PreparedBorrowedPresentFence>{}
                        :
#endif
                    context->borrowedFences->bindPresentedIdentity(
                        *context->reservedBorrowedFence, presented, &fenceFailure);
                if (!preparedFence) {
                    static_cast<void>(recoverAcquire());
                    cleanReservations(&preparedLease, nullptr);
                    context->consumed = true;
                    throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                        "reserved Adaptive borrowed fence bind failed");
                }
            }
            if (!recovery.markProducerPrepared()) {
                static_cast<void>(recoverAcquire());
                cleanReservations(&preparedLease, &preparedFence);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "reserved Adaptive producer authority was lost");
            }
            const VkSemaphore waits[]{preparationWaitSemaphore,
                context->acquireBacking->semaphore.handle()};
            const VkPipelineStageFlags stages[]{
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
            const VkSemaphore signal = context->presentBacking->ready.handle();
            const VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 2,
                .pWaitSemaphores = waits,
                .pWaitDstStageMask = stages,
                .commandBufferCount = 1,
                .pCommandBuffers = &command,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &signal};
            {
                const std::scoped_lock queueLock(*context->queueMutex);
                result = context->dispatch.QueueSubmit(context->queue, 1,
                    &submitInfo, context->producerBacking->fence.handle());
            }
            if (!recovery.bindProducerSubmitResult(result)) {
                cleanReservations(&preparedLease, &preparedFence);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "reserved Adaptive producer result was lost");
            }
            if (result != VK_SUCCESS) {
                const auto recoveryState = recovery.state();
                if (recoveryState == Adaptive1xRecoveryAuthority::State::DeviceLost
                        || recoveryState
                            == Adaptive1xRecoveryAuthority::State::Indeterminate) {
                    // QueueSubmit may have consumed the Acquire semaphore and
                    // referenced the command/resources.  The recovery authority
                    // has already transferred the strong backing to the device
                    // reactor; retain the bound physical-image lease as well.
                    // No logical present occurred, so the application fence
                    // association remains uncommitted and is aborted.
                    if (preparedFence && context->borrowedFences)
                        context->borrowedFences->abort(*preparedFence);
                    if (!context->leases->commit(*preparedLease))
                        context->leases->abort(*preparedLease);
                    context->consumed = true;
                    throw ls::vulkan_error(result,
                        "reserved Adaptive producer submit was indeterminate");
                }
                if (result == VK_ERROR_OUT_OF_HOST_MEMORY
                        || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
                    static_cast<void>(recoverAcquire());
                cleanReservations(&preparedLease, &preparedFence);
                context->consumed = true;
                throw ls::vulkan_error(result,
                    "reserved Adaptive producer submit failed");
            }
            if (!recovery.markReadyForLogicalPresent()) {
                static_cast<void>(recovery.recoverProducerFailureOnly());
                cleanReservations(&preparedLease, &preparedFence);
                context->consumed = true;
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "reserved Adaptive producer readiness was lost");
            }
            Adaptive1xPreparedLogicalFinal prepared(context->swapchain,
                realImageIdx, signal, context->applicationPresentFence,
                context->presentBacking->internalPresentFence.handle(), presented,
                std::move(recovery), context->leases, *preparedLease,
                context->borrowedFences, std::move(preparedFence),
                std::move(context->consumedAcquires), context->recoveryBacking,
                std::move(context->preparedOperations));
            context->consumed = true;
            return prepared;
        },
        .abortUnused = [context] {
            if (context->reservedBorrowedFence && context->borrowedFences)
                context->borrowedFences->abort(*context->reservedBorrowedFence);
            if (context->reservedLease.valid() && context->leases)
                context->leases->abort(context->reservedLease);
            context->consumed = true;
        },
        .valid = [context] { return !context->consumed && context->owner
                && context->publication && context->recovery
                && context->reservedLease.valid(); },
        .failureState = [context] {
            if (!context->recovery)
                return Adaptive1xPreparationReservation::State::CleanlyAborted;
            using RecoveryState = Adaptive1xRecoveryAuthority::State;
            switch (context->recovery->state()) {
            case RecoveryState::DeviceLost:
            case RecoveryState::Indeterminate:
                return Adaptive1xPreparationReservation::State::ConservativelyRetained;
            case RecoveryState::Released:
                return Adaptive1xPreparationReservation::State::Recovered;
            default:
                return Adaptive1xPreparationReservation::State::CleanlyAborted;
            }
        },
        .attachWaitBacking = [slot = context->waitBackingSlot](
                std::shared_ptr<void> backing) noexcept {
            if (!slot || slot->value || !backing) return false;
            slot->value = std::move(backing);
            return true;
        }};
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    if (consumeAdaptiveBuilderAt(
            test::AdaptiveBuilderFailurePoint::PhysicalLeaseReserve))
        test::failNextPresentedPhysicalLeaseReserve();
#endif
    auto reservedLease = context->leases->reserve(
        unboundIdentity, context->recoveryBacking, &leaseFailure,
        applicationPresentFence == VK_NULL_HANDLE
            ? std::function<VkResult()>{[backing = context->presentBacking,
                    device, waitForFences] {
                return backing->waitForCompletion(device, waitForFences);
            }} : std::function<VkResult()>{});
    if (!reservedLease) {
        if (leaseFailure
                == PresentedPhysicalImageLeaseRegistry::PrepareFailure::OutOfHostMemory)
            throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                "failed to reserve physical-image WSI lease");
        throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
            "physical-image WSI lease reservation conflict");
    }
    context->reservedLease = *reservedLease;
    if (applicationPresentFence != VK_NULL_HANDLE) {
        if (!context->borrowedFences) {
            context->leases->abort(context->reservedLease);
            throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                "borrowed present-fence registry is unavailable");
        }
        BorrowedPresentFenceRegistry::PrepareFailure fenceFailure{};
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (consumeAdaptiveBuilderAt(
                test::AdaptiveBuilderFailurePoint::BorrowedFenceReserve))
            test::failNextBorrowedPresentFenceReserve();
#endif
        context->reservedBorrowedFence = context->borrowedFences->reserve(
            applicationPresentFence, context->leases, &fenceFailure);
        if (!context->reservedBorrowedFence) {
            context->leases->abort(context->reservedLease);
            if (fenceFailure
                    == BorrowedPresentFenceRegistry::PrepareFailure::OutOfHostMemory)
                throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                    "failed to reserve borrowed present-fence generation");
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                "application present fence has no current object identity");
        }
    }
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    failAdaptiveBuilderAt(
        test::AdaptiveBuilderFailurePoint::AfterBothRegistryReservations);
#endif
    return Adaptive1xPreparationReservation(std::move(reservationOperations));
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
void Swapchain::primeAdaptiveBuilderForTesting(const vk::Vulkan& vk,
        uint32_t family,
        std::shared_ptr<D3B3DeviceLifetimeQuarantine> quarantine,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases) {
    this->presentedPhysicalImages = std::move(leases);
    this->ensureGraphicsFinalResources(vk, family, std::move(quarantine));
}

SwapchainReleaseBackend Swapchain::setReleaseBackendForTesting(
        SwapchainReleaseBackend value) noexcept {
    const auto previous = this->info.releaseBackend;
    this->info.releaseBackend = value;
    return previous;
}
#endif

Adaptive1xPreparedLogicalFinal Swapchain::prepareAdaptive1xLogicalFinal(
        const vk::Vulkan& vk, VkQueue queue,
        std::shared_ptr<std::mutex> queueMutex, VkSwapchainKHR swapchain,
        uint32_t virtualImageIndex, VkSemaphore preparationWaitSemaphore,
        VkFence applicationPresentFenceValue, std::stop_token stopToken,
        std::shared_ptr<void> runtimeGpuLifetime,
        std::shared_ptr<DeviceRetirementReactor> deviceRetirementReactor,
        std::shared_ptr<BorrowedPresentFenceRegistry> borrowedPresentFences,
        std::shared_ptr<std::unique_ptr<VirtualPresentPendingOperation>>
            completionPublication) {
    if (!queueMutex || queue == VK_NULL_HANDLE || swapchain == VK_NULL_HANDLE
            || preparationWaitSemaphore == VK_NULL_HANDLE)
        throw ls::error("virtual Adaptive preparation inputs are unavailable");
    const auto& swapchainImage = this->info.images.at(virtualImageIndex);
    const auto& outputImages = this->info.realImages;
    const auto compensateAbortedLogicalPresentFence =
        [&](const char*) -> VkResult {
            if (applicationPresentFenceValue == VK_NULL_HANDLE)
                return VK_SUCCESS;
            const auto submit = [&](VkQueue actualQueue,
                    const VkSubmitInfo& submitInfo, VkFence actualFence) {
                const std::scoped_lock queueLock(*queueMutex);
                return vk.df().QueueSubmit(
                    actualQueue, 1, &submitInfo, actualFence);
            };
            return compensateEnqueuedPresentAbort(queue,
                applicationPresentFenceValue, submit);
        };
        auto acquireBacking = std::make_shared<AdaptiveAcquireSyncBacking>(vk);
        auto producerBacking = std::make_shared<AdaptiveProducerSyncBacking>(vk);
        auto commandBacking = std::make_shared<AdaptiveProducerCommandBacking>(
            vk, this->virtualFinalCommandFamily);
        auto presentBacking = std::make_shared<PresentedWsiSemaphoreBacking>(vk);
        auto recoveryBacking = combineGpuRetirementBacking(
            combineGpuRetirementBacking(
                combineGpuRetirementBacking(acquireBacking, producerBacking),
                presentBacking),
            runtimeGpuLifetime);
        recoveryBacking = combineGpuRetirementBacking(
            std::move(recoveryBacking), commandBacking);
        std::vector<PresentedPhysicalImageAcquireToken> consumedAcquires;
        consumedAcquires.reserve(1);
        if (!completionPublication || !deviceRetirementReactor)
            throw ls::error("virtual Adaptive event retirement is unavailable");
        // All heap-backed state required to publish/finalize the prepared
        // logical final is committed before the physical Acquire.  In
        // particular, ProducerSubmitAccepted must be followed only by
        // allocation-free moves/binds into Adaptive1xPreparedLogicalFinal.
        auto pendingHolder =
            std::make_shared<std::unique_ptr<ReactorFencePendingOperation>>(
                std::make_unique<ReactorFencePendingOperation>(
                    shared_from_this(), deviceRetirementReactor, -2,
                    std::vector<PresentedPhysicalImageAcquireToken>{}, true,
                    recoveryBacking));
        const auto device = vk.dev();
        const auto waitForFences = vk.df().WaitForFences;
        const auto getFenceFd = vk.df().GetFenceFdKHR;
        const auto releaseKhr = vk.df().ReleaseSwapchainImagesKHR;
        const auto releaseExt = vk.df().ReleaseSwapchainImagesEXT;
        Adaptive1xPreparedLogicalFinal::Operations preparedOperations{
            .armInternalPresentFence = [device, getFenceFd, presentBacking,
                    leases = this->presentedPhysicalImages,
                    deviceRetirementReactor](
                    const PresentedPhysicalImageIdentity& identity) {
                return armInternalPresentFence(device, getFenceFd, presentBacking,
                    leases, identity, deviceRetirementReactor);
            },
            .exportProducerCompletion = [device, getFenceFd,
                    producerBacking](int* fd) {
                if (!fd || !getFenceFd) return VK_ERROR_EXTENSION_NOT_PRESENT;
                const VkFenceGetFdInfoKHR info{
                    .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
                    .fence = producerBacking->fence.handle(),
                    .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
                return getFenceFd(device, &info, fd);
            },
            .publishCompletion = [pendingHolder, completionPublication](int fd,
                    std::vector<PresentedPhysicalImageAcquireToken>&& tokens) {
                if (!completionPublication || !*pendingHolder
                        || !(*pendingHolder)->bindCompletion(
                            fd, std::move(tokens)))
                    return VK_ERROR_DEVICE_LOST;
                *completionPublication = std::move(*pendingHolder);
                return VK_SUCCESS;
            },
            .retainConservatively = [deviceRetirementReactor](
                    std::shared_ptr<void> value) {
                return deviceRetirementReactor
                    && deviceRetirementReactor->retainEventSourceLost(
                        std::move(value));
            }};
        const auto releaseBackend = this->info.releaseBackend;
        const auto releasePhysicalImage = [device, swapchain, releaseBackend,
                releaseKhr, releaseExt](uint32_t index) {
            const VkReleaseSwapchainImagesInfoKHR releaseInfo{
                .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
                .swapchain = swapchain, .imageIndexCount = 1,
                .pImageIndices = &index};
            if (releaseBackend == SwapchainReleaseBackend::Khr && releaseKhr)
                return releaseKhr(device, &releaseInfo);
            if (releaseBackend == SwapchainReleaseBackend::Ext && releaseExt)
                return releaseExt(device, &releaseInfo);
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        };
        Adaptive1xRecoveryAuthority recovery(
            this->terminalContext->deviceIdentity(),
            this->physicalSwapchainLifecycleIdentity,
            nextRetirementOperationIdentity.fetch_add(1, std::memory_order_relaxed),
            swapchain, this->info.releaseBackend,
            acquireBacking->semaphore.handle(), acquireBacking->fence.handle(),
            producerBacking->fence.handle(), recoveryBacking, {
                .waitAcquireCompletion = [device, waitForFences, acquireBacking] {
                    const auto fence = acquireBacking->fence.handle();
                    return waitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
                },
                .waitProducerCompletion = [device, waitForFences, producerBacking] {
                    const auto fence = producerBacking->fence.handle();
                    return waitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
                },
                .releasePhysicalImage = releasePhysicalImage,
                .retainTerminal = [deviceRetirementReactor](std::shared_ptr<void> value) {
                    return deviceRetirementReactor
                        && deviceRetirementReactor->retainEventSourceLost(std::move(value));
                }});
        if (!recovery.eligible() || !recovery.markAcquireCalled())
            throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
                "virtual Adaptive 1x recovery authority is unavailable");

        uint32_t realImageIdx{};
        auto res = acquireRealSwapchainImage(vk, swapchain,
            acquireBacking->semaphore.handle(), &realImageIdx,
            stopToken, acquireBacking->fence.handle());
        if (!recovery.bindAcquireResult(res, realImageIdx))
            throw ls::error("physical Acquire recovery generation was lost");
        if (classifyPresentResult(res) == PresentResultClass::EnqueuedRejection) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden Adaptive 1x acquire");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

        if (const auto token = this->presentedPhysicalImages->reacquired(
                this->terminalContext->deviceIdentity(),
                this->physicalSwapchainLifecycleIdentity,
                reinterpret_cast<uintptr_t>(swapchain), realImageIdx,
                nextRetirementOperationIdentity.fetch_add(
                    1, std::memory_order_relaxed)))
            consumedAcquires.push_back(*token);
        const auto recoverAcquireOnly = [&] {
            const auto result = recovery.recoverAcquireFailureOnly();
            if (result == VK_SUCCESS) {
                for (const auto& token : consumedAcquires)
                    if (!this->presentedPhysicalImages
                            ->retireAfterAcquireCompletion(token))
                        return VK_ERROR_DEVICE_LOST;
                consumedAcquires.clear();
            }
            return result;
        };
        if (realImageIdx >= outputImages.size()) {
            static_cast<void>(recoverAcquireOnly());
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                "physical Acquire returned an out-of-range image index");
        }
        const auto& realImage = outputImages.at(realImageIdx);
        auto& finalCmdbuf = *commandBacking->commandBuffer;
        try {
        finalCmdbuf.begin(vk);
        finalCmdbuf.blitImage(vk,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    fullSwapchainRange(this->info.arrayLayers)
                ),
                barrierHelper(realImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    fullSwapchainRange(this->info.arrayLayers)
                ),
            },
            { swapchainImage, realImage },
            this->info.extent,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    fullSwapchainRange(this->info.arrayLayers)
                ),
                barrierHelper(realImage,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    fullSwapchainRange(this->info.arrayLayers)
                ),
            },
            this->info.arrayLayers
        );
        finalCmdbuf.end(vk);
        } catch (...) {
            static_cast<void>(recoverAcquireOnly());
            throw;
        }

        const PresentedPhysicalImageIdentity presented{
            .deviceLifetimeIdentity = this->terminalContext->deviceIdentity(),
            .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
            .physicalSwapchainIdentity = reinterpret_cast<uintptr_t>(swapchain),
            .physicalImageIndex = realImageIdx,
            .presentOperationIdentity = nextRetirementOperationIdentity.fetch_add(
                1, std::memory_order_relaxed)};
        PresentedPhysicalImageLeaseRegistry::PrepareFailure leaseFailure{};
        const auto applicationFence = applicationPresentFenceValue;
        auto preparedLease = this->presentedPhysicalImages->prepare(
            presented, recoveryBacking, &leaseFailure,
            applicationFence == VK_NULL_HANDLE
                ? std::function<VkResult()>{[backing = presentBacking, device,
                        waitForFences] {
                    return backing->waitForCompletion(device, waitForFences);
                }} : std::function<VkResult()>{});
        if (!preparedLease) {
            static_cast<void>(recoverAcquireOnly());
            if (leaseFailure == PresentedPhysicalImageLeaseRegistry::PrepareFailure::OutOfHostMemory)
                throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                    "failed to prepare physical-image WSI lease");
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                "physical image already owns a WSI lease reservation");
        }
        std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
            preparedFence;
        if (applicationFence != VK_NULL_HANDLE && borrowedPresentFences) {
            BorrowedPresentFenceRegistry::PrepareFailure fenceFailure{};
            preparedFence = borrowedPresentFences->prepare(applicationFence,
                this->presentedPhysicalImages, presented, &fenceFailure);
            if (!preparedFence) {
                this->presentedPhysicalImages->abort(*preparedLease);
                static_cast<void>(recoverAcquireOnly());
                if (fenceFailure == BorrowedPresentFenceRegistry::PrepareFailure::OutOfHostMemory)
                    throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                        "failed to prepare borrowed present-fence generation");
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "application present fence has no current object identity");
            }
        }
        if (!recovery.markProducerPrepared())
            throw ls::error("virtual Adaptive producer authority was not prepared");
        try {
            const std::scoped_lock queueLock(*queueMutex);
            finalCmdbuf.submit(vk, queue,
                {
                    preparationWaitSemaphore,
                    acquireBacking->semaphore.handle()
                },
                VK_NULL_HANDLE, 0,
                { presentBacking->ready.handle() },
                VK_NULL_HANDLE, 0,
                producerBacking->fence.handle()
            );
            if (!recovery.bindProducerSubmitResult(VK_SUCCESS)
                    || !recovery.markReadyForLogicalPresent())
                throw ls::error("virtual Adaptive producer acceptance was lost");
        } catch (const ls::vulkan_error& e) {
            if (preparedFence) borrowedPresentFences->abort(*preparedFence);
            this->presentedPhysicalImages->abort(*preparedLease);
            if (recovery.state()
                    == Adaptive1xRecoveryAuthority::State::ProducerPrepared) {
                static_cast<void>(recovery.bindProducerSubmitResult(
                    e.error()));
                if (e.error() == VK_ERROR_OUT_OF_HOST_MEMORY
                        || e.error() == VK_ERROR_OUT_OF_DEVICE_MEMORY)
                    static_cast<void>(recoverAcquireOnly());
            } else if (recovery.state()
                    == Adaptive1xRecoveryAuthority::State::ProducerSubmitAccepted
                    || recovery.state()
                        == Adaptive1xRecoveryAuthority::State::ReadyForLogicalPresent
                    || recovery.state()
                        == Adaptive1xRecoveryAuthority::State::LogicalPresentCalled) {
                static_cast<void>(recovery.recoverProducerFailureOnly());
            }
            throw;
        } catch (...) {
            if (preparedFence) borrowedPresentFences->abort(*preparedFence);
            this->presentedPhysicalImages->abort(*preparedLease);
            if (recovery.state()
                    == Adaptive1xRecoveryAuthority::State::ProducerPrepared)
                static_cast<void>(recovery.bindProducerSubmitResult(VK_ERROR_UNKNOWN));
            else if (recovery.state()
                    == Adaptive1xRecoveryAuthority::State::ProducerSubmitAccepted
                    || recovery.state()
                        == Adaptive1xRecoveryAuthority::State::ReadyForLogicalPresent
                    || recovery.state()
                        == Adaptive1xRecoveryAuthority::State::LogicalPresentCalled)
                static_cast<void>(recovery.recoverProducerFailureOnly());
            throw;
        }

        Adaptive1xPreparedLogicalFinal prepared(
            swapchain, realImageIdx, presentBacking->ready.handle(),
            applicationFence, presentBacking->internalPresentFence.handle(),
            presented, std::move(recovery), this->presentedPhysicalImages,
            *preparedLease, borrowedPresentFences, std::move(preparedFence),
            std::move(consumedAcquires), recoveryBacking,
            std::move(preparedOperations));
        if (!prepared.valid())
            throw ls::error("virtual Adaptive prepared logical final is invalid");
        return prepared;
}

SwapchainPresentResult Swapchain::present(const vk::Vulkan& vk,
        VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
        VkSwapchainKHR swapchain,
        void* next_chain, uint32_t imageIdx,
        const std::vector<VkSemaphore>& semaphores,
        std::stop_token stopToken,
        std::optional<std::chrono::steady_clock::time_point> sourcePresentTime,
        bool d3bSingleSwapchainEligible,
        const GraphicsFinalQueueInfo* graphicsFinalQueue,
        BorrowedGraphicsQueueLease* graphicsLease,
        bool* stopAfterCompletion,
        std::shared_ptr<D3B3DeviceLifetimeQuarantine> deviceQuarantine,
        std::shared_ptr<DeviceRetirementReactor> deviceRetirementReactor,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> physicalImageLeases,
        std::shared_ptr<BorrowedPresentFenceRegistry> borrowedPresentFences,
        std::unique_ptr<VirtualPresentPendingOperation>* pendingCompletion,
        PresentedPhysicalImageIdentity* presentedIdentity,
        std::shared_ptr<void> virtualGpuBacking) {
    const bool transformationBypass =
        !fgTransformationEligible(this->profile, this->info);

    // A non-virtualized bypass is an exact logical native present. Keep this
    // before every cross-device, capture, scheduling and hidden-acquire path.
    if (transformationBypass && !this->info.virtualized) {
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = next_chain,
            .waitSemaphoreCount = static_cast<uint32_t>(semaphores.size()),
            .pWaitSemaphores = semaphores.empty() ? nullptr : semaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIdx,
        };
        return SwapchainPresentResult::logical(
            vk.df().QueuePresentKHR(queue, &presentInfo));
    }

    auto runtimeGpuLifetime = combineGpuRetirementBacking(
        shared_from_this(), std::move(virtualGpuBacking));
    if (physicalImageLeases) {
        if (this->presentedPhysicalImages
                && this->presentedPhysicalImages != physicalImageLeases)
            throw ls::error("physical-image lease owner changed within swapchain lifecycle");
        this->presentedPhysicalImages = std::move(physicalImageLeases);
    }
    if (this->info.virtualized) {
        if (!graphicsFinalQueue || !graphicsLease
                || graphicsFinalExecutionMode(*graphicsFinalQueue, true)
                    != GraphicsFinalExecutionMode::BORROWED_SYNCHRONOUS
                || !graphicsLease->validFor(
                    graphicsFinalQueue->queue, graphicsFinalQueue->family)
                || queue != graphicsFinalQueue->queue)
            throw ls::error("borrowed graphics-final queue contract is unavailable");
        this->ensureGraphicsFinalResources(
            vk, graphicsFinalQueue->family, std::move(deviceQuarantine));
    }
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY
            && this->d3b1State == D3B1PresentationState::PASS) {
        if (!pendingCompletion)
            throw ls::error("D3B1 drain async completion is unavailable");
        if (semaphores.size() != 1 || queue == VK_NULL_HANDLE || !queueMutex)
            throw ls::error("D3B1 terminal ready-semaphore drain is unavailable");
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        const VkSubmitInfo drain{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = semaphores.data(),
            .pWaitDstStageMask = &waitStage};
        if (!this->renderFence.has_value())
            throw ls::error("D3B1 terminal drain fence is unavailable");
        this->renderFence->reset(vk);
        VkResult result{};
        {
            const std::scoped_lock queueLock(*queueMutex);
            result = vk.df().QueueSubmit(
                queue, 1, &drain, this->renderFence->handle());
        }
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "D3B1 terminal ready-semaphore drain failed");
        if (!deviceRetirementReactor)
            throw ls::error("D3B1 terminal drain reactor is unavailable");
        int completionFd{-1};
        try { completionFd = this->renderFence->exportSyncFd(vk); }
        catch (...) {
            if (!deviceRetirementReactor->retainEventSourceLost(shared_from_this()))
                throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                    "failed to retain D3B1 drain backing");
            throw;
        }
        *pendingCompletion = std::make_unique<ReactorFencePendingOperation>(
            shared_from_this(), deviceRetirementReactor, completionFd,
            std::vector<PresentedPhysicalImageAcquireToken>{}, false,
            runtimeGpuLifetime);
        return SwapchainPresentResult::internal(VK_SUCCESS);
    }
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY) {
        if (this->generatedOutputReturnDiagnosticSession
                && this->generatedOutputReturnDiagnosticSession
                    ->hasAcceptedBReturnFailure()) {
            const auto status = this->generatedOutputReturnDiagnosticSession
                ->tryRetireAcceptedBReturnFailure(
                    this->instance.get(), this->runtimeGenerateDiagnosticSession.get());
            if (status == backend::RuntimeRetirementStatus::NOT_READY)
                throw ls::error("D3B3 post-submit B-return retirement is pending");
            if (status == backend::RuntimeRetirementStatus::DEVICE_LOST) {
                if (this->terminalContext) this->terminalContext->quarantineDeviceLost();
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "D3B3 post-submit B-return retirement lost the device");
            }
            if (status != backend::RuntimeRetirementStatus::RETIRED)
                throw ls::error("D3B3 post-submit B-return retirement failed");
            const auto released = this->generatedOutputReturnDiagnosticSession
                ->releaseAcceptedBReturnFailure(
                    this->instance.get(), this->runtimeGenerateDiagnosticSession.get());
            if (released != backend::RuntimeRetirementStatus::RETIRED)
                throw ls::error("D3B3 retired B-return resources were not released");
            this->generatedOutputReturnDiagnosticSession.reset();
        }
        if (this->captureOnlyPhase == 0) {
            const auto sourceImage = this->info.images.at(imageIdx);
            std::cerr << "[DG2X-P4C-B0] Cross-device capture-only runtime\n"
                << "  RuntimeDevicePair: CROSS_PHYSICAL_DEVICE\n"
                << "  P4B runtime image channel: PASS\n"
                << "  Runtime mode: CAPTURE_ONLY\n"
                << "  Frame transport connected: NO\n"
                << "  vkCreateSwapchainKHR continuation: PASS\n"
                << "  VirtualSwapchainRuntime creation: PASS\n"
                << "  Virtual images exposed: PASS\n"
                << "  vkQueuePresentKHR reached: PASS\n"
                << "  Virtual image index: " << imageIdx << "\n"
                << "  Source VkImage: " << sourceImage << "\n"
                << "  Source format: " << static_cast<int>(this->info.format) << "\n"
                << "  Source extent: " << this->info.extent.width << 'x'
                << this->info.extent.height << "\n"
                << "  Present wait semaphore count at capture boundary: "
                << semaphores.size() << "\n"
                << "  bridgePresentWaits reached: PASS\n"
                << "  Capture hook reached: PASS\n"
                << "  Backend real-frame workload: NO\n"
                << "  LSFG execution on B: NO\n"
                << "  Presentation from B: NO\n"
                << "  Capture hook reached: PASS\n";
            if (!this->frameTransportReady)
                throw ls::error("P4C-C frame transport resources unavailable");
            vk::RuntimeImageEndpoint::executeRealFrameTransport(
                this->frameTransportA, this->frameTransportB,
                vk::RuntimeFrameTransportSource{sourceImage,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, this->info.format,
                    this->info.extent, 1, this->frameSourceLifetime},
                semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            this->instance.get().validateRuntimePrepass(
                this->frameTransportB.image(), this->info.extent,
                VK_FORMAT_B8G8R8A8_UNORM, this->frameTransportBacking.modifier(),
                1.0F / this->profile.flow_scale, this->profile.performance_mode);
            this->captureRealFrameOnce(vk, sourceImage, imageIdx, {});
            std::cerr << "DG2X_P4C_B0_CAPTURE_ONLY_RUNTIME_PASS\n"
                << "cross-device capture hook reached, but real frame transport is not connected yet\n";
            this->captureOnlyPhase = 1;
            return SwapchainPresentResult::internal(VK_SUCCESS);
        }
        if (this->captureOnlyPhase <= 2) {
            if (this->captureOnlyPhase == 1) {
                auto& session = this->instance.get().openRuntimePrepassSession(
                    this->info.extent, VK_FORMAT_B8G8R8A8_UNORM,
                    this->frameTransportBacking.modifier(),
                    1.0F / this->profile.flow_scale, this->profile.performance_mode);
                this->runtimePrepassSession = ls::owned_ptr<ls::R<backend::RuntimePrepassSession>>(
                    new ls::R<backend::RuntimePrepassSession>(session),
                    [backend = &this->instance.get()](ls::R<backend::RuntimePrepassSession>& value) {
                        backend->closeRuntimePrepassSession(value);
                    });
            }
            const auto sourceImage = this->info.images.at(imageIdx);
            auto transport = vk::RuntimeImageEndpoint::submitRealFrameTransportA(
                this->frameTransportA,
                vk::RuntimeFrameTransportSource{sourceImage,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, this->info.format,
                    this->info.extent, this->captureOnlyPhase + 1,
                    this->frameSourceLifetime},
                semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            this->instance.get().processRuntimePrepass(
                this->runtimePrepassSession.get(), this->frameTransportB.image(),
                std::move(transport));
            this->captureOnlyPhase++;
            return SwapchainPresentResult::internal(VK_SUCCESS);
        }
        if (this->captureOnlyPhase <= 5) {
            if (this->captureOnlyPhase == 3) {
                auto& session = this->instance.get().openRuntimeGenerateSession(
                    this->info.extent, VK_FORMAT_B8G8R8A8_UNORM,
                    this->frameTransportBacking.modifier(),
                    1.0F / this->profile.flow_scale, this->profile.performance_mode);
                this->runtimeGenerateDiagnosticSession =
                    ls::owned_ptr<ls::R<backend::RuntimeGenerateSession>>(
                        new ls::R<backend::RuntimeGenerateSession>(session),
                        [backend = &this->instance.get()](
                                ls::R<backend::RuntimeGenerateSession>& value) {
                            backend->closeRuntimeGenerateSession(value);
                        });
            }
            const auto sourceImage = this->info.images.at(imageIdx);
            auto transport = vk::RuntimeImageEndpoint::submitRealFrameTransportA(
                this->frameTransportA,
                vk::RuntimeFrameTransportSource{sourceImage,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, this->info.format,
                    this->info.extent, this->captureOnlyPhase + 1,
                    this->frameSourceLifetime},
                semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            auto generated = this->instance.get().submitRuntimeGenerateDiagnostic(
                this->runtimeGenerateDiagnosticSession.get(), this->frameTransportB.image(),
                std::move(transport));
            if (generated) {
                const bool d3bResourcesReady = queue != VK_NULL_HANDLE && queueMutex
                    && graphicsFinalQueue
                    && graphicsFinalQueue->family != VK_QUEUE_FAMILY_IGNORED
                    && this->virtualFinalCommandBuffer.has_value()
                    && this->virtualFinalAcquireSemaphore.has_value()
                    && this->virtualFinalPresentSemaphore.has_value()
                    && this->renderFence.has_value() && !this->info.realImages.empty();
                const bool d3b = d3b1PresentEligible(d3b1PresentationDiagnosticEnabled(),
                    d3bSingleSwapchainEligible, next_chain != nullptr,
                    this->info.virtualized, d3bResourcesReady, this->d3b1State);
                const bool d3b2 = d3b1PresentEligible(d3b2InsertionDiagnosticEnabled(),
                    d3bSingleSwapchainEligible, next_chain != nullptr,
                    this->info.virtualized, d3bResourcesReady
                        && this->info.releaseBackend != SwapchainReleaseBackend::None
                        && this->d3b2OriginalAcquireSemaphore.has_value()
                        && this->d3b2GeneratedPresentSemaphore.has_value()
                        && this->d3b2OriginalPresentSemaphore.has_value(),
                    this->d3b1State);
                const auto terminalConsumer = selectGeneratedOutputTerminalConsumer(d3b, d3b2);
                std::optional<vk::RuntimeForeignImageHandoffInfo> handoff;
                if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                    try {
                        if (!this->terminalHandoffBinding)
                            throw std::logic_error("terminal context unavailable");
                        this->returnedForGraphicsLease.emplace(
                            this->terminalHandoffBinding->acquire(
                                static_cast<uint64_t>(this->captureOnlyPhase + 1)));
                    }
                    catch (...) {
                        if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                            this->d3b1State = D3B1PresentationState::FAILED;
                        throw;
                    }
                    handoff = this->returnedForGraphicsLease->handoffInfo();
                }
                try {
                    this->generatedOutputReturnDiagnosticSession =
                        std::make_unique<GeneratedOutputReturnSession>(
                            ProductionReturnExecution{}, this->devicePair,
                            this->instance.get().runtimeExchangeEndpoint(),
                            vk::makeRuntimeExchangeEndpoint(vk), true);
                    this->generatedOutputReturnDiagnosticSession
                        ->executeProductionGpuChained(
                            std::move(*generated), this->instance.get(),
                            this->runtimeGenerateDiagnosticSession.get(), handoff);
                    if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly)
                        this->returnedForGraphicsLease->signalSubmitted();
                    if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly)
                        this->returnedForGraphicsWaitAuthority.emplace(
                            this->returnedForGraphicsLease->deriveWaitAuthority());
                } catch (...) {
                    if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                        const auto failure = std::current_exception();
                        bool deviceLost{};
                        try { std::rethrow_exception(failure); }
                        catch (const ls::vulkan_error& error) {
                            deviceLost = error.error() == VK_ERROR_DEVICE_LOST;
                        }
                        catch (...) {}
                        if (deviceLost && this->terminalContext)
                            this->terminalContext->quarantineDeviceLost();
                        if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                            this->d3b1State = D3B1PresentationState::FAILED;
                        this->returnedForGraphicsWaitAuthority.reset();
                        if (!this->generatedOutputReturnDiagnosticSession
                                || !this->generatedOutputReturnDiagnosticSession
                                    ->presentationHandoffPending())
                            this->returnedForGraphicsLease.reset();
                    }
                    throw;
                }
                if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                    if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                        this->d3b1State = D3B1PresentationState::A_HANDOFF_SUBMITTED;
                    const auto view = this->generatedOutputReturnDiagnosticSession
                        ->returnedImageView();
                    VkFormatProperties2 destinationProperties{
                        VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
                    vk.fi().GetPhysicalDeviceFormatProperties2(
                        vk.physdev(), this->info.format, &destinationProperties);
                    if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B2) {
                        if (!pendingCompletion)
                            throw std::logic_error("D3B2 async presenter completion is unavailable");
                        const auto self = shared_from_this();
                        const auto* vkPtr = &vk;
                        auto operationBacking =
                            std::make_shared<D3B2TerminalPendingGpuBacking>(
                                vk, graphicsFinalQueue->family);
                        auto detachedGpuBacking = combineGpuRetirementBacking(
                            operationBacking, runtimeGpuLifetime);
                        this->d3b2State = D3B2InsertionState::INACTIVE;
                        D3B2InsertionPath insertion{
                            .generated = {
                                .image = view.image(), .format = view.format(),
                                .extent = view.extent(), .layout = view.layout(),
                                .sourceQueueFamily = view.sourceQueueFamily(),
                                .destinationQueueFamily = view.destinationQueueFamily(),
                                .ownershipAcquireRequired = view.handoffPendingAcquire(),
                                .blitSourceSupported = view.valid()
                                    && modifierSupportsBlitSource(vk, view.format(), view.modifier())},
                            .original = {
                                .image = this->info.images.at(imageIdx),
                                .format = this->info.format, .extent = this->info.extent,
                                .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                .sourceQueueFamily = graphicsFinalQueue->family,
                                .destinationQueueFamily = graphicsFinalQueue->family,
                                .blitSourceSupported = (destinationProperties.formatProperties.optimalTilingFeatures
                                    & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0},
                            .originalReady = semaphores.size() > 1 ? semaphores.at(1) : VK_NULL_HANDLE,
                            .returnedForGraphics = this->returnedForGraphicsWaitAuthority->semaphore(),
                            .acquireGenerated = operationBacking->generatedAcquireSemaphore(),
                            .acquireOriginal = operationBacking->originalAcquireSemaphore(),
                            .generatedPresentReady = operationBacking->generatedPresentSemaphore(),
                            .originalPresentReady = operationBacking->originalPresentSemaphore(),
                            .graphicsFence = operationBacking->completionFence(),
                            .commandPoolFamily = this->virtualFinalCommandFamily,
                            .submitQueueFamily = graphicsFinalQueue->family,
                            .submitQueueFlags = graphicsFinalQueue->flags,
                            .singleSwapchain = d3bSingleSwapchainEligible,
                            .hasPNext = next_chain != nullptr,
                            .fifo = this->info.adaptivePresentMode == VK_PRESENT_MODE_FIFO_KHR,
                            .hiddenBlitDestinationSupported = (destinationProperties.formatProperties.optimalTilingFeatures
                                & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0,
                            .maintenanceReleaseCapable = this->info.releaseBackend
                                != SwapchainReleaseBackend::None,
                            .presentFences = {
                                operationBacking->generatedPresentationFence(),
                                operationBacking->originalPresentationFence()},
                            .state = &this->d3b2State};
                        insertion.acquire = [self, vkPtr, swapchain](VkSemaphore acquireSemaphore) {
                            uint32_t index{};
                            const auto result = vkPtr->df().AcquireNextImageKHR(vkPtr->dev(), swapchain,
                                2ULL * 1000 * 1000 * 1000, acquireSemaphore, VK_NULL_HANDLE, &index);
                            if (result != VK_SUCCESS)
                                throw ls::vulkan_error(result, "D3B2 hidden WSI acquire failed");
                            return D3B2HiddenImage{self->info.realImages.at(index), index, self->info.extent};
                        };
                        insertion.record = [self, vkPtr, view, imageIdx,
                                operationBacking](const D3B2HiddenImage& generated,
                                const D3B2HiddenImage& original) {
                            const auto& command = operationBacking->command();
                            command.begin(*vkPtr);
                            std::vector<vk::Barrier> generatedPre;
                            if (view.handoffPendingAcquire())
                                generatedPre.push_back(barrierHelper(view.image(), 0,
                                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    view.sourceQueueFamily(), view.destinationQueueFamily()));
                            generatedPre.push_back(barrierHelper(generated.image, 0,
                                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
                            command.blitImage(*vkPtr, generatedPre, {view.image(), generated.image},
                                {view.extent(), generated.extent}, {barrierHelper(generated.image,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)});
                            command.blitImage(*vkPtr, {
                                    barrierHelper(self->info.images.at(imageIdx), 0,
                                        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                                    barrierHelper(original.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)},
                                {self->info.images.at(imageIdx), original.image},
                                self->info.extent,
                                {barrierHelper(self->info.images.at(imageIdx),
                                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
                                 barrierHelper(original.image, VK_ACCESS_TRANSFER_WRITE_BIT,
                                    VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)});
                            command.end(*vkPtr);
                        };
                        const auto originalReady = insertion.originalReady;
                        const auto returnedForGraphics = insertion.returnedForGraphics;
                        const auto acquireGenerated = insertion.acquireGenerated;
                        const auto acquireOriginal = insertion.acquireOriginal;
                        const auto generatedPresentReady = insertion.generatedPresentReady;
                        const auto originalPresentReady = insertion.originalPresentReady;
                        const auto graphicsFence = insertion.graphicsFence;
                        insertion.submit = [self, vkPtr, queue, queueMutex, originalReady,
                                returnedForGraphics, acquireGenerated, acquireOriginal,
                                generatedPresentReady, originalPresentReady, graphicsFence,
                                operationBacking] {
                            const std::vector<VkSemaphore> waits{
                                originalReady, returnedForGraphics,
                                acquireGenerated, acquireOriginal};
                            const std::scoped_lock queueLock(*queueMutex);
                            submitReturnedForGraphicsTerminalWait(
                                *self->returnedForGraphicsWaitAuthority,
                                TerminalSubmitRole::D3B2TerminalWait,
                                self->terminalContext->contextIdentity(),
                                operationBacking->command(), *vkPtr, queue, waits,
                                {generatedPresentReady, originalPresentReady},
                                graphicsFence, VK_PIPELINE_STAGE_TRANSFER_BIT);
                            return VK_SUCCESS;
                        };
                        insertion.present = [vkPtr, queue, queueMutex, swapchain](
                                const D3B2HiddenImage& image, VkSemaphore ready) {
                            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .waitSemaphoreCount = 1, .pWaitSemaphores = &ready,
                                .swapchainCount = 1, .pSwapchains = &swapchain,
                                .pImageIndices = &image.index};
                            const std::scoped_lock queueLock(*queueMutex);
                            return vkPtr->df().QueuePresentKHR(queue, &info);
                        };
                        insertion.presentWithFence = [vkPtr, queue, queueMutex, swapchain](
                                const D3B2HiddenImage& image, VkSemaphore ready,
                                VkFence presentFence) {
                            const VkSwapchainPresentFenceInfoKHR fences{
                                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                                .swapchainCount = 1, .pFences = &presentFence};
                            const VkPresentInfoKHR info{
                                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .pNext = &fences,
                                .waitSemaphoreCount = 1, .pWaitSemaphores = &ready,
                                .swapchainCount = 1, .pSwapchains = &swapchain,
                                .pImageIndices = &image.index};
                            const std::scoped_lock queueLock(*queueMutex);
                            return vkPtr->df().QueuePresentKHR(queue, &info);
                        };
                        insertion.tryRetireGraphicsFence = [] { return VK_NOT_READY; };
                        insertion.generatedIntegrity = [self] {
                            self->generatedOutputReturnDiagnosticSession->completePresentationDiagnostics();
                            return self->generatedOutputReturnDiagnosticSession->gpuChainedPassed();
                        };
                        const auto originalImage = insertion.original.image;
                        insertion.originalIdentity = [self, imageIdx, originalImage] {
                            return originalImage == self->info.images.at(imageIdx);
                        };
                        insertion.retire = [self] {
                            self->generatedOutputReturnDiagnosticSession.reset();
                            self->returnedForGraphicsWaitAuthority->terminalWaitRetired();
                            self->returnedForGraphicsWaitAuthority.reset();
                            self->returnedForGraphicsLease.reset();
                        };
                        insertion.emitMarker = [] { std::cerr << "DG2X_P4C_D3B2_GENERATED_THEN_ORIGINAL_PRESENT_PASS\n"; };
                        insertion.releaseAcquiredImages = [self, vkPtr, swapchain](
                                const std::vector<uint32_t>& indices) {
                            VkReleaseSwapchainImagesInfoKHR releaseInfo{
                                .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
                                .swapchain = swapchain,
                                .imageIndexCount = static_cast<uint32_t>(indices.size()),
                                .pImageIndices = indices.data()};
                            if (self->info.releaseBackend == SwapchainReleaseBackend::Khr
                                    && vkPtr->df().ReleaseSwapchainImagesKHR)
                                return vkPtr->df().ReleaseSwapchainImagesKHR(vkPtr->dev(), &releaseInfo);
                            if (self->info.releaseBackend == SwapchainReleaseBackend::Ext
                                    && vkPtr->df().ReleaseSwapchainImagesEXT)
                                return vkPtr->df().ReleaseSwapchainImagesEXT(vkPtr->dev(), &releaseInfo);
                            return VK_ERROR_EXTENSION_NOT_PRESENT;
                        };
                        D3B2PendingInsertion pending;
                        const auto result = submitD3B2InsertionNonblocking(insertion, pending);
                        std::vector<int> completionFds;
                        try {
                            completionFds.push_back(
                                operationBacking->exportCompletionFd(vk));
                            completionFds.push_back(
                                operationBacking->exportGeneratedPresentationFd(vk));
                            completionFds.push_back(
                                operationBacking->exportOriginalPresentationFd(vk));
                        } catch (...) {
                            for (const auto fd : completionFds)
                                if (fd >= 0) static_cast<void>(::close(fd));
                            completionFds.clear();
                            if (deviceRetirementReactor)
                                if (!deviceRetirementReactor->retainEventSourceLost(
                                        detachedGpuBacking))
                                    throw ls::vulkan_error(
                                        VK_ERROR_OUT_OF_HOST_MEMORY,
                                        "failed to retain D3B2 accepted backing");
                        }
                        *pendingCompletion = std::make_unique<D3B2VirtualPresentPendingOperation>(
                            self, std::move(insertion), std::move(pending),
                            deviceRetirementReactor, detachedGpuBacking,
                            std::move(completionFds));
                        return SwapchainPresentResult::internal(result);
                    }
                    if (!pendingCompletion)
                        throw std::logic_error("D3B1 async presenter completion is unavailable");
                    const auto self = shared_from_this();
                    const auto* vkPtr = &vk;
                    auto operationBacking =
                        std::make_shared<D3B1TerminalPendingGpuBacking>(
                            vk, graphicsFinalQueue->family);
                    auto detachedGpuBacking = combineGpuRetirementBacking(
                        operationBacking, runtimeGpuLifetime);
                    D3B1PresentPath presentPath{
                        .source = {
                            .image = view.image(), .format = view.format(),
                            .extent = view.extent(), .layout = view.layout(),
                            .modifier = view.modifier(),
                            .sourceQueueFamily = view.sourceQueueFamily(),
                            .destinationQueueFamily = view.destinationQueueFamily(),
                            .ownershipAcquireRequired = view.handoffPendingAcquire()},
                        .hiddenAcquire = operationBacking->acquireSemaphore(),
                        .returnedForPresent = this->returnedForGraphicsWaitAuthority->semaphore(),
                        .finalPresentSemaphore = operationBacking->presentSemaphore(),
                        .renderFence = operationBacking->completionFence(),
                        .commandPoolFamily = this->virtualFinalCommandFamily,
                        .submitQueueFamily = graphicsFinalQueue->family,
                        .submitQueueFlags = graphicsFinalQueue->flags,
                        .sourceBlitSupported = view.valid()
                            && view.destinationQueueFamily() == graphicsFinalQueue->family
                            && modifierSupportsBlitSource(vk, view.format(), view.modifier()),
                        .destinationBlitSupported =
                            (destinationProperties.formatProperties.optimalTilingFeatures
                                & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0,
                        .state = &this->d3b1State};
                    presentPath.acquireHidden = [self, vkPtr, swapchain, stopToken,
                            operationBacking] {
                        uint32_t index{};
                        const auto result = acquireRealSwapchainImage(*vkPtr, swapchain,
                            operationBacking->acquireSemaphore(), &index, stopToken);
                        if (result != VK_SUCCESS)
                            throw ls::vulkan_error(result, "D3B1 hidden WSI acquire failed");
                        return D3B1HiddenImage{
                            self->info.realImages.at(index), index, self->info.extent};
                    };
                    presentPath.recordBlit = [vkPtr, operationBacking](const auto& pre,
                            VkImage source,
                            VkImage destination, VkExtent2D sourceExtent,
                            VkExtent2D destinationExtent, const auto& post) {
                        const auto& finalCommand = operationBacking->command();
                        finalCommand.begin(*vkPtr);
                        finalCommand.blitImage(*vkPtr, pre, {source, destination},
                            {sourceExtent, destinationExtent}, post);
                        finalCommand.end(*vkPtr);
                    };
                    presentPath.submitAndPresent = [self, vkPtr, queue, queueMutex,
                            swapchain, operationBacking](
                            const auto& waits, const auto&,
                            VkSemaphore signal, VkFence fence, uint32_t index) {
                        const auto presentFence = operationBacking->presentationFence();
                        const VkSwapchainPresentFenceInfoKHR fences{
                            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                            .swapchainCount = 1, .pFences = &presentFence};
                        const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                            .pNext = &fences,
                            .waitSemaphoreCount = 1, .pWaitSemaphores = &signal,
                            .swapchainCount = 1, .pSwapchains = &swapchain,
                            .pImageIndices = &index};
                        const std::scoped_lock queueLock(*queueMutex);
                        submitReturnedForGraphicsTerminalWait(
                            *self->returnedForGraphicsWaitAuthority,
                            TerminalSubmitRole::D3B1TerminalConsumption,
                            self->terminalContext->contextIdentity(),
                            operationBacking->command(), *vkPtr, queue, waits,
                            {signal}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
                        const auto presentResult = vkPtr->df().QueuePresentKHR(queue, &info);
                        return D3B1SubmitPresentResult{presentResult, true};
                    };
                    presentPath.tryRetireRenderFence = [] { return VK_NOT_READY; };
                    presentPath.completeDiagnostics = [self] {
                        self->generatedOutputReturnDiagnosticSession
                            ->completePresentationDiagnostics();
                        const auto result = self->generatedOutputReturnDiagnosticSession
                            ->gpuChainedPassed();
                        return result;
                    };
                    presentPath.retire = [self] {
                        self->generatedOutputReturnDiagnosticSession.reset();
                        self->returnedForGraphicsWaitAuthority->terminalWaitRetired();
                        self->returnedForGraphicsWaitAuthority.reset();
                        self->returnedForGraphicsLease.reset();
                    };
                    presentPath.emitMarker = [] {
                        std::cerr << "DG2X_P4C_D3B1_RETURNED_GENERATED_FRAME_PRESENT_PASS\n";
                    };
                    D3B1PendingPresent pending;
                    const auto result = submitD3B1PresentNonblocking(presentPath, pending);
                    std::vector<int> completionFds;
                    try {
                        completionFds.push_back(
                            operationBacking->exportCompletionFd(vk));
                        completionFds.push_back(
                            operationBacking->exportPresentationFd(vk));
                    } catch (...) {
                        for (const auto fd : completionFds)
                            if (fd >= 0) static_cast<void>(::close(fd));
                        completionFds.clear();
                        if (deviceRetirementReactor)
                            if (!deviceRetirementReactor->retainEventSourceLost(
                                        detachedGpuBacking))
                                throw ls::vulkan_error(
                                    VK_ERROR_OUT_OF_HOST_MEMORY,
                                    "failed to retain D3B1 accepted backing");
                    }
                    *pendingCompletion = std::make_unique<D3B1VirtualPresentPendingOperation>(
                        self, std::move(presentPath), std::move(pending),
                        deviceRetirementReactor, detachedGpuBacking,
                        std::move(completionFds));
                    return SwapchainPresentResult::internal(result);
                }
            }
            this->captureOnlyPhase++;
            if (this->captureOnlyPhase <= 5)
                return SwapchainPresentResult::internal(VK_SUCCESS);
        }
        throw ls::error(
            d3b1PresentationDiagnosticEnabled()
                ? "D3B1_ASYNC_PRESENTATION_NOT_YET_CONNECTED"
                : "cross-device generated-frame diagnostic completed; output transport remains disconnected");
    }
    const auto& swapchainImage = this->info.images.at(imageIdx);
    const auto& outputImages = this->info.virtualized
        ? this->info.realImages
        : this->info.images;

    const bool fixedMode = isFixedMode(this->profile);
    // Every virtual topology presents from the prepared offload queue. Fixed
    // pacing itself remains separately gated by fixedMode.
    const bool workerOffload = this->info.virtualized && queueMutex;

    // Stage 3C5G: select the hidden real WSI present mode per presentation.
    // These modes were declared at swapchain creation, so hot reload changes
    // only the internal context and never the VkImages known to the app.
    const VkPresentModeKHR selectedPresentMode = fixedMode
        ? this->info.fixedPresentMode
        : this->info.adaptivePresentMode;
    VkSwapchainPresentModeInfoKHR dynamicPresentModeInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
        .swapchainCount = 1,
        .pPresentModes = &selectedPresentMode
    };
    const auto presentNextChain = [&](void* logicalNext) -> const void* {
        if (!this->info.dynamicPresentModeEligible)
            return logicalNext;

        // Avoid duplicating the structure when the application already owns
        // one. Existing pacing=None behavior below rewrites that structure to
        // the mode selected by the current LSFG context.
        if (hasPresentModeInfo(logicalNext))
            return logicalNext;

        dynamicPresentModeInfo.pNext = logicalNext;
        return &dynamicPresentModeInfo;
    };

    const auto compensateAbortedLogicalPresentFence =
        [&](const char* stage) -> VkResult {
            const auto fence = applicationPresentFence(next_chain);
            if (fence == VK_NULL_HANDLE)
                return VK_SUCCESS;

            // The application-facing vkQueuePresentKHR is about to return
            // OUT_OF_DATE before the logical present carrying this fence could
            // reach WSI. Complete the consumed logical present fence after all
            // earlier hidden queue work so the application can safely retire
            // its present resources and recreate the swapchain.
            const auto submit = [&](VkQueue actualQueue,
                    const VkSubmitInfo& submitInfo, VkFence actualFence) {
                if (workerOffload) {
                    const std::scoped_lock queueLock(*queueMutex);
                    return vk.df().QueueSubmit(
                        actualQueue, 1, &submitInfo, actualFence);
                }
                return vk.df().QueueSubmit(
                    actualQueue, 1, &submitInfo, actualFence);
            };
            const auto signalResult = compensateEnqueuedPresentAbort(
                queue, fence, submit);

            if (signalResult == VK_SUCCESS) {
                std::cerr << "lsfg-vk: compensated aborted logical present fence after "
                    << stage << " OUT_OF_DATE\n";
            } else {
                std::cerr << "lsfg-vk: failed to compensate aborted logical present fence after "
                    << stage << " OUT_OF_DATE: " << signalResult << '\n';
            }
            return signalResult;
        };

    // 3C5E: Adaptive 1x over the stable virtual topology. The runtime has
    // already consumed the application's present waits and supplies one ready
    // semaphore. No LSFG generation images or backend context exist here.
    if (transformationBypass) {
        if (semaphores.size() != 1 || !queueMutex || !pendingCompletion
                || !this->virtualFinalCommandBuffer.has_value()
                || !this->virtualFinalAcquireSemaphore.has_value()
                || !this->virtualFinalPresentSemaphore.has_value()) {
            throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                "virtual Adaptive 1x bridge is not initialized");
        }

        if (this->info.releaseBackend == SwapchainReleaseBackend::None)
            throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
                "virtual Adaptive 1x recovery requires swapchain maintenance release");

        const auto applicationFence = applicationPresentFence(next_chain);
        auto completionPublication = std::make_shared<
            std::unique_ptr<VirtualPresentPendingOperation>>();
        auto reservation = this->reserveAdaptive1xPreparation(vk, queue,
            queueMutex, swapchain, imageIdx, applicationFence, stopToken,
            runtimeGpuLifetime, deviceRetirementReactor,
            borrowedPresentFences, completionPublication);
        const auto executeReservation = [&]() -> Adaptive1xPreparedLogicalFinal {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
            struct ExecuteHookGuard final {
                void (*after)(){};
                ~ExecuteHookGuard() { if (after) after(); }
            };
            const auto before = adaptiveReservationBeforeExecute.load(
                std::memory_order_acquire);
            const auto after = adaptiveReservationAfterExecute.load(
                std::memory_order_acquire);
            if (before) before();
            ExecuteHookGuard hookGuard{after};
#endif
            return reservation.execute(semaphores.front());
        };
        auto prepared = [&]() -> Adaptive1xPreparedLogicalFinal {
            try {
                return executeReservation();
            } catch (const ls::vulkan_error& e) {
                // Preserve the qualified single-swapchain behavior: if the
                // hidden physical Acquire rejects an otherwise consumed logical
                // present, complete only this logical present's application
                // fence. Future multi-swapchain batch code must handle this at
                // whole-batch scope instead of reusing this per-entry policy.
                if (classifyPresentResult(e.error())
                        == PresentResultClass::EnqueuedRejection) {
                    const auto fenceResult = compensateAbortedLogicalPresentFence(
                        "hidden Adaptive 1x acquire");
                    if (fenceResult != VK_SUCCESS)
                        throw ls::vulkan_error(fenceResult,
                            "failed to compensate aborted logical present fence");
                }
                throw;
            }
        }();
        VkResult res{VK_SUCCESS};

        OwnedInternalPresentFenceProjection internalFenceProjection;
        if (prepared.requiresInternalPresentFence()
                && !internalFenceProjection.prepare(next_chain,
                    prepared.internalPresentFence())) {
            static_cast<void>(prepared.abandonWithoutLogicalPresent());
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                "failed to prepare owned internal present-fence projection");
        }
        if (!prepared.markLogicalPresentCalled())
            throw ls::error("virtual Adaptive logical-present call authority was lost");
        const auto physicalSwapchain = prepared.physicalSwapchain();
        const auto physicalImageIndex = prepared.physicalImageIndex();
        const auto readySemaphore = prepared.presentReadySemaphore();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = presentNextChain(prepared.requiresInternalPresentFence()
                ? const_cast<void*>(internalFenceProjection.head) : next_chain),
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &readySemaphore,
            .swapchainCount = 1,
            .pSwapchains = &physicalSwapchain,
            .pImageIndices = &physicalImageIndex};
        // Preparation deliberately does not carry queue-lock ownership across
        // this boundary. Both operations target the same queue (whose Vulkan
        // submission order is externally synchronized by each short scope),
        // and the logical present additionally waits on this operation's
        // private producer-ready binary semaphore. This permits multiple
        // preparations to coexist without weakening producer->present order.
        {
            const std::scoped_lock queueLock(*queueMutex);
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        }
        const auto presentClass = classifyPresentResult(res);
        if ((presentClass == PresentResultClass::Normal
                || presentClass == PresentResultClass::EnqueuedRejection)
                && presentedIdentity)
            *presentedIdentity = prepared.presentedIdentity();
        const auto finalized = prepared.finalizeLogicalPresent(res);
        if (finalized != VK_SUCCESS)
            throw ls::vulkan_error(finalized,
                "virtual Adaptive logical-present finalization failed");
        if (*completionPublication)
            *pendingCompletion = std::move(*completionPublication);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LogicalPresentError(res, "logical vkQueuePresentKHR() failed");
        return SwapchainPresentResult::logical(res);
    }

    const auto& sourceImage = this->sourceImages.at(this->fidx % 2);
    FixedFrameScheduler::Plan fixedPlan{};
    if (fixedMode) {
        const auto now = sourcePresentTime.value_or(std::chrono::steady_clock::now());
        if (this->lastSourcePresent.has_value()) {
            fixedPlan = this->fixedScheduler.plan(
                std::chrono::duration_cast<FixedFrameScheduler::Duration>(
                    now - *this->lastSourcePresent));
        }
        this->lastSourcePresent = now;

        // If the application itself is faster than the requested output target,
        // pace the real-frame path instead of attempting negative interpolation.
        if (!workerOffload
                && fixedPlan.sourceDelay > FixedFrameScheduler::Duration::zero())
            std::this_thread::sleep_for(fixedPlan.sourceDelay);
    }

    const size_t generatedFrames = fixedMode
        ? fixedPlan.timestamps.size()
        : this->destinationImages.size();
    const auto timeline = vk::makeExchangeTimelineFrame(this->idx, generatedFrames);

    // Fixed virtual mode must not rely exclusively on FIFO/vblank to space
    // generated output. Use a cancellable host-side target cadence; the WSI
    // present mode is intentionally left unchanged by this pacing step.
    const auto paceFixedWorkerOutput = [&]() {
        if (!workerOffload || !fixedMode)
            return;

        constexpr auto MAX_SLEEP_SLICE = std::chrono::milliseconds(2);
        while (true) {
            if (stopToken.stop_requested())
                throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR,
                    "fixed output pacing worker stopped");

            const auto delay = this->fixedOutputPacer.delayUntilNext(
                std::chrono::steady_clock::now());
            if (delay <= FixedOutputPacer::Duration::zero())
                return;

            std::this_thread::sleep_for(std::min(
                delay,
                std::chrono::duration_cast<FixedOutputPacer::Duration>(
                    MAX_SLEEP_SLICE)));
        }
    };

    const auto markFixedWorkerOutput = [&]() {
        if (workerOffload && fixedMode)
            this->fixedOutputPacer.markPresented(
                std::chrono::steady_clock::now());
    };

    // Schedule frame generation. Adaptive continues through the original API.
    // Fixed uses the dynamic backend, including zero generated frames, so the
    // backend's temporal history advances for every real application frame.
    try {
        if (fixedMode)
            this->instance.get().scheduleFrames(this->ctx.get(), fixedPlan.timestamps);
        else
            this->instance.get().scheduleFrames(this->ctx.get());
    } catch (const std::exception& e) {
        throw ls::error("failed to schedule frames", e);
    }

    // Application-provided per-present mode structures are immutable input.
    // When the application owns one, preserve it exactly; LSFG injects its
    // internal mode only when the chain has no such structure.

    // wait for completion of previous frame
    if (this->fidx) {
        const auto previous = vk.df().GetFenceStatus(
            vk.dev(), this->renderFence->handle());
        if (previous != VK_SUCCESS)
            throw ls::vulkan_error(previous,
                "previous frame completion is not retired");
    }
    this->renderFence->reset(vk);

    // copy application-visible swapchain image into backend source image
    const auto& cmdbuf = *this->renderCommandBuffer;
    cmdbuf.begin(vk);

    const auto sourcePreBarrier = this->fidx == 0
        ? barrierHelper(sourceImage.handle(),
            VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        : vk::sourceAcquireFromBackend(
            sourceImage.handle(), vk.queueFamilyIndex());
    cmdbuf.blitImage(vk,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                fullSwapchainRange(this->info.arrayLayers)
            ),
            sourcePreBarrier,
        },
        { swapchainImage, sourceImage.handle() },
        sourceImage.getExtent(),
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            ),
            vk::sourceReleaseToBackend(
                sourceImage.handle(), vk.queueFamilyIndex()),
        }
    );

    // Generate samples both source descriptors. Seed the second alternating
    // resource with the first real frame so B can legally acquire/read both on
    // its first dispatch; subsequent frames update one resource at a time.
    if (this->fidx == 0) {
        const auto& secondSource = this->sourceImages.at(1);
        cmdbuf.blitImage(vk,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_NONE, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                barrierHelper(secondSource.handle(),
                    VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
            },
            { swapchainImage, secondSource.handle() },
            secondSource.getExtent(),
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
                vk::sourceReleaseToBackend(
                    secondSource.handle(), vk.queueFamilyIndex()),
            });
    }

    cmdbuf.end(vk);

    // With zero generated frames there is no generated post-copy pass. Keep a
    // binary semaphore to chain either the legacy final present or the virtual
    // real-frame copy.
    vk::Semaphore* zeroPresentSemaphore{};
    const auto sourceReturn = this->sourceReturnValues.at(this->fidx % 2);
    const VkSemaphore sourceWaitSemaphore = sourceReturn.has_value()
        ? this->syncSemaphore->handle() : VK_NULL_HANDLE;
    const uint64_t sourceWaitValue = sourceReturn.value_or(0);
    if (generatedFrames == 0) {
        auto& pcs = this->postCopySemaphores.at(
            this->idx % this->postCopySemaphores.size());
        zeroPresentSemaphore = &pcs.second;
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), timeline.sourceReady,
                this->info.virtualized ? VK_NULL_HANDLE : this->renderFence->handle(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        }
    } else {
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                {}, this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                {}, this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        }
    }
    this->idx = timeline.sourceReady + 1;
    this->sourceReturnValues.at((this->fidx + 1) % 2) = timeline.sourceReturn;

    std::vector<PresentedPhysicalImageAcquireToken> consumedPhysicalAcquires;
    for (size_t i = 0; i < generatedFrames; i++) {
        auto& pcs = this->postCopySemaphores.at(this->idx % this->postCopySemaphores.size());
        auto& destinationImage = this->destinationImages.at(i);
        auto& pass = this->passes.at(i);

        // acquire underlying real swapchain image
        uint32_t aqImageIdx{};
        auto res = acquireRealSwapchainImage(vk, swapchain,
            pass.acquireSemaphore.handle(), &aqImageIdx,
            workerOffload ? stopToken : std::stop_token{});
        if (classifyPresentResult(res) == PresentResultClass::EnqueuedRejection) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden generated-frame acquire");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

        std::optional<PresentedPhysicalImageAcquireToken> priorPhysicalLease;
        if (this->info.virtualized) {
            priorPhysicalLease = this->presentedPhysicalImages->reacquired(
                this->terminalContext->deviceIdentity(),
                this->physicalSwapchainLifecycleIdentity,
                reinterpret_cast<uintptr_t>(swapchain), aqImageIdx,
                nextRetirementOperationIdentity.fetch_add(
                    1, std::memory_order_relaxed));
        }

        const auto& acquiredSwapchainImage = outputImages.at(aqImageIdx);

        // copy backend destination image into real swapchain image
        auto& passCmdbuf = pass.commandBuffer;
        passCmdbuf.begin(vk);

        passCmdbuf.blitImage(vk,
            {
                vk::destinationAcquireFromBackend(
                    destinationImage.handle(), vk.queueFamilyIndex()),
                barrierHelper(acquiredSwapchainImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                ),
            },
            { destinationImage.handle(), acquiredSwapchainImage },
            destinationImage.getExtent(),
            {
                vk::destinationReleaseToBackend(
                    destinationImage.handle(), vk.queueFamilyIndex()),
                barrierHelper(acquiredSwapchainImage,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                ),
            }
        );

        std::vector<VkSemaphore> waitSemaphores{ pass.acquireSemaphore.handle() };
        if (i) {
            const auto& prevPCS = this->postCopySemaphores.at(
                (this->idx - 1) % this->postCopySemaphores.size());
            waitSemaphores.push_back(prevPCS.second.handle());
        }

        std::shared_ptr<PresentedWsiSemaphoreBacking> generatedPresentBacking;
        VkSemaphore generatedPresentReady = pcs.first.handle();
        if (this->info.virtualized) {
            generatedPresentBacking =
                std::make_shared<PresentedWsiSemaphoreBacking>(vk);
            generatedPresentReady = generatedPresentBacking->ready.handle();
        }
        const std::vector<VkSemaphore> signalSemaphores{
            generatedPresentReady, pcs.second.handle()};

        if (priorPhysicalLease) {
            consumedPhysicalAcquires.push_back(*priorPhysicalLease);
        }

        std::optional<PresentedPhysicalImageIdentity> generatedPresented;
        std::optional<PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease>
            generatedPreparedLease;
        std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
            generatedPreparedFence;
        OwnedInternalPresentFenceProjection generatedInternalProjection;
        bool generatedUsesInternalFence{};
        if (generatedPresentBacking) {
            generatedPresented = PresentedPhysicalImageIdentity{
                .deviceLifetimeIdentity = this->terminalContext->deviceIdentity(),
                .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
                .physicalSwapchainIdentity = reinterpret_cast<uintptr_t>(swapchain),
                .physicalImageIndex = aqImageIdx,
                .presentOperationIdentity = nextRetirementOperationIdentity.fetch_add(
                    1, std::memory_order_relaxed)};
            const auto generatedFence = applicationPresentFence(
                this->info.virtualized ? nullptr : ((!i) ? next_chain : nullptr));
            PresentedPhysicalImageLeaseRegistry::PrepareFailure leaseFailure{};
            generatedPreparedLease = this->presentedPhysicalImages->prepare(
                *generatedPresented, generatedPresentBacking, &leaseFailure,
                generatedFence == VK_NULL_HANDLE
                    ? std::function<VkResult()>{[backing = generatedPresentBacking,
                            vkPtr = &vk] {
                        return backing->waitForCompletion(*vkPtr);
                    }} : std::function<VkResult()>{});
            if (!generatedPreparedLease) {
                if (leaseFailure == PresentedPhysicalImageLeaseRegistry::PrepareFailure::OutOfHostMemory)
                    throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                        "failed to prepare generated physical-image WSI lease");
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "generated physical image already owns a WSI lease reservation");
            }
            if (generatedFence != VK_NULL_HANDLE && borrowedPresentFences) {
                BorrowedPresentFenceRegistry::PrepareFailure fenceFailure{};
                generatedPreparedFence = borrowedPresentFences->prepare(
                    generatedFence, this->presentedPhysicalImages,
                    *generatedPresented, &fenceFailure);
                if (!generatedPreparedFence) {
                    this->presentedPhysicalImages->abort(*generatedPreparedLease);
                    if (fenceFailure == BorrowedPresentFenceRegistry::PrepareFailure::OutOfHostMemory)
                        throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                            "failed to prepare generated present-fence generation");
                    throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                        "generated application present fence is untracked");
                }
            }
            generatedUsesInternalFence = generatedFence == VK_NULL_HANDLE;
            const void* generatedLogicalChain = this->info.virtualized
                ? nullptr : ((!i) ? next_chain : nullptr);
            if (generatedUsesInternalFence
                    && !generatedInternalProjection.prepare(generatedLogicalChain,
                        generatedPresentBacking->internalPresentFence.handle())) {
                if (generatedPreparedFence)
                    borrowedPresentFences->abort(*generatedPreparedFence);
                this->presentedPhysicalImages->abort(*generatedPreparedLease);
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "failed to prepare generated internal present-fence projection");
            }
        }

        // All ownership needed to retain the producer semaphore now exists;
        // only then may the producer submit signal it.
        passCmdbuf.end(vk);
        auto& destinationReturn = this->destinationReturnStates.at(i);
        vk::LayerDestinationReturnSubmitStorage generatedSubmitStorage;
        vk::prepareLayerDestinationReturnSubmit(waitSemaphores, signalSemaphores,
            this->syncSemaphore->handle(), timeline.destinationReady(i),
            this->destinationReturnSemaphores.at(i).handle(), destinationReturn,
            generatedSubmitStorage);
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            passCmdbuf.submit(vk, queue, generatedSubmitStorage.submission);
        } else {
            passCmdbuf.submit(vk, generatedSubmitStorage.submission,
                (!this->info.virtualized && i == generatedFrames - 1)
                    ? this->renderFence->handle() : VK_NULL_HANDLE);
        }
        // The generation becomes authoritative only after QueueSubmit accepts
        // the release-to-backend and matching GPU timeline signal.
        destinationReturn.returnSubmitAccepted();

        // Generated frames never carry the application's pNext when the
        // application is rendering into virtual images. The logical present
        // metadata belongs to the final real application frame.
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = generatedUsesInternalFence
                ? presentNextChain(const_cast<void*>(generatedInternalProjection.head))
                : (this->info.virtualized
                    ? presentNextChain(nullptr) : ((!i) ? next_chain : nullptr)),
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &generatedPresentReady,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &aqImageIdx,
        };
        paceFixedWorkerOutput();
        try { if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        } else {
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        } } catch (...) {
            if (generatedPreparedFence)
                borrowedPresentFences->abort(*generatedPreparedFence);
            if (generatedPreparedLease)
                this->presentedPhysicalImages->abort(*generatedPreparedLease);
            throw;
        }
        if (generatedPreparedLease) {
            const auto generatedClass = classifyPresentResult(res);
            if (generatedClass == PresentResultClass::PreEnqueueFailure) {
                if (generatedPresentBacking && deviceRetirementReactor
                        && !deviceRetirementReactor->retainEventSourceLost(
                            generatedPresentBacking))
                    throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                        "failed to retain generated producer backing after present OOM");
                if (generatedPreparedFence)
                    borrowedPresentFences->abort(*generatedPreparedFence);
                this->presentedPhysicalImages->abort(*generatedPreparedLease);
            } else if (!this->presentedPhysicalImages->commit(*generatedPreparedLease)) {
                throw ls::error("prepared generated WSI lease commit was lost");
            } else if (generatedPreparedFence
                    && (generatedClass == PresentResultClass::Normal
                        || generatedClass == PresentResultClass::EnqueuedRejection)) {
                static_cast<void>(borrowedPresentFences->commit(*generatedPreparedFence));
                if (generatedPreparedFence->valid())
                    throw ls::error("prepared generated present-fence commit was lost");
            } else if (generatedPreparedFence) {
                borrowedPresentFences->abort(*generatedPreparedFence);
            }
            if (generatedUsesInternalFence
                    && (generatedClass == PresentResultClass::Normal
                        || generatedClass == PresentResultClass::EnqueuedRejection))
                static_cast<void>(armInternalPresentFence(vk,
                    generatedPresentBacking, this->presentedPhysicalImages,
                    *generatedPresented, deviceRetirementReactor));
        }
        // In the virtual topology generated-frame presents never carry the
        // application's pNext chain. If one goes OUT_OF_DATE, the later logical
        // present will not run and its present fence would otherwise remain
        // unsignaled. Legacy non-virtual generated presents may already have
        // forwarded the application's pNext, so never compensate those here.
        if (this->info.virtualized
                && classifyPresentResult(res) == PresentResultClass::EnqueuedRejection) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden generated-frame present");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "hidden generated vkQueuePresentKHR() failed");
        markFixedWorkerOutput();

        this->idx++;
    }
    this->idx = timeline.nextBase;

    const VkSemaphore finalWaitSemaphore = generatedFrames
        ? this->postCopySemaphores.at(
            (this->idx - 1) % this->postCopySemaphores.size()).second.handle()
        : zeroPresentSemaphore->handle();

    if (!this->info.virtualized) {
        // Legacy Adaptive/Fixed-3B path: application image is a real WSI image.
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = generatedFrames ? nullptr : next_chain,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &finalWaitSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIdx,
        };
        auto res = vk.df().QueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LogicalPresentError(res, "logical vkQueuePresentKHR() failed");

        this->fidx++;
        return SwapchainPresentResult::logical(res);
    }

    // Virtual swapchain path: the application's image cannot be passed to WSI.
    // The virtual presentation worker acquires a real image, copies the logical source frame
    // into it and presents on the dedicated queue. The virtual image is not
    // recycled until the GPU has finished reading it.
    uint32_t realImageIdx{};
    auto res = acquireRealSwapchainImage(vk, swapchain,
        this->virtualFinalAcquireSemaphore->handle(), &realImageIdx,
        workerOffload ? stopToken : std::stop_token{});
    if (classifyPresentResult(res) == PresentResultClass::EnqueuedRejection) {
        const auto fenceResult =
            compensateAbortedLogicalPresentFence("hidden final acquire");
        if (fenceResult != VK_SUCCESS)
            throw ls::vulkan_error(fenceResult,
                "failed to compensate aborted logical present fence");
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

    if (const auto token = this->presentedPhysicalImages->reacquired(
            this->terminalContext->deviceIdentity(),
            this->physicalSwapchainLifecycleIdentity,
            reinterpret_cast<uintptr_t>(swapchain), realImageIdx,
            nextRetirementOperationIdentity.fetch_add(
                1, std::memory_order_relaxed)))
        consumedPhysicalAcquires.push_back(*token);
    auto finalPresentBacking =
        std::make_shared<PresentedWsiSemaphoreBacking>(vk);

    const auto& realImage = outputImages.at(realImageIdx);
    const auto& finalCmdbuf = *this->virtualFinalCommandBuffer;
    finalCmdbuf.begin(vk);
    finalCmdbuf.blitImage(vk,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ),
            barrierHelper(realImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                fullSwapchainRange(this->info.arrayLayers)
            ),
        },
        { swapchainImage, realImage },
        this->info.extent,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                fullSwapchainRange(this->info.arrayLayers)
            ),
            barrierHelper(realImage,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                fullSwapchainRange(this->info.arrayLayers)
            ),
        },
        this->info.arrayLayers
    );
    finalCmdbuf.end(vk);

    const PresentedPhysicalImageIdentity finalPresented{
        .deviceLifetimeIdentity = this->terminalContext->deviceIdentity(),
        .swapchainLifecycleIdentity = this->physicalSwapchainLifecycleIdentity,
        .physicalSwapchainIdentity = reinterpret_cast<uintptr_t>(swapchain),
        .physicalImageIndex = realImageIdx,
        .presentOperationIdentity = nextRetirementOperationIdentity.fetch_add(
            1, std::memory_order_relaxed)};
    PresentedPhysicalImageLeaseRegistry::PrepareFailure leaseFailure{};
    const auto applicationFence = applicationPresentFence(next_chain);
    auto preparedLease = this->presentedPhysicalImages->prepare(
        finalPresented, finalPresentBacking, &leaseFailure,
        applicationFence == VK_NULL_HANDLE
            ? std::function<VkResult()>{[backing = finalPresentBacking, vkPtr = &vk] {
                return backing->waitForCompletion(*vkPtr);
            }} : std::function<VkResult()>{});
    if (!preparedLease) {
        if (leaseFailure == PresentedPhysicalImageLeaseRegistry::PrepareFailure::OutOfHostMemory)
            throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                "failed to prepare final physical-image WSI lease");
        throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
            "final physical image already owns a WSI lease reservation");
    }
    std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
        preparedFence;
    if (applicationFence != VK_NULL_HANDLE && borrowedPresentFences) {
        BorrowedPresentFenceRegistry::PrepareFailure fenceFailure{};
        preparedFence = borrowedPresentFences->prepare(applicationFence,
            this->presentedPhysicalImages, finalPresented, &fenceFailure);
        if (!preparedFence) {
            this->presentedPhysicalImages->abort(*preparedLease);
            if (fenceFailure == BorrowedPresentFenceRegistry::PrepareFailure::OutOfHostMemory)
                throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                    "failed to prepare final borrowed present-fence generation");
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
            "final application present fence has no current object identity");
        }
    }
    OwnedInternalPresentFenceProjection internalFenceProjection;
    const bool useInternalFence = applicationFence == VK_NULL_HANDLE;
    if (useInternalFence && !internalFenceProjection.prepare(next_chain,
            finalPresentBacking->internalPresentFence.handle())) {
        if (preparedFence) borrowedPresentFences->abort(*preparedFence);
        this->presentedPhysicalImages->abort(*preparedLease);
        throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
            "failed to prepare final internal present-fence projection");
    }

    try { if (workerOffload) {
        const std::scoped_lock queueLock(*queueMutex);
        finalCmdbuf.submit(vk, queue,
            {
                finalWaitSemaphore,
                this->virtualFinalAcquireSemaphore->handle()
            },
            VK_NULL_HANDLE, 0,
            { finalPresentBacking->ready.handle() },
            VK_NULL_HANDLE, 0,
            this->renderFence->handle()
        );
    } else {
        finalCmdbuf.submit(vk,
            {
                finalWaitSemaphore,
                this->virtualFinalAcquireSemaphore->handle()
            },
            VK_NULL_HANDLE, 0,
            { finalPresentBacking->ready.handle() },
            VK_NULL_HANDLE, 0,
            this->renderFence->handle()
        );
    } } catch (...) {
        if (preparedFence) borrowedPresentFences->abort(*preparedFence);
        this->presentedPhysicalImages->abort(*preparedLease);
        throw;
    }

    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = presentNextChain(useInternalFence
            ? const_cast<void*>(internalFenceProjection.head) : next_chain),
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &finalPresentBacking->ready.handle(),
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &realImageIdx,
    };
    paceFixedWorkerOutput();
    if (workerOffload) {
        const std::scoped_lock queueLock(*queueMutex);
        res = vk.df().QueuePresentKHR(queue, &presentInfo);
    } else {
        res = vk.df().QueuePresentKHR(queue, &presentInfo);
    }
    for (const auto& token : consumedPhysicalAcquires) {
        if (!this->presentedPhysicalImages->markAcquireConsumed(token))
            throw ls::error("final acquire consumption authority was lost");
    }
    const auto presentClass = classifyPresentResult(res);
    if (presentClass == PresentResultClass::PreEnqueueFailure) {
        if (deviceRetirementReactor
                && !deviceRetirementReactor->retainEventSourceLost(finalPresentBacking))
            throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                "failed to retain producer backing after present OOM");
        if (preparedFence) borrowedPresentFences->abort(*preparedFence);
        this->presentedPhysicalImages->abort(*preparedLease);
    } else if (!this->presentedPhysicalImages->commit(*preparedLease)) {
        throw ls::error("prepared final physical-image WSI lease commit was lost");
    } else if (preparedFence && (presentClass == PresentResultClass::Normal
            || presentClass == PresentResultClass::EnqueuedRejection)) {
        static_cast<void>(borrowedPresentFences->commit(*preparedFence));
        if (preparedFence->valid())
            throw ls::error("prepared final borrowed present-fence commit was lost");
    } else if (preparedFence) {
        borrowedPresentFences->abort(*preparedFence);
    }
    if (useInternalFence && (presentClass == PresentResultClass::Normal
            || presentClass == PresentResultClass::EnqueuedRejection))
        static_cast<void>(armInternalPresentFence(vk, finalPresentBacking,
            this->presentedPhysicalImages, finalPresented,
            deviceRetirementReactor));
    if ((presentClass == PresentResultClass::Normal
            || presentClass == PresentResultClass::EnqueuedRejection)
            && presentedIdentity)
        *presentedIdentity = finalPresented;
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LogicalPresentError(res, "logical vkQueuePresentKHR() failed");
    markFixedWorkerOutput();

    if (pendingCompletion) {
        if (!deviceRetirementReactor)
            throw ls::error("fixed virtual event retirement is unavailable");
        int completionFd{-1};
        try { completionFd = this->renderFence->exportSyncFd(vk); }
        catch (...) {
            if (!deviceRetirementReactor->retainEventSourceLost(shared_from_this()))
                throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                    "failed to retain fixed virtual accepted backing");
            throw;
        }
        *pendingCompletion = std::make_unique<ReactorFencePendingOperation>(
            shared_from_this(), deviceRetirementReactor, completionFd,
            std::move(consumedPhysicalAcquires), true, runtimeGpuLifetime);
        return SwapchainPresentResult::logical(res);
    }
    throw ls::error("fixed virtual presentation requires persistent event retirement");
}
