/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "aborted_present_semantics.hpp"
#include "swapchain_release_backend.hpp"

#include <cstdint>
#include <functional>
#include <memory>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

class Adaptive1xRecoveryAuthority final {
public:
    enum class State : uint8_t {
        Reserved, AcquireCalled, AcquireFailed, AcquireAccepted,
        ProducerPrepared, ProducerSubmitFailedClean, ProducerSubmitAccepted,
        ReadyForLogicalPresent, LogicalPresentCalled, PresentNormal,
        PresentEnqueuedRejection, PresentPreEnqueueFailure,
        AcquireReleasePending, ProducerReleasePending, Released,
        PresentRetirementPending, Retired, DeviceLost, Indeterminate
    };
    enum class TerminalAction : uint8_t {
        None, ReleaseWithoutPresent, PresentRetirement, ConservativeRetention
    };
    struct Operations final {
        std::function<VkResult()> waitAcquireCompletion;
        std::function<VkResult()> waitProducerCompletion;
        std::function<VkResult(uint32_t)> releasePhysicalImage;
        std::function<bool(std::shared_ptr<void>)> retainTerminal;
    };

    Adaptive1xRecoveryAuthority(uint64_t deviceIdentity,
        uint64_t swapchainLifecycle, uint64_t operationGeneration,
        VkSwapchainKHR physicalSwapchain, SwapchainReleaseBackend releaseBackend,
        VkSemaphore acquireSemaphore, VkFence acquireFence,
        VkFence producerFence, std::shared_ptr<void> strongBacking,
        Operations operations) noexcept;
    Adaptive1xRecoveryAuthority(const Adaptive1xRecoveryAuthority&) = delete;
    Adaptive1xRecoveryAuthority& operator=(const Adaptive1xRecoveryAuthority&) = delete;
    Adaptive1xRecoveryAuthority(Adaptive1xRecoveryAuthority&&) noexcept = default;
    Adaptive1xRecoveryAuthority& operator=(Adaptive1xRecoveryAuthority&&) noexcept = default;

    [[nodiscard]] bool eligible() const noexcept;
    [[nodiscard]] State state() const noexcept { return stateValue; }
    [[nodiscard]] TerminalAction terminalAction() const noexcept { return action; }
    [[nodiscard]] uint64_t generation() const noexcept { return operationGeneration; }
    [[nodiscard]] bool matchesGeneration(uint64_t value) const noexcept {
        return value != 0 && value == operationGeneration;
    }
    [[nodiscard]] VkSemaphore acquireSemaphore() const noexcept { return acquireSemaphoreValue; }
    [[nodiscard]] VkFence acquireFence() const noexcept { return acquireFenceValue; }
    [[nodiscard]] VkFence producerFence() const noexcept { return producerFenceValue; }
    [[nodiscard]] uint32_t physicalImageIndex() const noexcept { return imageIndex; }

    [[nodiscard]] bool markAcquireCalled() noexcept;
    [[nodiscard]] bool bindAcquireResult(VkResult, uint32_t imageIndex) noexcept;
    [[nodiscard]] bool markProducerPrepared() noexcept;
    [[nodiscard]] bool bindProducerSubmitResult(VkResult) noexcept;
    [[nodiscard]] bool markReadyForLogicalPresent() noexcept;
    [[nodiscard]] bool markLogicalPresentCalled() noexcept;
    [[nodiscard]] bool finalizeLogicalPresent(VkResult) noexcept;
    [[nodiscard]] VkResult recoverAcquireFailureOnly() noexcept;
    [[nodiscard]] VkResult recoverProducerFailureOnly() noexcept;
    [[nodiscard]] bool markPresentRetired() noexcept;

private:
    [[nodiscard]] bool retainConservatively(State) noexcept;
    [[nodiscard]] VkResult releaseAfter(const std::function<VkResult()>&,
        State pendingState) noexcept;

    uint64_t deviceIdentityValue{};
    uint64_t swapchainLifecycleValue{};
    uint64_t operationGeneration{};
    VkSwapchainKHR physicalSwapchainValue{};
    SwapchainReleaseBackend releaseBackendValue{};
    VkSemaphore acquireSemaphoreValue{};
    VkFence acquireFenceValue{};
    VkFence producerFenceValue{};
    uint32_t imageIndex{};
    std::shared_ptr<void> backing;
    Operations ops;
    State stateValue{State::Reserved};
    TerminalAction action{TerminalAction::None};
};

} // namespace lsfgvk::layer
