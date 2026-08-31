/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "adaptive_1x_recovery_authority.hpp"

namespace lsfgvk::layer {

Adaptive1xRecoveryAuthority::Adaptive1xRecoveryAuthority(
        uint64_t deviceIdentity, uint64_t swapchainLifecycle,
        uint64_t generation, VkSwapchainKHR physicalSwapchain,
        SwapchainReleaseBackend releaseBackend, VkSemaphore acquireSemaphore,
        VkFence acquireFence, VkFence producerFence,
        std::shared_ptr<void> strongBacking, Operations operations) noexcept
    : deviceIdentityValue(deviceIdentity),
      swapchainLifecycleValue(swapchainLifecycle),
      operationGeneration(generation),
      physicalSwapchainValue(physicalSwapchain),
      releaseBackendValue(releaseBackend),
      acquireSemaphoreValue(acquireSemaphore), acquireFenceValue(acquireFence),
      producerFenceValue(producerFence), backing(std::move(strongBacking)),
      ops(std::move(operations)) {}

bool Adaptive1xRecoveryAuthority::eligible() const noexcept {
    return deviceIdentityValue && swapchainLifecycleValue && operationGeneration
        && physicalSwapchainValue && acquireSemaphoreValue && acquireFenceValue
        && producerFenceValue && backing
        && releaseBackendValue != SwapchainReleaseBackend::None
        && ops.waitAcquireCompletion && ops.waitProducerCompletion
        && ops.releasePhysicalImage && ops.retainTerminal;
}

bool Adaptive1xRecoveryAuthority::markAcquireCalled() noexcept {
    if (!eligible() || stateValue != State::Reserved) return false;
    stateValue = State::AcquireCalled;
    return true;
}

bool Adaptive1xRecoveryAuthority::bindAcquireResult(
        VkResult result, uint32_t value) noexcept {
    if (stateValue != State::AcquireCalled) return false;
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
        imageIndex = value;
        stateValue = State::AcquireAccepted;
    } else {
        stateValue = result == VK_ERROR_DEVICE_LOST
            ? State::DeviceLost : State::AcquireFailed;
    }
    return true;
}

bool Adaptive1xRecoveryAuthority::markProducerPrepared() noexcept {
    if (stateValue != State::AcquireAccepted) return false;
    stateValue = State::ProducerPrepared;
    return true;
}

bool Adaptive1xRecoveryAuthority::bindProducerSubmitResult(VkResult result) noexcept {
    if (stateValue != State::ProducerPrepared) return false;
    if (result == VK_SUCCESS) stateValue = State::ProducerSubmitAccepted;
    else if (result == VK_ERROR_OUT_OF_HOST_MEMORY
            || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        stateValue = State::ProducerSubmitFailedClean;
    else if (result == VK_ERROR_DEVICE_LOST)
        return retainConservatively(State::DeviceLost);
    else
        return retainConservatively(State::Indeterminate);
    return true;
}

bool Adaptive1xRecoveryAuthority::markReadyForLogicalPresent() noexcept {
    if (stateValue != State::ProducerSubmitAccepted) return false;
    stateValue = State::ReadyForLogicalPresent;
    return true;
}

bool Adaptive1xRecoveryAuthority::markLogicalPresentCalled() noexcept {
    if (stateValue != State::ReadyForLogicalPresent) return false;
    stateValue = State::LogicalPresentCalled;
    return true;
}

bool Adaptive1xRecoveryAuthority::finalizeLogicalPresent(VkResult result) noexcept {
    if (stateValue != State::LogicalPresentCalled) return false;
    switch (classifyPresentResult(result)) {
    case PresentResultClass::Normal:
        stateValue = State::PresentNormal;
        action = TerminalAction::PresentRetirement;
        stateValue = State::PresentRetirementPending;
        return true;
    case PresentResultClass::EnqueuedRejection:
        stateValue = State::PresentEnqueuedRejection;
        action = TerminalAction::PresentRetirement;
        stateValue = State::PresentRetirementPending;
        return true;
    case PresentResultClass::PreEnqueueFailure:
        stateValue = State::PresentPreEnqueueFailure;
        return true;
    case PresentResultClass::DeviceLost:
        return retainConservatively(State::DeviceLost);
    case PresentResultClass::Indeterminate:
        return retainConservatively(State::Indeterminate);
    }
    return false;
}

bool Adaptive1xRecoveryAuthority::retainConservatively(State state) noexcept {
    if (!ops.retainTerminal || !ops.retainTerminal(backing)) return false;
    action = TerminalAction::ConservativeRetention;
    stateValue = state;
    backing.reset();
    return true;
}

VkResult Adaptive1xRecoveryAuthority::releaseAfter(
        const std::function<VkResult()>& wait, State pendingState) noexcept {
    if (!wait || !ops.releasePhysicalImage) return VK_ERROR_INITIALIZATION_FAILED;
    stateValue = pendingState;
    const auto completion = wait();
    if (completion == VK_ERROR_DEVICE_LOST) {
        if (!retainConservatively(State::DeviceLost)) return VK_ERROR_DEVICE_LOST;
        return VK_ERROR_DEVICE_LOST;
    }
    if (completion != VK_SUCCESS) {
        if (!retainConservatively(State::Indeterminate)) return completion;
        return completion;
    }
    const auto released = ops.releasePhysicalImage(imageIndex);
    if (released != VK_SUCCESS) {
        if (!retainConservatively(released == VK_ERROR_DEVICE_LOST
                ? State::DeviceLost : State::Indeterminate))
            return released;
        return released;
    }
    action = TerminalAction::ReleaseWithoutPresent;
    stateValue = State::Released;
    backing.reset();
    return VK_SUCCESS;
}

VkResult Adaptive1xRecoveryAuthority::recoverAcquireFailureOnly() noexcept {
    if (stateValue != State::AcquireAccepted
            && stateValue != State::ProducerPrepared
            && stateValue != State::ProducerSubmitFailedClean)
        return VK_ERROR_INITIALIZATION_FAILED;
    return releaseAfter(ops.waitAcquireCompletion, State::AcquireReleasePending);
}

VkResult Adaptive1xRecoveryAuthority::recoverProducerFailureOnly() noexcept {
    if (stateValue != State::ProducerSubmitAccepted
            && stateValue != State::ReadyForLogicalPresent
            && stateValue != State::PresentPreEnqueueFailure)
        return VK_ERROR_INITIALIZATION_FAILED;
    return releaseAfter(ops.waitProducerCompletion, State::ProducerReleasePending);
}

bool Adaptive1xRecoveryAuthority::markPresentRetired() noexcept {
    if (stateValue != State::PresentRetirementPending
            || action != TerminalAction::PresentRetirement)
        return false;
    stateValue = State::Retired;
    backing.reset();
    return true;
}

} // namespace lsfgvk::layer
