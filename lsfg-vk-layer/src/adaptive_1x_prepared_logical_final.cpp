/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "adaptive_1x_prepared_logical_final.hpp"

#include <unistd.h>

namespace lsfgvk::layer {

Adaptive1xPreparedLogicalFinal::Adaptive1xPreparedLogicalFinal(
        VkSwapchainKHR swapchain, uint32_t index, VkSemaphore ready,
        VkFence application, VkFence internal,
        PresentedPhysicalImageIdentity identity,
        Adaptive1xRecoveryAuthority recoveryAuthority,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leaseRegistry,
        PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease lease,
        std::shared_ptr<BorrowedPresentFenceRegistry> borrowedRegistry,
        std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
            borrowedFence,
        std::vector<PresentedPhysicalImageAcquireToken> reacquires,
        std::shared_ptr<void> strongBacking, Operations operations) noexcept
    : physicalSwapchainValue(swapchain), imageIndex(index), readySemaphore(ready),
      applicationFence(application), internalFence(internal), presented(identity),
      recovery(std::move(recoveryAuthority)), leases(std::move(leaseRegistry)),
      preparedLease(lease), borrowedFences(std::move(borrowedRegistry)),
      preparedBorrowedFence(std::move(borrowedFence)),
      consumedAcquires(std::move(reacquires)), backing(std::move(strongBacking)),
      ops(std::move(operations)) {}

bool Adaptive1xPreparedLogicalFinal::valid() const noexcept {
    return stateValue == State::Ready && physicalSwapchainValue
        && readySemaphore && presented.valid() && leases && preparedLease.valid()
        && backing && recovery.state()
            == Adaptive1xRecoveryAuthority::State::ReadyForLogicalPresent;
}

bool Adaptive1xPreparedLogicalFinal::markLogicalPresentCalled() noexcept {
    if (!valid() || !recovery.markLogicalPresentCalled()) return false;
    stateValue = State::LogicalPresentCalled;
    return true;
}

void Adaptive1xPreparedLogicalFinal::abortReservations() noexcept {
    if (preparedBorrowedFence && borrowedFences)
        borrowedFences->abort(*preparedBorrowedFence);
    if (preparedLease.valid() && leases) leases->abort(preparedLease);
}

bool Adaptive1xPreparedLogicalFinal::consumeReacquires() noexcept {
    for (const auto& token : consumedAcquires)
        if (!leases->markAcquireConsumed(token)) return false;
    return true;
}

bool Adaptive1xPreparedLogicalFinal::retireReacquires() noexcept {
    for (const auto& token : consumedAcquires)
        if (!leases->retireAfterSafeCompletion(token)) return false;
    consumedAcquires.clear();
    return true;
}

VkResult Adaptive1xPreparedLogicalFinal::retainBacking(VkResult result) noexcept {
    if (!ops.retainConservatively || !ops.retainConservatively(backing))
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    backing.reset();
    stateValue = State::Retained;
    return result;
}

VkResult Adaptive1xPreparedLogicalFinal::finalizeLogicalPresent(
        VkResult result) noexcept {
    if (stateValue != State::LogicalPresentCalled
            || !recovery.finalizeLogicalPresent(result))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!consumeReacquires()) return retainBacking(VK_ERROR_DEVICE_LOST);

    const auto resultClass = classifyPresentResult(result);
    if (resultClass == PresentResultClass::PreEnqueueFailure) {
        if (preparedBorrowedFence && borrowedFences)
            borrowedFences->abort(*preparedBorrowedFence);
        const auto recovered = recovery.recoverProducerFailureOnly();
        if (recovered != VK_SUCCESS) return retainBacking(recovered);
        if (preparedLease.valid()) leases->abort(preparedLease);
        if (!retireReacquires()) return retainBacking(VK_ERROR_DEVICE_LOST);
        backing.reset();
        stateValue = State::Finalized;
        return VK_SUCCESS;
    }

    if (!leases->commit(preparedLease))
        return retainBacking(VK_ERROR_DEVICE_LOST);
    const bool enqueued = resultClass == PresentResultClass::Normal
        || resultClass == PresentResultClass::EnqueuedRejection;
    if (preparedBorrowedFence && borrowedFences) {
        if (enqueued) {
            static_cast<void>(borrowedFences->commit(*preparedBorrowedFence));
            if (preparedBorrowedFence->valid())
                return retainBacking(VK_ERROR_DEVICE_LOST);
        } else {
            borrowedFences->abort(*preparedBorrowedFence);
        }
    }
    if (!enqueued) {
        stateValue = State::Retained;
        backing.reset();
        return VK_SUCCESS;
    }
    if (requiresInternalPresentFence()
            && (!ops.armInternalPresentFence
                || !ops.armInternalPresentFence(presented)))
        return retainBacking(VK_ERROR_DEVICE_LOST);

    int completionFd{-1};
    const auto exported = ops.exportProducerCompletion
        ? ops.exportProducerCompletion(&completionFd)
        : VK_ERROR_INITIALIZATION_FAILED;
    if (exported != VK_SUCCESS) return retainBacking(exported);
    const auto published = ops.publishCompletion
        ? ops.publishCompletion(completionFd, std::move(consumedAcquires))
        : VK_ERROR_INITIALIZATION_FAILED;
    if (published != VK_SUCCESS) {
        if (completionFd >= 0) static_cast<void>(::close(completionFd));
        return retainBacking(published);
    }
    backing.reset();
    stateValue = State::Finalized;
    return VK_SUCCESS;
}

VkResult Adaptive1xPreparedLogicalFinal::abandonWithoutLogicalPresent() noexcept {
    if (!valid()) return VK_ERROR_INITIALIZATION_FAILED;
    if (!consumeReacquires()) return retainBacking(VK_ERROR_DEVICE_LOST);
    const auto recovered = recovery.recoverProducerFailureOnly();
    if (recovered != VK_SUCCESS) return retainBacking(recovered);
    abortReservations();
    if (!retireReacquires()) return retainBacking(VK_ERROR_DEVICE_LOST);
    backing.reset();
    stateValue = State::Abandoned;
    return VK_SUCCESS;
}

} // namespace lsfgvk::layer
