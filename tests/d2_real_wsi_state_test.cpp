#include "d2_real_wsi_state.hpp"

#include <array>
#include <cassert>

using namespace lsfgvk::layer;

int main() {
    D2RealWsiState state(3);
    state.retireForLsfgWork();
    assert(state.retiredForLsfgWork());
    // Retirement blocks future LSFG work, not genuine WSI acquire/present or
    // maintenance1 release of application-held physical images.
    assert(state.acquired(2));
    assert(!state.acquired(2));
    const auto acquired = state.state(2);
    assert(acquired && acquired->appAcquired
        && acquired->firstAppAcquireSeen
        && acquired->lastApplicationExit == D2ApplicationExit::None);

    state.presented(2, 7);
    const auto presented = state.state(2);
    assert(presented && !presented->appAcquired
        && presented->lastApplicationExit == D2ApplicationExit::Present
        && presented->lastApplicationPresentFamily == 7
        && !presented->shadowValid && !presented->carrierEligible);

    assert(state.acquired(0) && state.acquired(2));
    const std::array<uint32_t, 2> batch{0, 2};
    assert(state.prepareRelease(batch));
    state.finishRelease(batch, false);
    assert(state.state(0)->appAcquired && state.state(2)->appAcquired);

    assert(state.prepareRelease(batch));
    state.finishRelease(batch, true);
    for (const auto index : batch) {
        const auto released = state.state(index);
        assert(released && !released->appAcquired
            && released->lastApplicationExit == D2ApplicationExit::Release
            && !released->shadowValid && !released->carrierEligible);
    }

    assert(state.acquired(0));
    state.presented(0);
    const auto unknownFamilyPresent = state.state(0);
    assert(unknownFamilyPresent && !unknownFamilyPresent->appAcquired
        && unknownFamilyPresent->lastApplicationExit == D2ApplicationExit::Present
        && !unknownFamilyPresent->lastApplicationPresentFamily.has_value());

    const std::array<uint32_t, 2> duplicate{1, 1};
    assert(state.acquired(1));
    assert(!state.prepareRelease(duplicate));
    assert(state.state(1)->appAcquired);

    // Mixed-batch D2 preparation is allocation-free, generation-bound, and
    // excludes release/reacquire until it is committed or abandoned.
    D2RealWsiState batchPrepared(1);
    const uint32_t batchIndex = 0;
    assert(batchPrepared.acquired(batchIndex));
    const auto batchGeneration = batchPrepared.reserveBatchPresent(batchIndex);
    assert(batchGeneration && !batchPrepared.reserveBatchPresent(batchIndex));
    assert(!batchPrepared.prepareRelease({&batchIndex, 1}));
    assert(!batchPrepared.acquired(batchIndex));
    assert(batchPrepared.abortBatchPresent(batchIndex, *batchGeneration));
    assert(batchPrepared.prepareRelease({&batchIndex, 1}));
    batchPrepared.finishRelease({&batchIndex, 1}, false);
    const auto committedGeneration =
        batchPrepared.reserveBatchPresent(batchIndex);
    assert(committedGeneration == batchGeneration);
    assert(batchPrepared.commitBatchPresent(
        batchIndex, *committedGeneration, 11));
    assert(!batchPrepared.state(batchIndex)->appAcquired
        && !batchPrepared.state(batchIndex)->batchPresentPrepared
        && batchPrepared.state(batchIndex)->applicationGeneration == 1
        && batchPrepared.state(batchIndex)->lastApplicationPresentFamily == 11);

    // B1-B8: carrier eligibility, monotonic application/shadow generations,
    // one save per application generation and restore authority.
    D2RealWsiState carrier(3);
    carrier.configureCarrierCapability(true, true);
    assert(!carrier.carrierEligible(2)); // never application-acquired
    assert(carrier.acquired(0));
    carrier.presented(0, 5);
    assert(carrier.carrierEligible(0));
    const auto generation1 = carrier.state(0)->applicationGeneration;
    assert(generation1 == 1
        && carrier.recordHiddenAcquireSuccess(0)
            == D2HiddenAcquireClassification::EligibleCarrier);
    assert(carrier.shadowSaveRequired(0, generation1));
    assert(carrier.commitShadowSave(0, generation1));
    assert(!carrier.shadowSaveRequired(0, generation1));
    assert(carrier.commitShadowSave(0, generation1)); // no second save
    assert(carrier.markHiddenClobbered(0, generation1));
    assert(carrier.restoreDisposition(0) == D2RestoreDisposition::Required);
    assert(carrier.commitRestore(0));
    assert(carrier.restoreDisposition(0)
        == D2RestoreDisposition::NotClobbered);
    assert(carrier.acquired(0));
    carrier.presented(0, 5);
    const auto generation2 = carrier.state(0);
    assert(generation2 && generation2->applicationGeneration == 2
        && !generation2->shadowValid && !generation2->hiddenClobbered);

    assert(carrier.acquired(1));
    const uint32_t releasedIndex = 1;
    assert(carrier.prepareRelease({&releasedIndex, 1}));
    carrier.finishRelease({&releasedIndex, 1}, true);
    assert(!carrier.carrierEligible(1));

    assert(carrier.acquired(2));
    carrier.presented(2);
    assert(!carrier.carrierEligible(2)); // unknown family

    carrier.retireForLsfgWork();
    assert(!carrier.carrierEligible(0));

    // S1: a new application acquire invalidates carrier eligibility.
    D2RealWsiState reacquire(1);
    reacquire.configureCarrierCapability(true, true);
    assert(reacquire.acquired(0));
    reacquire.presented(0, 5);
    assert(reacquire.carrierEligible(0));
    assert(reacquire.acquired(0));
    assert(!reacquire.carrierEligible(0));

    // S2: a new present generation clears a stale queue family.
    reacquire.presented(0);
    assert(!reacquire.state(0)->lastApplicationPresentFamily.has_value());
    assert(!reacquire.carrierEligible(0));

    // S3: successful hidden acquire owns an ineligible, never-app-acquired
    // image and does not retry for a different carrier.
    D2RealWsiState classification(2);
    classification.configureCarrierCapability(true, true);
    assert(classification.acquired(0));
    classification.presented(0, 5);
    uint32_t hiddenCalls{};
    const auto swapchain = reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x44});
    const auto internalSemaphore =
        reinterpret_cast<VkSemaphore>(uintptr_t{0x45});
    const auto neverAcquired = attemptD2HiddenAcquire(classification,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR observedSwapchain, uint64_t timeout,
                VkSemaphore semaphore, VkFence fence, uint32_t* index) {
            ++hiddenCalls;
            assert(observedSwapchain == swapchain && timeout == 0
                && semaphore == internalSemaphore && fence == VK_NULL_HANDLE);
            *index = 1;
            return VK_SUCCESS;
        });
    assert(hiddenCalls == 1
        && neverAcquired.outcome == D2HiddenAcquireOutcome::OwnedIneligible
        && neverAcquired.internalAcquireSyncRetirementRequired);
    assert(classification.state(1)->hiddenAcquireOwned
        && classification.state(1)->hiddenOwnership
            == D2HiddenAcquireOwnership::AcquiredUntouched);
    assert(!classification.acquired(1));

    D2RealWsiState localConflict(2);
    localConflict.configureCarrierCapability(true, true);
    assert(localConflict.acquired(0));
    assert(localConflict.acquired(1));
    localConflict.presented(1, 6);
    uint32_t localConflictCalls{};
    const auto conflictResult = attemptD2HiddenAcquire(localConflict,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* index) {
            ++localConflictCalls;
            *index = 0;
            return VK_SUCCESS;
        });
    assert(localConflictCalls == 1
        && conflictResult.outcome == D2HiddenAcquireOutcome::OwnedTerminal);
    assert(localConflict.state(0)->hiddenAcquireOwned
        && localConflict.state(0)->hiddenOwnership
            == D2HiddenAcquireOwnership::Terminal
        && localConflict.terminalForLsfgWork());
    const auto conflictRetry = attemptD2HiddenAcquire(localConflict,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++localConflictCalls;
            return VK_NOT_READY;
        });
    assert(localConflictCalls == 1
        && conflictRetry.outcome == D2HiddenAcquireOutcome::NoImage);

    // S4-S6: clean release is two-phase and preserves application history;
    // downstream failure never falsely commits Released.
    D2RealWsiState abortable(1);
    abortable.configureCarrierCapability(true, true);
    assert(abortable.acquired(0));
    const uint32_t appReleasedIndex = 0;
    assert(abortable.prepareRelease({&appReleasedIndex, 1}));
    abortable.finishRelease({&appReleasedIndex, 1}, true);
    assert(abortable.recordHiddenAcquireSuccess(0)
        == D2HiddenAcquireClassification::IneligibleCarrier);
    assert(abortable.prepareHiddenRelease(0));
    assert(abortable.state(0)->hiddenAcquireOwned);
    abortable.finishHiddenRelease(0, VK_ERROR_SURFACE_LOST_KHR);
    assert(abortable.state(0)->hiddenAcquireOwned
        && abortable.state(0)->hiddenOwnership
            == D2HiddenAcquireOwnership::AcquiredUntouched);
    assert(abortable.prepareHiddenRelease(0));
    abortable.finishHiddenRelease(0, VK_SUCCESS);
    assert(abortable.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Released);
    assert(abortable.state(0)->lastApplicationExit
        == D2ApplicationExit::Release);

    D2RealWsiState history(1);
    history.configureCarrierCapability(true, true);
    assert(history.acquired(0));
    history.presented(0, 9);
    const auto historyGeneration = history.state(0)->applicationGeneration;
    assert(history.recordHiddenAcquireSuccess(0)
        == D2HiddenAcquireClassification::EligibleCarrier);
    assert(history.prepareHiddenRelease(0));
    history.finishHiddenRelease(0, VK_SUCCESS);
    assert(history.state(0)->lastApplicationExit == D2ApplicationExit::Present
        && history.state(0)->applicationGeneration == historyGeneration
        && history.state(0)->lastApplicationPresentFamily == 9
        && history.carrierEligible(0));

    // S7: retirement before the call suppresses downstream acquire. A success
    // already returned during the race is still recorded as owned/ineligible.
    D2RealWsiState retiredBeforeCall(1);
    retiredBeforeCall.configureCarrierCapability(true, true);
    retiredBeforeCall.retireForLsfgWork();
    uint32_t retiredCalls{};
    const auto retiredAttempt = attemptD2HiddenAcquire(retiredBeforeCall,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++retiredCalls;
            return VK_SUCCESS;
        });
    assert(retiredCalls == 0
        && retiredAttempt.outcome == D2HiddenAcquireOutcome::NoImage);
    D2RealWsiState retiredRace(1);
    retiredRace.configureCarrierCapability(true, true);
    assert(retiredRace.acquired(0));
    retiredRace.presented(0, 4);
    const auto raced = attemptD2HiddenAcquire(retiredRace,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* index) {
            *index = 0;
            retiredRace.retireForLsfgWork();
            return VK_SUCCESS;
        });
    assert(raced.outcome == D2HiddenAcquireOutcome::OwnedIneligible
        && retiredRace.state(0)->hiddenAcquireOwned);

    // Successful out-of-range acquire retains terminal authority in state.
    D2RealWsiState invalidIndex(1);
    invalidIndex.configureCarrierCapability(true, true);
    assert(invalidIndex.acquired(0));
    invalidIndex.presented(0, 7);
    uint32_t invalidIndexCalls{};
    const auto invalid = attemptD2HiddenAcquire(invalidIndex,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* index) {
            ++invalidIndexCalls;
            *index = 999;
            return VK_SUCCESS;
        });
    assert(invalid.outcome == D2HiddenAcquireOutcome::OwnedTerminal
        && invalid.internalAcquireSyncRetirementRequired
        && invalidIndex.terminalHiddenAcquireAuthority() == 999
        && invalidIndex.terminalForLsfgWork());
    const auto invalidRetry = attemptD2HiddenAcquire(invalidIndex,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++invalidIndexCalls;
            return VK_NOT_READY;
        });
    assert(invalidIndexCalls == 1
        && invalidRetry.outcome == D2HiddenAcquireOutcome::NoImage);

    // S8: clobbered contents with invalidated shadow authority fail closed.
    D2RealWsiState invalidShadow(1);
    invalidShadow.configureCarrierCapability(true, true);
    assert(invalidShadow.acquired(0));
    invalidShadow.presented(0, 2);
    const auto shadowGeneration = invalidShadow.state(0)->applicationGeneration;
    assert(invalidShadow.recordHiddenAcquireSuccess(0)
        == D2HiddenAcquireClassification::EligibleCarrier);
    assert(invalidShadow.commitShadowSave(0, shadowGeneration));
    assert(invalidShadow.markHiddenClobbered(0, shadowGeneration));
    invalidShadow.invalidateShadowAuthority(0);
    assert(invalidShadow.restoreDisposition(0)
        == D2RestoreDisposition::InvalidShadowAuthority);

    // S9A: NOT_READY gets one timeout-zero attempt when a real candidate exists.
    D2RealWsiState notReadyState(1);
    notReadyState.configureCarrierCapability(true, true);
    assert(notReadyState.acquired(0));
    notReadyState.presented(0, 8);
    hiddenCalls = 0;
    const auto notReady = attemptD2HiddenAcquire(notReadyState,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR observedSwapchain, uint64_t timeout,
                VkSemaphore semaphore, VkFence fence, uint32_t*) {
            ++hiddenCalls;
            assert(observedSwapchain == swapchain && timeout == 0
                && semaphore == internalSemaphore && fence == VK_NULL_HANDLE);
            return VK_NOT_READY;
        });
    assert(hiddenCalls == 1
        && notReady.outcome == D2HiddenAcquireOutcome::NoImage
        && !notReady.internalAcquireSyncRetirementRequired);

    // S9B: without a real candidate there is no downstream acquire.
    uint32_t noCandidateCalls{};
    const auto noCandidate = attemptD2HiddenAcquire(abortable,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++noCandidateCalls;
            return VK_NOT_READY;
        });
    assert(noCandidateCalls == 0
        && noCandidate.outcome == D2HiddenAcquireOutcome::NoImage
        && !noCandidate.internalAcquireSyncRetirementRequired);

    D2RealWsiState otherNoCandidates(2);
    otherNoCandidates.configureCarrierCapability(true, true);
    uint32_t otherNoCandidateCalls{};
    const auto countUnexpectedAcquire =
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++otherNoCandidateCalls;
            return VK_NOT_READY;
        };
    assert(attemptD2HiddenAcquire(otherNoCandidates, swapchain,
            internalSemaphore, countUnexpectedAcquire).outcome
        == D2HiddenAcquireOutcome::NoImage); // all never acquired
    assert(otherNoCandidates.acquired(0));
    assert(attemptD2HiddenAcquire(otherNoCandidates, swapchain,
            internalSemaphore, countUnexpectedAcquire).outcome
        == D2HiddenAcquireOutcome::NoImage); // only APP_ACQUIRED
    assert(otherNoCandidates.acquired(1));
    otherNoCandidates.presented(1); // unknown family
    assert(attemptD2HiddenAcquire(otherNoCandidates, swapchain,
            internalSemaphore, countUnexpectedAcquire).outcome
        == D2HiddenAcquireOutcome::NoImage);
    assert(otherNoCandidateCalls == 0);

    // S10: candidate history alone cannot bypass either capability gate.
    D2RealWsiState disabledCapability(1);
    assert(disabledCapability.acquired(0));
    disabledCapability.presented(0, 10);
    uint32_t disabledCalls{};
    disabledCapability.configureCarrierCapability(false, true);
    const auto noCopy = attemptD2HiddenAcquire(disabledCapability,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++disabledCalls;
            return VK_NOT_READY;
        });
    disabledCapability.configureCarrierCapability(true, false);
    const auto noMaintenance = attemptD2HiddenAcquire(disabledCapability,
        swapchain, internalSemaphore,
        [&](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) {
            ++disabledCalls;
            return VK_NOT_READY;
        });
    assert(disabledCalls == 0
        && noCopy.outcome == D2HiddenAcquireOutcome::NoImage
        && noMaintenance.outcome == D2HiddenAcquireOutcome::NoImage);
    return 0;
}
