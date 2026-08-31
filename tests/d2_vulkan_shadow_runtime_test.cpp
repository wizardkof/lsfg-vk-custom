#include "d2_vulkan_shadow_runtime.hpp"
#include "shadow_b_return_execution_harness.hpp"

#include <array>
#include <cassert>

using namespace lsfgvk::layer;
using lsfgvk::test::ShadowBReturnExecutionHarness;
using lsfgvk::test::ShadowHarnessFailurePoint;

int main() {
    const auto semaphoreBoth =
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT
        | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    const auto fenceBoth = VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT
        | VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT;
    assert(d2ExternalSyncFdFeaturesEligible(semaphoreBoth, fenceBoth));
    assert(!d2ExternalSyncFdFeaturesEligible(
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT, fenceBoth));
    assert(!d2ExternalSyncFdFeaturesEligible(semaphoreBoth,
        VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT));
    assert(d2ExternalSyncFdFeaturesEligible(semaphoreBoth,
        VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT));

    ShadowBReturnExecutionHarness harness;
    auto vulkan = harness.createQueuePresentVulkan();
    const std::array<VkImage, 2> realImages{
        reinterpret_cast<VkImage>(uintptr_t{0xD201}),
        reinterpret_cast<VkImage>(uintptr_t{0xD202})};
    auto queueMutex = std::make_shared<std::mutex>();
    D2VulkanShadowRuntime runtime(*vulkan, harness.endpoint().queue,
        queueMutex, realImages, VK_FORMAT_R8G8B8A8_UNORM, {1280, 720});
    assert(runtime.shadowCount() == 2);
    assert(runtime.shadowImage(0) != VK_NULL_HANDLE);

    // V6: downstream receives only LSFG synchronization, then the application
    // semaphore receives a temporary SYNC_FD payload.
    D2RealWsiState clean(2);
    const auto appSemaphore = reinterpret_cast<VkSemaphore>(uintptr_t{0xD210});
    uint32_t index{};
    const auto cleanResult = runtime.publicAcquire(clean, appSemaphore,
        VK_NULL_HANDLE, &index,
        [&](VkSemaphore semaphore, VkFence fence, uint32_t* returned) {
            assert(semaphore != VK_NULL_HANDLE && semaphore != appSemaphore);
            assert(fence != VK_NULL_HANDLE);
            *returned = 0;
            return VK_SUBOPTIMAL_KHR;
        });
    assert(cleanResult == VK_SUBOPTIMAL_KHR && index == 0);
    assert(clean.state(0)->appAcquired);
    assert(harness.d2SemaphoreImportCalls() == 1);

    // V4/V5: an ineligible hidden image consumes its acquire semaphore with
    // no image commands, then commits Release only on downstream success.
    D2RealWsiState hiddenRelease(2);
    hiddenRelease.configureCarrierCapability(true, true);
    assert(hiddenRelease.acquired(0));
    const uint32_t releasedIndex = 0;
    assert(hiddenRelease.prepareRelease({&releasedIndex, 1}));
    hiddenRelease.finishRelease({&releasedIndex, 1}, true);
    assert(hiddenRelease.acquired(1));
    hiddenRelease.presented(1, 3);
    uint32_t hiddenAcquireCalls{};
    const auto hiddenReleased = runtime.attemptHiddenAcquire(hiddenRelease,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD214}),
        [&](VkSwapchainKHR, uint64_t timeout, VkSemaphore semaphore,
                VkFence fence, uint32_t* returned) {
            ++hiddenAcquireCalls;
            assert(timeout == 0 && semaphore != VK_NULL_HANDLE
                && fence != VK_NULL_HANDLE);
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t releaseIndex) {
            assert(releaseIndex == 0);
            return VK_SUCCESS;
        });
    assert(hiddenAcquireCalls == 1
        && hiddenReleased.outcome == D2HiddenAcquireOutcome::OwnedIneligible);
    assert(hiddenRelease.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Released);

    D2RealWsiState hiddenReleaseFailure(2);
    hiddenReleaseFailure.configureCarrierCapability(true, true);
    assert(hiddenReleaseFailure.acquired(1));
    hiddenReleaseFailure.presented(1, 3);
    const auto hiddenFailed = runtime.attemptHiddenAcquire(hiddenReleaseFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD215}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_SURFACE_LOST_KHR; });
    assert(hiddenFailed.outcome == D2HiddenAcquireOutcome::OwnedIneligible);
    assert(hiddenReleaseFailure.state(0)->hiddenAcquireOwned
        && hiddenReleaseFailure.state(0)->hiddenOwnership
            == D2HiddenAcquireOwnership::Terminal
        && hiddenReleaseFailure.terminalForLsfgWork());

    // V7/V8: a valid shadow restore precedes the fence temporary import.
    D2RealWsiState restore(2);
    restore.configureCarrierCapability(true, true);
    assert(restore.acquired(1));
    restore.presented(1, vulkan->queueFamilyIndex());
    const auto generation = restore.state(1)->applicationGeneration;
    const auto hiddenSave = runtime.attemptHiddenAcquire(restore,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD211}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 1;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; });
    assert(hiddenSave.outcome == D2HiddenAcquireOutcome::OwnedEligible);
    assert(runtime.submitPreparedShadowSave(restore, 1, generation,
        [](uint32_t) { return VK_ERROR_UNKNOWN; }) == VK_SUCCESS);
    assert(restore.markHiddenClobbered(1, generation));
    assert(restore.modelCarrierReturnedToWsiForTesting(1));
    assert(restore.publicAcquireRelayRequired());
    const auto appFence = reinterpret_cast<VkFence>(uintptr_t{0xD212});
    auto fenceLifecycles = std::make_shared<BorrowedPresentFenceRegistry>(71);
    assert(fenceLifecycles->created(appFence).valid());
    AcquireFenceProxyRegistry proxies(fenceLifecycles);
    const auto restoreResult = runtime.publicAcquire(restore, VK_NULL_HANDLE,
        appFence, &index,
        [appFence](VkSemaphore semaphore, VkFence fence, uint32_t* returned) {
            assert(semaphore != VK_NULL_HANDLE && fence != VK_NULL_HANDLE
                && fence != appFence);
            *returned = 1;
            return VK_SUCCESS;
        }, {}, &proxies);
    assert(restoreResult == VK_SUCCESS && index == 1);
    assert(!restore.state(1)->hiddenClobbered && restore.state(1)->appAcquired);
    assert(!restore.publicAcquireRelayRequired());
    assert(harness.observedImageCopy().calls >= 2);
    assert(harness.d2SemaphoreImportCalls() == 1);
    assert(harness.d2FenceImportCalls() == 0);
    assert(proxies.associationCount() == 1);
    const auto firstProxy = proxies.project(appFence);
    assert(firstProxy != VK_NULL_HANDLE && firstProxy != appFence);
    const VkFence mixed[]{reinterpret_cast<VkFence>(uintptr_t{0xD299}), appFence,
        appFence};
    const auto projected = proxies.project(3, mixed);
    assert(projected.size() == 3 && projected[0] == mixed[0]
        && projected[1] == firstProxy && projected[2] == firstProxy);
    assert(harness.fenceWaitCallsForTesting() == 0);

    // 4B1C post-acquire recovery: after restore submit acceptance, a failed
    // semaphore import waits only the internal completion fence and returns
    // the restored physical image through Maintenance1.
    D2RealWsiState recovery(2);
    recovery.configureCarrierCapability(true, true, vulkan->queueFamilyIndex());
    assert(recovery.acquired(0));
    recovery.presented(0, vulkan->queueFamilyIndex());
    const auto recoveryGeneration = recovery.state(0)->applicationGeneration;
    const auto recoveryHidden = runtime.attemptHiddenAcquire(recovery,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD218}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; });
    assert(recoveryHidden.outcome == D2HiddenAcquireOutcome::OwnedEligible);
    assert(runtime.submitPreparedShadowSave(recovery, 0, recoveryGeneration,
        [](uint32_t) { return VK_ERROR_UNKNOWN; }) == VK_SUCCESS);
    assert(recovery.markHiddenClobbered(0, recoveryGeneration));
    assert(recovery.modelCarrierReturnedToWsiForTesting(0));
    const auto recoveryFence = reinterpret_cast<VkFence>(uintptr_t{0xD219});
    assert(fenceLifecycles->created(recoveryFence).valid());
    const auto waitsBeforeRecovery = harness.fenceWaitCallsForTesting();
    uint32_t recoveryReleases{};
    harness.setFenceWaitResultForTesting(VK_SUCCESS);
    harness.setD2SemaphoreImportResult(VK_ERROR_INVALID_EXTERNAL_HANDLE);
    assert(runtime.publicAcquire(recovery, appSemaphore, recoveryFence, &index,
        [](VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [&](uint32_t released) {
            ++recoveryReleases;
            assert(released == 0);
            return VK_SUCCESS;
        }, &proxies) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
    harness.setD2SemaphoreImportResult(VK_SUCCESS);
    harness.setFenceWaitResultForTesting(VK_TIMEOUT);
    assert(recoveryReleases == 1
        && harness.fenceWaitCallsForTesting() == waitsBeforeRecovery + 1
        && !recovery.state(0)->publicAcquirePending
        && !recovery.state(0)->appAcquired
        && !recovery.state(0)->hiddenClobbered
        && proxies.project(recoveryFence) == recoveryFence);

    // Record failure occurs after downstream physical Acquire but before a
    // queue consumer. The internal Acquire fence is the exact failure-path
    // authority for releasing the untouched image.
    assert(recovery.recordHiddenAcquireSuccess(0)
        == D2HiddenAcquireClassification::EligibleCarrier);
    assert(recovery.markHiddenClobbered(0, recoveryGeneration));
    assert(recovery.modelCarrierReturnedToWsiForTesting(0));
    const auto recordWaitsBefore = harness.fenceWaitCallsForTesting();
    const auto importsBeforeRecordFailure = harness.d2SemaphoreImportCalls();
    uint32_t publicRecordReleases{};
    harness.setFenceWaitResultForTesting(VK_SUCCESS);
    harness.setShadowFailurePoint(ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER);
    assert(runtime.publicAcquire(recovery, appSemaphore, recoveryFence, &index,
        [](VkSemaphore, VkFence fence, uint32_t* returned) {
            assert(fence != VK_NULL_HANDLE);
            *returned = 0;
            return VK_SUCCESS;
        }, [&](uint32_t released) {
            ++publicRecordReleases;
            assert(released == 0);
            return VK_SUCCESS;
        }, &proxies) == VK_ERROR_INITIALIZATION_FAILED);
    harness.setShadowFailurePoint(ShadowHarnessFailurePoint::NONE);
    harness.setFenceWaitResultForTesting(VK_TIMEOUT);
    assert(publicRecordReleases == 1
        && harness.fenceWaitCallsForTesting() == recordWaitsBefore + 1
        && harness.d2SemaphoreImportCalls() == importsBeforeRecordFailure
        && !recovery.state(0)->publicAcquirePending
        && !recovery.state(0)->appAcquired
        && recovery.state(0)->hiddenClobbered);

    // V10: invalid shadow authority fails closed with physical ownership kept.
    D2RealWsiState invalid(1);
    invalid.configureCarrierCapability(true, true);
    assert(invalid.acquired(0));
    invalid.presented(0, vulkan->queueFamilyIndex());
    const auto invalidGeneration = invalid.state(0)->applicationGeneration;
    assert(invalid.recordHiddenAcquireSuccess(0)
        == D2HiddenAcquireClassification::EligibleCarrier);
    assert(runtime.submitShadowSave(invalid, 0, invalidGeneration,
        reinterpret_cast<VkSemaphore>(uintptr_t{0xD213})) == VK_SUCCESS);
    assert(invalid.markHiddenClobbered(0, invalidGeneration));
    invalid.invalidateShadowAuthority(0);
    assert(invalid.modelCarrierReturnedToWsiForTesting(0));
    assert(runtime.publicAcquire(invalid, appSemaphore, VK_NULL_HANDLE, &index,
        [](VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }) == VK_ERROR_DEVICE_LOST);
    assert(invalid.terminalForLsfgWork()
        && invalid.state(0)->publicAcquirePending);

    // V11: both application objects use one fallible semaphore import plus a
    // preallocated logical fence proxy; no fence payload is imported.
    D2RealWsiState partial(1);
    uint32_t bothDownstreamCalls{};
    const auto importsBeforeBoth = harness.d2SemaphoreImportCalls();
    const auto fenceImportsBeforeBoth = harness.d2FenceImportCalls();
    assert(runtime.publicAcquire(partial, appSemaphore, appFence, &index,
        [&](VkSemaphore, VkFence, uint32_t* returned) {
            ++bothDownstreamCalls;
            *returned = 0;
            return VK_SUCCESS;
        }, {}, &proxies) == VK_SUCCESS);
    assert(bothDownstreamCalls == 1
        && harness.d2SemaphoreImportCalls() == importsBeforeBoth + 1
        && harness.d2FenceImportCalls() == fenceImportsBeforeBoth
        && !partial.terminalForLsfgWork()
        && partial.state(0)->appAcquired);
    assert(proxies.project(appFence) != appFence);
    proxies.importSuccess(appFence);
    assert(proxies.project(appFence) == appFence
        && proxies.associationCount() == 0);

    // V12: unsuccessful downstream acquire leaves state and application sync
    // untouched and performs no imports/submission handoff.
    D2RealWsiState unavailable(1);
    const auto semaphoreImports = harness.d2SemaphoreImportCalls();
    const auto fenceImports = harness.d2FenceImportCalls();
    assert(runtime.publicAcquire(unavailable, appSemaphore, VK_NULL_HANDLE, &index,
        [](VkSemaphore, VkFence, uint32_t*) { return VK_TIMEOUT; }) == VK_TIMEOUT);
    assert(!unavailable.state(0)->appAcquired
        && !unavailable.state(0)->publicAcquirePending);
    assert(harness.d2SemaphoreImportCalls() == semaphoreImports
        && harness.d2FenceImportCalls() == fenceImports);
    assert(harness.observedQueuePresentCalls() == 0);
    assert(harness.applicationFenceWaitCalls() == 0);

    // Clean first-consumer OOM/OODM failures do not consume the Acquire
    // semaphore. Wait its paired internal fence, then release untouched Ri.
    for (const auto failure : {VK_ERROR_OUT_OF_HOST_MEMORY,
            VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        D2RealWsiState submitOom(1);
        const auto failingSubmit = harness.observedD2QueueSubmitCalls() + 1;
        harness.setQueueSubmitFailureOnCall(failingSubmit, failure);
        harness.setFenceWaitResultForTesting(VK_SUCCESS);
        const auto waitsBefore = harness.fenceWaitCallsForTesting();
        uint32_t releases{};
        assert(runtime.publicAcquire(submitOom, appSemaphore, VK_NULL_HANDLE,
            &index, [](VkSemaphore, VkFence fence, uint32_t* returned) {
                assert(fence != VK_NULL_HANDLE);
                *returned = 0;
                return VK_SUCCESS;
            }, [&](uint32_t released) {
                ++releases;
                assert(released == 0);
                return VK_SUCCESS;
            }) == failure);
        harness.clearQueueSubmitFailureOnCall();
        harness.setFenceWaitResultForTesting(VK_TIMEOUT);
        assert(releases == 1
            && harness.fenceWaitCallsForTesting() == waitsBefore + 1
            && !submitOom.state(0)->publicAcquirePending
            && !submitOom.state(0)->appAcquired);
    }

    // V13a: if the semaphore-drain submit fails for an untouched ineligible
    // hidden acquire, the physical image is still returned through
    // Maintenance1.  The transaction remains alive so the Acquire semaphore
    // object is not destroyed while its WSI signal may still be pending.
    D2RealWsiState drainFailure(2);
    drainFailure.configureCarrierCapability(true, true,
        vulkan->queueFamilyIndex());
    assert(drainFailure.acquired(0));
    const uint32_t appReleased = 0;
    assert(drainFailure.prepareRelease({&appReleased, 1}));
    drainFailure.finishRelease({&appReleased, 1}, true);
    assert(drainFailure.acquired(1));
    drainFailure.presented(1, vulkan->queueFamilyIndex());
    const auto drainFailureCall = harness.observedD2QueueSubmitCalls() + 1;
    harness.setQueueSubmitFailureOnCall(
        drainFailureCall, VK_ERROR_OUT_OF_HOST_MEMORY);
    const auto drainWaitsBefore = harness.fenceWaitCallsForTesting();
    harness.setFenceWaitResultForTesting(VK_SUCCESS);
    uint32_t drainReleaseCalls{};
    const auto drainFailed = runtime.attemptHiddenAcquire(drainFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD216}),
        [](VkSwapchainKHR, uint64_t timeout, VkSemaphore, VkFence,
                uint32_t* returned) {
            assert(timeout == 0);
            *returned = 0;
            return VK_SUCCESS;
        }, [&](uint32_t releaseIndex) {
            ++drainReleaseCalls;
            assert(releaseIndex == 0);
            return VK_SUCCESS;
        });
    harness.clearQueueSubmitFailureOnCall();
    harness.setFenceWaitResultForTesting(VK_TIMEOUT);
    assert(drainFailed.outcome == D2HiddenAcquireOutcome::OwnedIneligible);
    assert(drainReleaseCalls == 1);
    assert(!drainFailure.state(0)->hiddenAcquireOwned);
    assert(drainFailure.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Released);
    assert(!drainFailure.terminalForLsfgWork()
        && harness.fenceWaitCallsForTesting() == drainWaitsBefore + 1);

    // V13b: command-recording failure happened before any GPU use of Ri.
    // Recovery therefore re-releases the untouched physical image.
    D2RealWsiState recordFailure(1);
    recordFailure.configureCarrierCapability(true, true,
        vulkan->queueFamilyIndex());
    assert(recordFailure.acquired(0));
    recordFailure.presented(0, vulkan->queueFamilyIndex());
    const auto recordGeneration = recordFailure.state(0)->applicationGeneration;
    const auto recordHidden = runtime.attemptHiddenAcquire(recordFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD217}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; });
    assert(recordHidden.outcome == D2HiddenAcquireOutcome::OwnedEligible);
    harness.setShadowFailurePoint(ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER);
    uint32_t recordReleaseCalls{};
    assert(runtime.submitPreparedShadowSave(recordFailure, 0, recordGeneration,
        [&](uint32_t releaseIndex) {
            ++recordReleaseCalls;
            assert(releaseIndex == 0);
            return VK_SUCCESS;
        }) == VK_ERROR_DEVICE_LOST);
    harness.setShadowFailurePoint(ShadowHarnessFailurePoint::NONE);
    assert(recordReleaseCalls == 1);
    assert(!recordFailure.state(0)->hiddenAcquireOwned);
    assert(recordFailure.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Released);
    assert(recordFailure.terminalForLsfgWork());

    // V13c: first QueueSubmit failure likewise means no image command was
    // accepted.  Clean release is still legal and must not be skipped.
    D2RealWsiState firstSubmitFailure(1);
    firstSubmitFailure.configureCarrierCapability(true, true,
        vulkan->queueFamilyIndex());
    assert(firstSubmitFailure.acquired(0));
    firstSubmitFailure.presented(0, vulkan->queueFamilyIndex());
    const auto firstGeneration =
        firstSubmitFailure.state(0)->applicationGeneration;
    assert(runtime.attemptHiddenAcquire(firstSubmitFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD218}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; }).outcome
            == D2HiddenAcquireOutcome::OwnedEligible);
    harness.setQueueSubmitFailureOnCall(
        harness.observedD2QueueSubmitCalls() + 1,
        VK_ERROR_OUT_OF_HOST_MEMORY);
    uint32_t firstFailureReleaseCalls{};
    assert(runtime.submitPreparedShadowSave(firstSubmitFailure, 0,
        firstGeneration, [&](uint32_t releaseIndex) {
            ++firstFailureReleaseCalls;
            assert(releaseIndex == 0);
            return VK_SUCCESS;
        }) == VK_ERROR_DEVICE_LOST);
    harness.clearQueueSubmitFailureOnCall();
    assert(firstFailureReleaseCalls == 1);
    assert(firstSubmitFailure.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Released);
    assert(firstSubmitFailure.terminalForLsfgWork());

    // V13d: a first-submit DEVICE_LOST has no clean failure-to-enqueue
    // guarantee.  Do not guess that Ri is untouched and do not Release it.
    D2RealWsiState indeterminateSubmitFailure(1);
    indeterminateSubmitFailure.configureCarrierCapability(true, true,
        vulkan->queueFamilyIndex());
    assert(indeterminateSubmitFailure.acquired(0));
    indeterminateSubmitFailure.presented(0, vulkan->queueFamilyIndex());
    const auto indeterminateGeneration =
        indeterminateSubmitFailure.state(0)->applicationGeneration;
    assert(runtime.attemptHiddenAcquire(indeterminateSubmitFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD21A}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; }).outcome
            == D2HiddenAcquireOutcome::OwnedEligible);
    harness.setQueueSubmitFailureOnCall(
        harness.observedD2QueueSubmitCalls() + 1, VK_ERROR_DEVICE_LOST);
    uint32_t indeterminateReleaseCalls{};
    assert(runtime.submitPreparedShadowSave(indeterminateSubmitFailure, 0,
        indeterminateGeneration, [&](uint32_t) {
            ++indeterminateReleaseCalls;
            return VK_SUCCESS;
        }) == VK_ERROR_DEVICE_LOST);
    harness.clearQueueSubmitFailureOnCall();
    assert(indeterminateReleaseCalls == 0);
    assert(indeterminateSubmitFailure.state(0)->hiddenAcquireOwned);
    assert(indeterminateSubmitFailure.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::Terminal);
    assert(indeterminateSubmitFailure.terminalForLsfgWork());

    // V13e: if the copy submit was accepted and only its retirement submit
    // fails, Ri has been used by the GPU.  The clean-release route is now
    // forbidden; retain the physical acquisition as terminal authority.
    D2RealWsiState secondSubmitFailure(1);
    secondSubmitFailure.configureCarrierCapability(true, true,
        vulkan->queueFamilyIndex());
    assert(secondSubmitFailure.acquired(0));
    secondSubmitFailure.presented(0, vulkan->queueFamilyIndex());
    const auto secondGeneration =
        secondSubmitFailure.state(0)->applicationGeneration;
    assert(runtime.attemptHiddenAcquire(secondSubmitFailure,
        reinterpret_cast<VkSwapchainKHR>(uintptr_t{0xD219}),
        [](VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }, [](uint32_t) { return VK_ERROR_UNKNOWN; }).outcome
            == D2HiddenAcquireOutcome::OwnedEligible);
    const auto firstCopySubmit = harness.observedD2QueueSubmitCalls() + 1;
    harness.setQueueSubmitFailureOnCall(
        firstCopySubmit + 1, VK_ERROR_OUT_OF_HOST_MEMORY);
    uint32_t forbiddenReleaseCalls{};
    assert(runtime.submitPreparedShadowSave(secondSubmitFailure, 0,
        secondGeneration, [&](uint32_t) {
            ++forbiddenReleaseCalls;
            return VK_SUCCESS;
        }) == VK_ERROR_DEVICE_LOST);
    harness.clearQueueSubmitFailureOnCall();
    assert(forbiddenReleaseCalls == 0);
    assert(secondSubmitFailure.state(0)->hiddenAcquireOwned);
    assert(secondSubmitFailure.state(0)->hiddenOwnership
        == D2HiddenAcquireOwnership::UsedByGpu);
    assert(secondSubmitFailure.terminalForLsfgWork());

    // V13f: normal relay transactions do not accumulate until device destroy.
    // The already-submitted retirement fence is observed non-blockingly on a
    // later API entry and completed transactions are destroyed immediately.
    D2VulkanShadowRuntime reaperRuntime(*vulkan, harness.endpoint().queue,
        queueMutex, realImages, VK_FORMAT_R8G8B8A8_UNORM, {1280, 720});
    D2RealWsiState reaperState(2);
    uint32_t reaperIndex{};
    harness.setD2RetirementFenceStatus(VK_NOT_READY);
    assert(reaperRuntime.publicAcquire(reaperState, appSemaphore,
        VK_NULL_HANDLE, &reaperIndex,
        [](VkSemaphore, VkFence, uint32_t* returned) {
            *returned = 0;
            return VK_SUCCESS;
        }) == VK_SUCCESS);
    assert(reaperRuntime.retainedTransactionCount() == 1);
    harness.setD2RetirementFenceStatus(VK_SUCCESS);
    assert(reaperRuntime.publicAcquire(reaperState, appSemaphore,
        VK_NULL_HANDLE, &reaperIndex,
        [](VkSemaphore, VkFence, uint32_t*) { return VK_TIMEOUT; })
            == VK_TIMEOUT);
    assert(reaperRuntime.retainedTransactionCount() == 0);
    return 0;
}
