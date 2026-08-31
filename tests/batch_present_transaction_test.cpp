#include "batch_present_transaction.hpp"

#include <cassert>
#include <cstdint>
#include <type_traits>

using namespace lsfgvk::layer;

template <typename T> T handle(uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}

int main() {
    VkSwapchainKHR swapchains[]{handle<VkSwapchainKHR>(11), handle<VkSwapchainKHR>(12)};
    uint32_t indices[]{3, 4};
    VkSemaphore waits[]{handle<VkSemaphore>(21), handle<VkSemaphore>(22)};
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.pNext = reinterpret_cast<void*>(0x1);
    info.waitSemaphoreCount = 2;
    info.pWaitSemaphores = waits;
    info.swapchainCount = 2;
    info.pSwapchains = swapchains;
    info.pImageIndices = indices;

    auto transaction = BatchPresentTransaction::create(
        handle<VkDevice>(1), handle<VkQueue>(2), 7, info);
    assert(transaction);
    assert(transaction->state() == BatchPresentTransaction::State::StorageReserved);
    assert(transaction->applicationHadPNext());
    assert(transaction->applicationWaits().size() == 2);
    assert(transaction->entries()[1].applicationImageIndex == 4);

    // Caller arrays are snapshotted before any accepted queue side effect.
    indices[1] = 99;
    waits[0] = VK_NULL_HANDLE;
    assert(transaction->entries()[1].applicationImageIndex == 4);
    assert(transaction->applicationWaits()[0] == handle<VkSemaphore>(21));
    assert(!transaction->hasQueueSideEffect());

    for (const auto failure : {
            BatchPresentTransaction::CreateFailureInjection::WaitStorage,
            BatchPresentTransaction::CreateFailureInjection::EntryStorage,
            BatchPresentTransaction::CreateFailureInjection::FinalStorage}) {
        auto failed = BatchPresentTransaction::create(
            handle<VkDevice>(1), handle<VkQueue>(2), 8, info, failure);
        assert(!failed);
    }
    assert(transaction->finalSwapchains().size() == 2);
    assert(transaction->finalImageIndices().size() == 2);
    assert(transaction->finalWaits().size() == 2);

    transaction->entry(0).path = BatchPresentTransaction::Path::Native;
    auto& prepared = transaction->entry(1);
    prepared.path = BatchPresentTransaction::Path::VirtualPrepared;
    prepared.physicalSwapchain = handle<VkSwapchainKHR>(31);
    prepared.physicalImageIndex = 5;
    prepared.completionSemaphore = handle<VkSemaphore>(32);
    prepared.deviceLifetimeIdentity = 41;
    prepared.lifecycleGeneration = 9;
    prepared.presentOperationIdentity = 51;
    prepared.preparationAccepted = true;

    assert(transaction->markWaitBridgePrepared());
    assert(!transaction->applicationWaitsConsumed());
    assert(transaction->markWaitBridgeAccepted());
    assert(transaction->applicationWaitsConsumed());
    assert(transaction->markEntriesPreparing());
    assert(transaction->markEntriesReady());
    const auto swapchainCapacity = transaction->finalSwapchains().capacity();
    const auto indexCapacity = transaction->finalImageIndices().capacity();
    const auto waitCapacity = transaction->finalWaits().capacity();
    assert(transaction->setFinalEntry(0, swapchains[0], 3));
    assert(transaction->setFinalEntry(1, prepared.physicalSwapchain,
        prepared.physicalImageIndex));
    assert(transaction->setFinalWait(0, prepared.completionSemaphore));
    VkResult results[2]{};
    const auto finalInfo = transaction->prepareFinalPresentInfo(info.pNext, results, 1);
    assert(finalInfo && finalInfo->swapchainCount == 2);
    assert(finalInfo->pNext == info.pNext && finalInfo->pResults == results);
    assert(finalInfo->pSwapchains[1] == prepared.physicalSwapchain);
    assert(transaction->finalSwapchains().capacity() == swapchainCapacity);
    assert(transaction->finalImageIndices().capacity() == indexCapacity);
    assert(transaction->finalWaits().capacity() == waitCapacity);

    // Dense wait storage is independent of logical entry indices.
    auto mixed = BatchPresentTransaction::create(
        handle<VkDevice>(1), handle<VkQueue>(2), 75, info);
    assert(mixed);
    mixed->entry(0).path = BatchPresentTransaction::Path::Native;
    mixed->entry(1) = prepared;
    assert(mixed->markWaitBridgePrepared());
    assert(mixed->markWaitBridgeAccepted());
    assert(mixed->markEntriesPreparing());
    assert(mixed->markEntriesReady());
    assert(mixed->setFinalEntry(0, swapchains[0], 3));
    assert(mixed->setFinalEntry(1, prepared.physicalSwapchain, 5));
    assert(mixed->appendFinalWait(prepared.completionSemaphore));
    const auto mixedInfo = mixed->prepareFinalPresentInfo(nullptr, nullptr, 1);
    assert(mixedInfo && mixedInfo->pWaitSemaphores[0]
        == prepared.completionSemaphore);
    assert(transaction->markFinalPresentCalled());
    assert(transaction->markFinalPresentResult(VK_SUBOPTIMAL_KHR));
    assert(transaction->state() == BatchPresentTransaction::State::FinalPresentNormal);
    assert(transaction->finalResultClass() == PresentResultClass::Normal);
    assert(transaction->finalResult() == VK_SUBOPTIMAL_KHR);
    assert(transaction->markRetirementPending());
    assert(transaction->markRetired());

    const auto classifyThroughTransaction = [&](VkResult value,
            PresentResultClass expected, BatchPresentTransaction::State expectedState) {
        auto item = BatchPresentTransaction::create(
            handle<VkDevice>(1), handle<VkQueue>(2), 90 + static_cast<uint64_t>(-value), info);
        assert(item);
        item->entry(0).path = BatchPresentTransaction::Path::Native;
        item->entry(1).path = BatchPresentTransaction::Path::Native;
        assert(item->markWaitBridgePrepared());
        assert(item->markWaitBridgeAccepted());
        assert(item->markEntriesPreparing());
        assert(item->markEntriesReady());
        assert(item->setFinalEntry(0, swapchains[0], 3));
        assert(item->setFinalEntry(1, swapchains[1], 4));
        assert(item->prepareFinalPresentInfo(nullptr, nullptr, 0));
        assert(item->markFinalPresentCalled());
        assert(item->markFinalPresentResult(value));
        assert(item->finalResultClass() == expected);
        assert(item->state() == expectedState);
    };
    classifyThroughTransaction(VK_SUCCESS, PresentResultClass::Normal,
        BatchPresentTransaction::State::FinalPresentNormal);
    classifyThroughTransaction(VK_ERROR_OUT_OF_DATE_KHR,
        PresentResultClass::EnqueuedRejection,
        BatchPresentTransaction::State::FinalPresentEnqueuedRejection);
    classifyThroughTransaction(VK_ERROR_SURFACE_LOST_KHR,
        PresentResultClass::EnqueuedRejection,
        BatchPresentTransaction::State::FinalPresentEnqueuedRejection);
    classifyThroughTransaction(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
        PresentResultClass::EnqueuedRejection,
        BatchPresentTransaction::State::FinalPresentEnqueuedRejection);
    classifyThroughTransaction(VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT,
        PresentResultClass::EnqueuedRejection,
        BatchPresentTransaction::State::FinalPresentEnqueuedRejection);
    classifyThroughTransaction(VK_ERROR_OUT_OF_HOST_MEMORY,
        PresentResultClass::PreEnqueueFailure,
        BatchPresentTransaction::State::FinalPresentPreEnqueueFailure);
    classifyThroughTransaction(VK_ERROR_OUT_OF_DEVICE_MEMORY,
        PresentResultClass::PreEnqueueFailure,
        BatchPresentTransaction::State::FinalPresentPreEnqueueFailure);
    classifyThroughTransaction(VK_ERROR_DEVICE_LOST, PresentResultClass::DeviceLost,
        BatchPresentTransaction::State::DeviceLost);
    classifyThroughTransaction(VK_ERROR_VALIDATION_FAILED_EXT,
        PresentResultClass::Indeterminate,
        BatchPresentTransaction::State::FinalPresentIndeterminate);

    // PRE-COMMIT metadata is sufficient; a present-operation identity is
    // mandatory and the transaction owns no lease/retirement authority.
    auto invalidReady = BatchPresentTransaction::create(
        handle<VkDevice>(1), handle<VkQueue>(2), 200, info);
    assert(invalidReady);
    invalidReady->entry(0).path = BatchPresentTransaction::Path::Native;
    auto& invalidPrepared = invalidReady->entry(1);
    invalidPrepared.path = BatchPresentTransaction::Path::VirtualPrepared;
    invalidPrepared.physicalSwapchain = handle<VkSwapchainKHR>(31);
    invalidPrepared.physicalImageIndex = 5;
    invalidPrepared.completionSemaphore = handle<VkSemaphore>(32);
    invalidPrepared.deviceLifetimeIdentity = 41;
    invalidPrepared.lifecycleGeneration = 10;
    invalidPrepared.preparationAccepted = true;
    assert(invalidReady->markWaitBridgePrepared());
    assert(invalidReady->markWaitBridgeAccepted());
    assert(invalidReady->markEntriesPreparing());
    assert(!invalidReady->markEntriesReady());

    VkPresentInfoKHR invalid{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    assert(!BatchPresentTransaction::create(
        handle<VkDevice>(1), handle<VkQueue>(2), 300, invalid));
    return 0;
}
