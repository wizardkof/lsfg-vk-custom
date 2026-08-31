/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "batch_present_transaction.hpp"

#include <new>

namespace lsfgvk::layer {

std::optional<BatchPresentTransaction> BatchPresentTransaction::create(
        VkDevice device,
        VkQueue queue,
        uint64_t transactionId,
        const VkPresentInfoKHR& info,
        CreateFailureInjection failureInjection) noexcept {
    if (!device || !queue || !transactionId || !info.swapchainCount
            || !info.pSwapchains || !info.pImageIndices
            || (info.waitSemaphoreCount && !info.pWaitSemaphores))
        return std::nullopt;

    try {
        BatchPresentTransaction result;
        result.deviceValue = device;
        result.queueValue = queue;
        result.idValue = transactionId;
        result.hadPNext = info.pNext != nullptr;
        result.stateValue = State::Validated;

        // All storage needed to classify entries and build the final arrays is
        // committed before a bridge submit can be accepted.
        if (failureInjection == CreateFailureInjection::WaitStorage)
            throw std::bad_alloc{};
        if (info.waitSemaphoreCount)
            result.applicationWaitsValue.assign(
                info.pWaitSemaphores,
                info.pWaitSemaphores + info.waitSemaphoreCount);
        if (failureInjection == CreateFailureInjection::EntryStorage)
            throw std::bad_alloc{};
        result.entriesValue.reserve(info.swapchainCount);
        for (uint32_t i = 0; i < info.swapchainCount; ++i) {
            result.entriesValue.push_back(Entry{
                .applicationSwapchain = info.pSwapchains[i],
                .applicationImageIndex = info.pImageIndices[i]});
        }
        if (failureInjection == CreateFailureInjection::FinalStorage)
            throw std::bad_alloc{};
        result.finalSwapchainsValue.resize(info.swapchainCount);
        result.finalImageIndicesValue.resize(info.swapchainCount);
        // One completion authority per entry is the conservative upper bound.
        result.finalWaitsValue.resize(info.swapchainCount);
        result.stateValue = State::StorageReserved;
        return result;
    } catch (const std::bad_alloc&) {
        return std::nullopt;
    }
}

bool BatchPresentTransaction::transition(State from, State to) noexcept {
    if (stateValue != from) return false;
    stateValue = to;
    return true;
}

bool BatchPresentTransaction::markWaitBridgePrepared() noexcept {
    return transition(State::StorageReserved, State::WaitBridgePrepared);
}
bool BatchPresentTransaction::markWaitBridgeAccepted() noexcept {
    if (!transition(State::WaitBridgePrepared, State::WaitBridgeAccepted))
        return false;
    bridgeAcceptedValue = true;
    return true;
}
bool BatchPresentTransaction::markEntriesPreparing() noexcept {
    return transition(State::WaitBridgeAccepted, State::EntriesPreparing);
}
bool BatchPresentTransaction::markEntriesReady() noexcept {
    if (stateValue != State::EntriesPreparing) return false;
    for (const auto& value : entriesValue)
        if (value.path == Path::Unsupported
                || (value.path == Path::D2
                    && (!value.preparationAccepted
                        || !value.lifecycleGeneration))
                || (value.path == Path::VirtualPrepared
                    && (!value.preparationAccepted || !value.physicalSwapchain
                        || !value.completionSemaphore
                        || !value.deviceLifetimeIdentity
                        || !value.lifecycleGeneration
                        || !value.presentOperationIdentity)))
            return false;
    stateValue = State::EntriesReady;
    return true;
}
bool BatchPresentTransaction::setFinalEntry(uint32_t index,
        VkSwapchainKHR swapchain, uint32_t imageIndex) noexcept {
    if (stateValue != State::EntriesReady || index >= finalSwapchainsValue.size()
            || !swapchain)
        return false;
    finalSwapchainsValue[index] = swapchain;
    finalImageIndicesValue[index] = imageIndex;
    return true;
}
bool BatchPresentTransaction::setFinalWait(
        uint32_t index, VkSemaphore semaphore) noexcept {
    if (stateValue != State::EntriesReady || index >= finalWaitsValue.size()
            || !semaphore)
        return false;
    finalWaitsValue[index] = semaphore;
    if (index >= finalWaitCountValue) finalWaitCountValue = index + 1;
    return true;
}
bool BatchPresentTransaction::appendFinalWait(VkSemaphore semaphore) noexcept {
    if (stateValue != State::EntriesReady || !semaphore
            || finalWaitCountValue >= finalWaitsValue.size())
        return false;
    finalWaitsValue[finalWaitCountValue++] = semaphore;
    return true;
}
std::optional<VkPresentInfoKHR> BatchPresentTransaction::prepareFinalPresentInfo(
        const void* pNext, VkResult* pResults, uint32_t waitCount) noexcept {
    if (stateValue != State::EntriesReady || waitCount != finalWaitCountValue)
        return std::nullopt;
    for (const auto swapchain : finalSwapchainsValue)
        if (!swapchain) return std::nullopt;
    for (uint32_t i = 0; i < waitCount; ++i)
        if (!finalWaitsValue[i]) return std::nullopt;
    stateValue = State::FinalPresentPrepared;
    return VkPresentInfoKHR{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = pNext,
        .waitSemaphoreCount = waitCount,
        .pWaitSemaphores = waitCount ? finalWaitsValue.data() : nullptr,
        .swapchainCount = static_cast<uint32_t>(finalSwapchainsValue.size()),
        .pSwapchains = finalSwapchainsValue.data(),
        .pImageIndices = finalImageIndicesValue.data(),
        .pResults = pResults};
}
bool BatchPresentTransaction::markFinalPresentPrepared() noexcept {
    return transition(State::EntriesReady, State::FinalPresentPrepared);
}
bool BatchPresentTransaction::markFinalPresentCalled() noexcept {
    return transition(State::FinalPresentPrepared, State::FinalPresentCalled);
}
bool BatchPresentTransaction::markFinalPresentResult(VkResult result) noexcept {
    if (stateValue != State::FinalPresentCalled) return false;
    finalResultValue = result;
    finalResultClassValue = classifyPresentResult(result);
    switch (finalResultClassValue) {
    case PresentResultClass::Normal:
        stateValue = State::FinalPresentNormal;
        break;
    case PresentResultClass::EnqueuedRejection:
        stateValue = State::FinalPresentEnqueuedRejection;
        break;
    case PresentResultClass::PreEnqueueFailure:
        stateValue = State::FinalPresentPreEnqueueFailure;
        break;
    case PresentResultClass::DeviceLost:
        stateValue = State::DeviceLost;
        break;
    case PresentResultClass::Indeterminate:
        stateValue = State::FinalPresentIndeterminate;
        break;
    }
    return true;
}
bool BatchPresentTransaction::markRetirementPending() noexcept {
    if (stateValue != State::FinalPresentNormal
            && stateValue != State::FinalPresentEnqueuedRejection
            && stateValue != State::FinalPresentIndeterminate)
        return false;
    stateValue = State::RetirementPending;
    return true;
}
bool BatchPresentTransaction::markRetired() noexcept {
    return transition(State::RetirementPending, State::Retired);
}

bool BatchPresentTransaction::hasQueueSideEffect() const noexcept {
    return bridgeAcceptedValue;
}

bool BatchPresentTransaction::applicationWaitsConsumed() const noexcept {
    return bridgeAcceptedValue;
}

} // namespace lsfgvk::layer
