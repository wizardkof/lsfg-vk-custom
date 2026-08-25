/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_normal_present_adapter.hpp"

#include <utility>

namespace lsfgvk::layer {

D3B3NormalPresentAdapter::D3B3NormalPresentAdapter(
        D3B3NormalPresentContext value,
        std::unique_ptr<D3B3ProductionRuntimeSession> runtime) :
    contextValue(value), runtimeSession(std::move(runtime)) {}

bool D3B3NormalPresentAdapter::structurallyReady() const noexcept {
    return readiness() == D3B3NormalAdapterResult::READY;
}

D3B3NormalAdapterResult D3B3NormalPresentAdapter::readiness() const noexcept {
    if (contextValue.swapchain == VK_NULL_HANDLE || contextValue.sourceImage == VK_NULL_HANDLE
            || contextValue.presentQueue == VK_NULL_HANDLE
            || contextValue.presentQueueFamily == VK_QUEUE_FAMILY_IGNORED
            || contextValue.swapchainGeneration == 0
            || !contextValue.runtimeDevicePairReady || !contextValue.exchangeChannelReady
            || !contextValue.terminalReady || !runtimeSession
            || !runtimeSession->structurallyReady()
            || runtimeSession->generation() != contextValue.swapchainGeneration)
        return D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
    return D3B3NormalAdapterResult::READY;
}

bool D3B3NormalPresentAdapter::constructOperations() {
    if (!structurallyReady()) {
        currentResult = D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
        return false;
    }
    currentResult = D3B3NormalAdapterResult::READY;
    return true;
}

D3B3NormalAdapterResult D3B3NormalPresentAdapter::processFrame(uint64_t frameId) noexcept {
    if ((currentResult != D3B3NormalAdapterResult::READY
            && currentResult != D3B3NormalAdapterResult::TEMPORARILY_BLOCKED)
            || !runtimeSession)
        return currentResult;
    const auto result = runtimeSession->processFrame(frameId);
    switch (result) {
    case D3B3RuntimeFrameResult::ACCEPTED:
    case D3B3RuntimeFrameResult::COMPLETED:
        ownershipState = D3B3PresentOwnership::D3B3_ROUTE;
        nextFrameSerial = runtimeSession->controller().finiteCounters().applicationFrames + 1;
        return currentResult = D3B3NormalAdapterResult::READY;
    case D3B3RuntimeFrameResult::TEMPORARILY_BLOCKED:
        return currentResult = D3B3NormalAdapterResult::TEMPORARILY_BLOCKED;
    case D3B3RuntimeFrameResult::DEVICE_LOST:
        return currentResult = D3B3NormalAdapterResult::DEVICE_LOST;
    case D3B3RuntimeFrameResult::FAILED:
        return currentResult = runtimeSession->ownsFrame()
            ? D3B3NormalAdapterResult::FAILED_AFTER_OWNERSHIP
            : D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
    }
    return currentResult = D3B3NormalAdapterResult::FAILED_AFTER_OWNERSHIP;
}

D3B3NormalAdapterResult D3B3NormalPresentAdapter::acceptOwnership() noexcept {
    if (currentResult != D3B3NormalAdapterResult::READY
            || ownershipState != D3B3PresentOwnership::OUTER_ROUTE)
        return currentResult;
    ownershipState = D3B3PresentOwnership::D3B3_ROUTE;
    currentResult = D3B3NormalAdapterResult::READY;
    return currentResult;
}

void D3B3NormalPresentAdapter::resetForSwapchainGeneration(
        D3B3NormalPresentContext replacement) {
    if (replacement.swapchainGeneration == contextValue.swapchainGeneration)
        return;
    contextValue = replacement;
    runtimeSession.reset();
    ownershipState = D3B3PresentOwnership::OUTER_ROUTE;
    currentResult = D3B3NormalAdapterResult::NOT_SELECTED;
    nextFrameSerial = 1;
}

}
