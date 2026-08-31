/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_normal_present_adapter.hpp"
#include "d3b3_per_swapchain_runtime.hpp"

#include <utility>

namespace lsfgvk::layer {

D3B3ProductionResourceState::Token::Token(D3B3ProductionResourceState* value,
        D3B3ProductionInputDomain domainValue, uintptr_t scope, uint64_t generation,
        uint64_t epoch) noexcept : owner(value), kind(domainValue), scopeValue(scope),
    generationValue(generation), epochValue(epoch) {}

D3B3ProductionResourceState::Token::Token(Token&& other) noexcept :
    owner(std::exchange(other.owner, nullptr)), kind(other.kind),
    scopeValue(std::exchange(other.scopeValue, 0)),
    generationValue(std::exchange(other.generationValue, 0)),
    epochValue(std::exchange(other.epochValue, 0)) {}

D3B3ProductionResourceState::Token&
D3B3ProductionResourceState::Token::operator=(Token&& other) noexcept {
    if (this == &other) return *this;
    retire();
    owner = std::exchange(other.owner, nullptr);
    kind = other.kind;
    scopeValue = std::exchange(other.scopeValue, 0);
    generationValue = std::exchange(other.generationValue, 0);
    epochValue = std::exchange(other.epochValue, 0);
    return *this;
}

D3B3ProductionResourceState::Token::~Token() { retire(); }

bool D3B3ProductionResourceState::Token::valid() const noexcept {
    return owner && scopeValue != 0 && generationValue != 0 && epochValue != 0;
}

void D3B3ProductionResourceState::Token::retire() noexcept {
    if (auto* value = std::exchange(owner, nullptr))
        value->outstanding.fetch_sub(1, std::memory_order_release);
    scopeValue = 0;
    generationValue = 0;
    epochValue = 0;
}

D3B3ProductionResourceState::Token D3B3ProductionResourceState::acquire(
        D3B3ProductionInputDomain domain, uintptr_t scope, uint64_t generation,
        uint64_t epoch) noexcept {
    if (scope == 0 || generation == 0 || epoch == 0)
        return {};
    outstanding.fetch_add(1, std::memory_order_acq_rel);
    return Token(this, domain, scope, generation, epoch);
}

D3B3NormalPresentAdapter::D3B3NormalPresentAdapter(
        D3B3NormalPresentContext value,
        D3B3AsyncFiniteComposition& composition) :
    contextValue(value), asyncComposition(&composition) {}

bool D3B3NormalPresentAdapter::structurallyReady() const noexcept {
    return readiness() == D3B3NormalAdapterResult::READY;
}

D3B3NormalAdapterResult D3B3NormalPresentAdapter::readiness() const noexcept {
    if (contextValue.swapchain == VK_NULL_HANDLE || contextValue.sourceImage == VK_NULL_HANDLE
            || contextValue.presentQueue == VK_NULL_HANDLE
            || contextValue.presentQueueFamily == VK_QUEUE_FAMILY_IGNORED
            || contextValue.swapchainGeneration == 0
            || !contextValue.runtimeDevicePairReady || !contextValue.exchangeChannelReady
            || !contextValue.terminalReady || !asyncComposition)
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
            || !asyncComposition)
        return currentResult;
    const auto result = asyncComposition->processFrame(frameId);
    switch (result) {
    case D3B3AsyncProgress::ACCEPTED:
        ownershipState = D3B3PresentOwnership::D3B3_ROUTE;
        nextFrameSerial = asyncComposition->counters().frames + 1;
        activeInput.reset();
        return currentResult = D3B3NormalAdapterResult::READY;
    case D3B3AsyncProgress::PENDING:
        ownershipState = D3B3PresentOwnership::D3B3_ROUTE;
        return currentResult = D3B3NormalAdapterResult::TEMPORARILY_BLOCKED;
    case D3B3AsyncProgress::FINISHED:
        ownershipState = D3B3PresentOwnership::TERMINAL;
        nextFrameSerial = asyncComposition->counters().frames + 1;
        activeInput.reset();
        return currentResult = D3B3NormalAdapterResult::FINITE_COMPLETE;
    case D3B3AsyncProgress::DEVICE_LOST:
        activeInput.reset();
        return currentResult = D3B3NormalAdapterResult::DEVICE_LOST;
    case D3B3AsyncProgress::FAILED:
        activeInput.reset();
        return currentResult = ownershipState == D3B3PresentOwnership::D3B3_ROUTE
            ? D3B3NormalAdapterResult::FAILED_AFTER_OWNERSHIP
            : D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
    }
    return currentResult = D3B3NormalAdapterResult::FAILED_AFTER_OWNERSHIP;
}

D3B3NormalAdapterResult D3B3NormalPresentAdapter::processFrame(
        D3B3AsyncPresentInput&& input) noexcept {
    const auto expectedScope = reinterpret_cast<uintptr_t>(contextValue.swapchain);
    if (activeInput || input.swapchain != contextValue.swapchain
            || input.swapchainGeneration != contextValue.swapchainGeneration
            || input.frameId == 0 || input.authorities.empty())
        return D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
    for (const auto& authority : input.authorities) {
        if (!authority.valid() || authority.scope() != expectedScope
                || authority.generation() != input.swapchainGeneration
                || authority.epoch() != input.frameId)
            return D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP;
    }
    activeInput.emplace(std::move(input));
    const auto result = processFrame(activeInput->frameId);
    if (result != D3B3NormalAdapterResult::TEMPORARILY_BLOCKED)
        activeInput.reset();
    return result;
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
    asyncComposition = nullptr;
    ownershipState = D3B3PresentOwnership::OUTER_ROUTE;
    currentResult = D3B3NormalAdapterResult::NOT_SELECTED;
    nextFrameSerial = 1;
}

}
