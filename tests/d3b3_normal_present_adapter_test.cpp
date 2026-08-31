#include "d3b3_per_swapchain_runtime.hpp"

#include <cassert>
#include <type_traits>

using namespace lsfgvk::layer;
namespace backend = lsfgvk::backend;

namespace {

template<typename T>
T handle(uintptr_t value) {
    return reinterpret_cast<T>(value);
}

D3B3NormalPresentContext readyContext() {
    return {
        .swapchain = handle<VkSwapchainKHR>(1),
        .sourceImage = handle<VkImage>(2),
        .presentQueue = handle<VkQueue>(3),
        .presentQueueFamily = 4,
        .swapchainGeneration = 7,
        .runtimeDevicePairReady = true,
        .exchangeChannelReady = true,
        .terminalReady = true};
}

D3B3AsyncFiniteOperations operations() {
    D3B3AsyncFiniteOperations value;
    value.submitIngest = [](uint64_t, backend::TemporalSourceSlot, bool) { return 1; };
    value.tryRetireIngest = [] { return D3B3RetirementStatus::RETIRED; };
    value.presentWarmupOriginal = [](uint64_t) { return true; };
    value.submitGenerate = [](backend::RuntimeTemporalPairIdentity) { return true; };
    value.tryRetireGenerate = [] { return backend::RuntimeRetirementStatus::RETIRED; };
    value.submitBReturn = [] {};
    value.tryRetireBReturn = [] { return backend::RuntimeRetirementStatus::RETIRED; };
    value.submitAReturn = [](backend::RuntimeTemporalPairIdentity) {};
    value.releaseGeneratedOutput = [] { return backend::RuntimeRetirementStatus::RETIRED; };
    value.takeReturnedOperation = [] { return backend::ReturnedGeneratedOperation{}; };
    value.makeOriginal = [](const backend::RuntimeTemporalPairIdentity&) {
        return D3B3OriginalSourceAuthority{};
    };
    value.makeTerminal = [](const backend::RuntimeTemporalPairIdentity&) {
        return D3B3PairTerminalDispatch{};
    };
    return value;
}

void testMissingDependencyIsSafe() {
    auto context = readyContext();
    context.exchangeChannelReady = false;
    D3B3AsyncFiniteComposition composition(operations());
    D3B3NormalPresentAdapter adapter(context, composition);
    assert(!adapter.structurallyReady());
    assert(!adapter.constructOperations());
    assert(adapter.ownership() == D3B3PresentOwnership::OUTER_ROUTE);
}

void testOwnershipBoundary() {
    D3B3AsyncFiniteComposition composition(operations());
    D3B3NormalPresentAdapter adapter(readyContext(), composition);
    assert(adapter.structurallyReady());
    assert(adapter.constructOperations());
    assert(adapter.ownership() == D3B3PresentOwnership::OUTER_ROUTE);
    assert(adapter.acceptOwnership() == D3B3NormalAdapterResult::READY);
    assert(adapter.ownership() == D3B3PresentOwnership::D3B3_ROUTE);
    assert(adapter.acceptOwnership() == D3B3NormalAdapterResult::READY);
}

void testSwapchainGenerationReset() {
    D3B3AsyncFiniteComposition composition(operations());
    D3B3NormalPresentAdapter adapter(readyContext(), composition);
    assert(adapter.constructOperations());
    auto replacement = readyContext();
    replacement.swapchainGeneration = 8;
    replacement.swapchain = handle<VkSwapchainKHR>(9);
    adapter.resetForSwapchainGeneration(replacement);
    assert(adapter.swapchainGeneration() == 8);
    assert(adapter.frameSerial() == 1);
    assert(adapter.ownership() == D3B3PresentOwnership::OUTER_ROUTE);
    assert(adapter.readiness() == D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP);
}

void testScopedMoveOnlyPresentInput() {
    static_assert(!std::is_copy_constructible_v<D3B3ProductionResourceState::Token>);
    static_assert(!std::is_copy_constructible_v<D3B3AsyncPresentInput>);
    D3B3ProductionResourceState resources;
    auto context = readyContext();
    const auto scope = reinterpret_cast<uintptr_t>(context.swapchain);
    auto token = resources.acquire(D3B3ProductionInputDomain::FRAME_TRANSPORT,
        scope, context.swapchainGeneration, 1);
    assert(token.valid() && resources.outstandingAuthorities() == 1);
    auto moved = std::move(token);
    assert(moved.valid() && !token.valid());

    D3B3AsyncFiniteComposition composition(operations());
    D3B3NormalPresentAdapter adapter(context, composition);
    assert(adapter.constructOperations());
    D3B3AsyncPresentInput input(context.swapchain,
        context.swapchainGeneration, 1);
    input.authorities.emplace_back(std::move(moved));
    assert(adapter.processFrame(std::move(input))
        == D3B3NormalAdapterResult::TEMPORARILY_BLOCKED);
    assert(resources.outstandingAuthorities() == 1);

    {
        auto wrong = resources.acquire(D3B3ProductionInputDomain::ORIGINAL_READY,
            scope + 1, context.swapchainGeneration, 2);
        D3B3AsyncPresentInput rejected(context.swapchain,
            context.swapchainGeneration, 2);
        rejected.authorities.emplace_back(std::move(wrong));
        assert(adapter.processFrame(std::move(rejected))
            == D3B3NormalAdapterResult::FAILED_BEFORE_OWNERSHIP);
        // Pre-accept rejection leaves the authority with its caller.
        assert(resources.outstandingAuthorities() == 2);
    }

    assert(adapter.processFrame(uint64_t{1}) == D3B3NormalAdapterResult::READY);
    // The accepted frame releases its provider authorities exactly once.
    assert(resources.outstandingAuthorities() == 0);
}

}

int main() {
    testMissingDependencyIsSafe();
    testOwnershipBoundary();
    testSwapchainGenerationReset();
    testScopedMoveOnlyPresentInput();
}
