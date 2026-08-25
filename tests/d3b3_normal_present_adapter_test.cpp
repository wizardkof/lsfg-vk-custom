#include "d3b3_normal_present_adapter.hpp"

#include <cassert>

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

D3B3FiniteProductionOperations operations() {
        D3B3FiniteProductionOperations value;
        value.ingest = [](uint64_t, backend::TemporalSourceSlot, bool) { return true; };
        value.retireWarmupIngest = [] { return D3B3RetirementStatus::RETIRED; };
        value.generate = [](uint64_t, uint64_t, backend::TemporalSourceSlot,
                backend::TemporalSourceSlot, uint64_t) { return true; };
        value.presentWarmupOriginal = [](uint64_t) { return true; };
        value.returnGenerated = [](const backend::RuntimeTemporalPairIdentity&) {
            return backend::ReturnedGeneratedOperation{};
        };
        value.makeOriginal = [](const backend::RuntimeTemporalPairIdentity&) {
            return D3B3OriginalSourceAuthority{};
        };
        value.makeTerminal = [](const backend::RuntimeTemporalPairIdentity&) {
            return D3B3PairTerminalDispatch{};
        };
        value.stopWorkerAndJoin = [] { return true; };
        value.record = [](const std::string&) {};
        value.emitMarker = [] {};
        return value;
}

std::unique_ptr<D3B3ProductionRuntimeSession> runtime(uint64_t generation = 7) {
    return std::make_unique<D3B3ProductionRuntimeSession>(generation, operations());
}

void testMissingDependencyIsSafe() {
    auto context = readyContext();
    context.exchangeChannelReady = false;
    D3B3NormalPresentAdapter adapter(context, runtime());
    assert(!adapter.structurallyReady());
    assert(!adapter.constructOperations());
    assert(adapter.ownership() == D3B3PresentOwnership::OUTER_ROUTE);
}

void testOwnershipBoundary() {
    D3B3NormalPresentAdapter adapter(readyContext(), runtime());
    assert(adapter.structurallyReady());
    assert(adapter.constructOperations());
    assert(adapter.ownership() == D3B3PresentOwnership::OUTER_ROUTE);
    assert(adapter.acceptOwnership() == D3B3NormalAdapterResult::READY);
    assert(adapter.ownership() == D3B3PresentOwnership::D3B3_ROUTE);
    assert(adapter.acceptOwnership() == D3B3NormalAdapterResult::READY);
}

void testSwapchainGenerationReset() {
    D3B3NormalPresentAdapter adapter(readyContext(), runtime());
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

}

int main() {
    testMissingDependencyIsSafe();
    testOwnershipBoundary();
    testSwapchainGenerationReset();
}
