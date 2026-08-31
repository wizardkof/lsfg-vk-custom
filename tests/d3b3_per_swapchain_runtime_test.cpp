/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_per_swapchain_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

using namespace lsfgvk::layer;

namespace {

VkSwapchainKHR handle(uintptr_t value) {
    return reinterpret_cast<VkSwapchainKHR>(value);
}

D3B3FiniteProductionOperations inertOperations() {
    D3B3FiniteProductionOperations operations;
    operations.ingest = [](uint64_t, lsfgvk::backend::TemporalSourceSlot, bool) { return true; };
    operations.retireWarmupIngest = [] { return D3B3RetirementStatus::RETIRED; };
    operations.generate = [](uint64_t, uint64_t, lsfgvk::backend::TemporalSourceSlot,
        lsfgvk::backend::TemporalSourceSlot, uint64_t) { return true; };
    operations.presentWarmupOriginal = [](uint64_t) { return true; };
    operations.returnGenerated = [](const lsfgvk::backend::RuntimeTemporalPairIdentity&) {
        return lsfgvk::backend::ReturnedGeneratedOperation{};
    };
    operations.makeOriginal = [](const lsfgvk::backend::RuntimeTemporalPairIdentity&) {
        return D3B3OriginalSourceAuthority{};
    };
    operations.makeTerminal = [](const lsfgvk::backend::RuntimeTemporalPairIdentity&) {
        return D3B3PairTerminalDispatch{};
    };
    operations.stopWorkerAndJoin = [] { return true; };
    operations.record = [](const std::string&) {};
    operations.emitMarker = [] {};
    return operations;
}

D3B3AsyncFiniteOperations inertAsyncOperations() {
    D3B3AsyncFiniteOperations operations;
    operations.submitIngest = [](uint64_t, lsfgvk::backend::TemporalSourceSlot, bool) {
        return 1;
    };
    operations.tryRetireIngest = [] { return D3B3RetirementStatus::RETIRED; };
    operations.presentWarmupOriginal = [](uint64_t) { return true; };
    operations.submitGenerate = [](lsfgvk::backend::RuntimeTemporalPairIdentity) {
        return true;
    };
    operations.tryRetireGenerate = [] {
        return lsfgvk::backend::RuntimeRetirementStatus::RETIRED;
    };
    operations.submitBReturn = [] {};
    operations.tryRetireBReturn = [] {
        return lsfgvk::backend::RuntimeRetirementStatus::RETIRED;
    };
    operations.submitAReturn = [](lsfgvk::backend::RuntimeTemporalPairIdentity) {};
    operations.releaseGeneratedOutput = [] {
        return lsfgvk::backend::RuntimeRetirementStatus::RETIRED;
    };
    operations.takeReturnedOperation = [] {
        return lsfgvk::backend::ReturnedGeneratedOperation{};
    };
    operations.makeOriginal = [](const lsfgvk::backend::RuntimeTemporalPairIdentity&) {
        return D3B3OriginalSourceAuthority{};
    };
    operations.makeTerminal = [](const lsfgvk::backend::RuntimeTemporalPairIdentity&) {
        return D3B3PairTerminalDispatch{};
    };
    return operations;
}

struct Fixture {
    vk::RuntimeDevicePair pair{};
    vk::RuntimeExchangeEndpoint generation{};
    vk::RuntimeExchangeEndpoint render{};

    Fixture() {
        render.identity.name = "render";
        render.identity.vendorId = 1;
        render.identity.deviceId = 1;
        render.identity.deviceUuid[0] = 1;
        generation.identity.name = "generation";
        generation.identity.vendorId = 2;
        generation.identity.deviceId = 2;
        generation.identity.deviceUuid[0] = 2;
        render.bufferDevice.device = reinterpret_cast<VkDevice>(uintptr_t{1});
        generation.bufferDevice.device = reinterpret_cast<VkDevice>(uintptr_t{2});
        pair.render = {render.identity, vk::RuntimeDeviceOwnership::ApplicationManaged};
        pair.generation = {generation.identity, vk::RuntimeDeviceOwnership::BackendManaged};
        pair.mode = vk::RuntimeDevicePairMode::CrossPhysicalDevice;
    }

    D3B3PerSwapchainRuntimeDescriptor descriptor(uintptr_t swapchain, uint64_t generationId) {
        D3B3PerSwapchainRuntimeDescriptor value;
        value.present.swapchain = handle(swapchain);
        value.present.sourceImage = reinterpret_cast<VkImage>(uintptr_t{3});
        value.present.format = VK_FORMAT_R8G8B8A8_UNORM;
        value.present.extent = {16, 16};
        value.present.presentQueue = reinterpret_cast<VkQueue>(uintptr_t{4});
        value.present.presentQueueFamily = 0;
        value.present.swapchainGeneration = generationId;
        value.present.runtimeDevicePairReady = true;
        value.present.exchangeChannelReady = true;
        value.present.terminalReady = true;
        value.core.backend = reinterpret_cast<lsfgvk::backend::Instance*>(uintptr_t{5});
        value.core.devicePair = &pair;
        value.core.exchangeChannel = reinterpret_cast<const vk::RuntimeExchangeChannel*>(uintptr_t{6});
        value.core.generationEndpoint = generation;
        value.core.renderEndpoint = render;
        value.core.extent = {16, 16};
        value.core.format = VK_FORMAT_R8G8B8A8_UNORM;
        value.core.sourceImage = [](lsfgvk::backend::TemporalSourceSlot) {
            return reinterpret_cast<VkImage>(uintptr_t{7});
        };
        return value;
    }
};

}

int main() {
    Fixture fixture;
    D3B3PerSwapchainRuntimeFactory factory;
    size_t bindings{};
    auto resources = std::make_shared<D3B3ProductionResourceState>();
    const D3B3PerSwapchainRuntimeAssembly assembly{
        .bindAsyncFiniteOperations = [&](D3B3ProductionCoreOwner& core,
                const D3B3PerSwapchainRuntimeDescriptor&) {
            assert(core.structurallyReady());
            ++bindings;
            return inertAsyncOperations();
        },
        .resourceState = resources};

    auto first = factory.create(fixture.descriptor(0x100, 1), assembly);
    auto second = factory.create(fixture.descriptor(0x200, 1), assembly);
    assert(factory.size() == 2 && bindings == 2);
    assert(first != second && &first->coreOwner() != &second->coreOwner());
    assert(factory.find(handle(0x100)) == first);

    // Destroy removes lookup ownership, but an in-flight handle remains safe.
    assert(factory.destroy(handle(0x100)));
    assert(!factory.find(handle(0x100)) && factory.find(handle(0x200)) == second);
    assert(first->structurallyReady());

    auto replacement = factory.recreate(fixture.descriptor(0x200, 2), assembly);
    assert(replacement != second && replacement->generation() == 2);
    assert(second->generation() == 1); // old in-flight generation stays alive

    // Failed local construction cannot replace or publish an entry.
    const D3B3PerSwapchainRuntimeAssembly failing{
        .bindAsyncFiniteOperations = [](D3B3ProductionCoreOwner&,
                const D3B3PerSwapchainRuntimeDescriptor&) -> D3B3AsyncFiniteOperations {
            throw std::runtime_error("injected assembly failure");
        },
        .resourceState = resources};
    try {
        (void)factory.recreate(fixture.descriptor(0x200, 3), failing);
        assert(false);
    } catch (const std::runtime_error&) {}
    assert(factory.find(handle(0x200)) == replacement);
    assert(factory.size() == 1);

    // Duplicate create and non-advancing recreate preserve the existing owner.
    try {
        (void)factory.create(fixture.descriptor(0x200, 3), assembly);
        assert(false);
    } catch (const std::logic_error&) {}
    try {
        (void)factory.recreate(fixture.descriptor(0x200, 2), assembly);
        assert(false);
    } catch (const std::invalid_argument&) {}
    assert(factory.find(handle(0x200)) == replacement);

    D3B3PerSwapchainRuntimeFactory activeFactory;
    auto activeResources = std::make_shared<D3B3ProductionResourceState>();
    auto activeToken = activeResources->acquire(
        D3B3ProductionInputDomain::FRAME_TRANSPORT, 0x300, 1, 1);
    const D3B3PerSwapchainRuntimeAssembly active{
        .bindAsyncFiniteOperations = [](D3B3ProductionCoreOwner&,
                const D3B3PerSwapchainRuntimeDescriptor&) {
            return inertAsyncOperations();
        },
        .resourceState = activeResources};
    auto pending = activeFactory.create(fixture.descriptor(0x300, 1), active);
    assert(!activeFactory.destroy(handle(0x300)));
    try {
        (void)activeFactory.recreate(fixture.descriptor(0x300, 2), assembly);
        assert(false);
    } catch (const std::logic_error&) {}
    assert(activeFactory.find(handle(0x300)) == pending);

    // Two owners independently reach the fully GPU-chained pre-Pair point.
    // Polling or invalid input for one owner cannot advance the other, and an
    // active owner cannot be destroyed or recreated.
    D3B3PerSwapchainRuntimeFactory concurrentFactory;
    auto left = concurrentFactory.create(fixture.descriptor(0x400, 1), assembly);
    auto right = concurrentFactory.create(fixture.descriptor(0x500, 1), assembly);
    for (const uint64_t frame : {uint64_t{'A'}, uint64_t{'B'}}) {
        assert(left->asyncComposition().processFrame(frame) == D3B3AsyncProgress::PENDING);
        assert(right->asyncComposition().processFrame(frame) == D3B3AsyncProgress::PENDING);
        assert(left->asyncComposition().processFrame(frame) == D3B3AsyncProgress::ACCEPTED);
        assert(right->asyncComposition().processFrame(frame) == D3B3AsyncProgress::ACCEPTED);
    }
    assert(left->asyncComposition().processFrame(uint64_t{'C'})
        == D3B3AsyncProgress::PENDING);
    assert(left->asyncComposition().phase() == D3B3PrePairPhase::A_RETURN_SUBMITTED);
    assert(right->asyncComposition().phase() == D3B3PrePairPhase::IDLE);
    assert(right->asyncComposition().processFrame(uint64_t{'C'})
        == D3B3AsyncProgress::PENDING);
    assert(right->asyncComposition().phase() == D3B3PrePairPhase::A_RETURN_SUBMITTED);
    const auto leftCounts = left->asyncComposition().counters();
    const auto rightCounts = right->asyncComposition().counters();
    assert(leftCounts.generateSubmits == 1 && leftCounts.bReturnSubmits == 1
        && leftCounts.aReturnSubmits == 1);
    assert(rightCounts.generateSubmits == 1 && rightCounts.bReturnSubmits == 1
        && rightCounts.aReturnSubmits == 1);
    assert(!concurrentFactory.destroy(handle(0x400)));
    try {
        (void)concurrentFactory.recreate(fixture.descriptor(0x400, 2), assembly);
        assert(false);
    } catch (const std::logic_error&) {}
    assert(concurrentFactory.find(handle(0x400)) == left);
    assert(concurrentFactory.find(handle(0x500)) == right);

    enum class SubmitFailure { Generate, BReturn, AReturn };
    for (const auto failure : {SubmitFailure::Generate, SubmitFailure::BReturn,
            SubmitFailure::AReturn}) {
        auto operations = inertAsyncOperations();
        operations.submitGenerate = [failure](auto) {
            return failure != SubmitFailure::Generate;
        };
        operations.submitBReturn = [failure] {
            if (failure == SubmitFailure::BReturn)
                throw std::runtime_error("B-return rejected");
        };
        operations.submitAReturn = [failure](auto) {
            if (failure == SubmitFailure::AReturn)
                throw std::runtime_error("A-return rejected");
        };
        D3B3AsyncFiniteComposition composition(std::move(operations));
        for (const uint64_t frame : {uint64_t{'A'}, uint64_t{'B'}}) {
            assert(composition.processFrame(frame) == D3B3AsyncProgress::PENDING);
            assert(composition.processFrame(frame) == D3B3AsyncProgress::ACCEPTED);
        }
        assert(composition.processFrame(uint64_t{'C'}) == D3B3AsyncProgress::FAILED);
        assert(composition.phase() == D3B3PrePairPhase::FAILED);
        const auto counts = composition.counters();
        assert(counts.generateSubmits
            == (failure == SubmitFailure::Generate ? 0u : 1u));
        assert(counts.bReturnSubmits
            == (failure == SubmitFailure::AReturn ? 1u : 0u));
        assert(counts.aReturnSubmits == 0);
        assert(composition.processFrame(uint64_t{'C'}) == D3B3AsyncProgress::FAILED);
    }

    auto deviceLostOperations = inertAsyncOperations();
    deviceLostOperations.tryRetireGenerate = [] {
        return lsfgvk::backend::RuntimeRetirementStatus::DEVICE_LOST;
    };
    deviceLostOperations.tryRetireBReturn = [] {
        return lsfgvk::backend::RuntimeRetirementStatus::NOT_READY;
    };
    D3B3AsyncFiniteComposition lost(std::move(deviceLostOperations));
    for (const uint64_t frame : {uint64_t{'A'}, uint64_t{'B'}}) {
        assert(lost.processFrame(frame) == D3B3AsyncProgress::PENDING);
        assert(lost.processFrame(frame) == D3B3AsyncProgress::ACCEPTED);
    }
    assert(lost.processFrame(uint64_t{'C'}) == D3B3AsyncProgress::PENDING);
    assert(lost.processFrame(uint64_t{'C'}) == D3B3AsyncProgress::DEVICE_LOST);
    assert(lost.phase() == D3B3PrePairPhase::FAILED);
    assert(lost.processFrame(uint64_t{'C'}) == D3B3AsyncProgress::FAILED);
}
