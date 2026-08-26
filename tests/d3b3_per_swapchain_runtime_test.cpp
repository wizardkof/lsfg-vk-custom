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
    const D3B3PerSwapchainRuntimeAssembly assembly{
        .bindFiniteOperations = [&](D3B3ProductionCoreOwner& core,
                const D3B3PerSwapchainRuntimeDescriptor&) {
            assert(core.structurallyReady());
            ++bindings;
            return inertOperations();
        },
        .bindRetirementReady = [](D3B3ProductionCoreOwner&) {
            return [] { return true; };
        }};

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
        .bindFiniteOperations = [](D3B3ProductionCoreOwner&,
                const D3B3PerSwapchainRuntimeDescriptor&) -> D3B3FiniteProductionOperations {
            throw std::runtime_error("injected assembly failure");
        },
        .bindRetirementReady = [](D3B3ProductionCoreOwner&) {
            return [] { return true; };
        }};
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
    const D3B3PerSwapchainRuntimeAssembly active{
        .bindFiniteOperations = [](D3B3ProductionCoreOwner&,
                const D3B3PerSwapchainRuntimeDescriptor&) { return inertOperations(); },
        .bindRetirementReady = [](D3B3ProductionCoreOwner&) {
            return [] { return false; };
        }};
    auto pending = activeFactory.create(fixture.descriptor(0x300, 1), active);
    assert(!activeFactory.destroy(handle(0x300)));
    try {
        (void)activeFactory.recreate(fixture.descriptor(0x300, 2), assembly);
        assert(false);
    } catch (const std::logic_error&) {}
    assert(activeFactory.find(handle(0x300)) == pending);
}
