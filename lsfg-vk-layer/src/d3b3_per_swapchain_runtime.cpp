/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_per_swapchain_runtime.hpp"

#include <stdexcept>
#include <utility>

namespace lsfgvk::layer {

D3B3PerSwapchainRuntimeOwner::D3B3PerSwapchainRuntimeOwner(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    if (descriptor.present.swapchain == VK_NULL_HANDLE
            || descriptor.present.swapchainGeneration == 0
            || !assembly.bindFiniteOperations || !assembly.bindRetirementReady)
        throw std::invalid_argument("incomplete D3B3 per-swapchain assembly");

    auto localCore = std::make_unique<D3B3ProductionCoreOwner>(descriptor.core);
    auto retirement = assembly.bindRetirementReady(*localCore);
    if (!retirement)
        throw std::invalid_argument("D3B3 retirement authority is incomplete");
    auto operations = assembly.bindFiniteOperations(*localCore, descriptor);
    auto runtime = std::make_unique<D3B3ProductionRuntimeSession>(
        descriptor.present.swapchainGeneration, std::move(operations));
    if (!runtime->structurallyReady())
        throw std::invalid_argument("D3B3 finite runtime assembly is incomplete");
    auto adapter = std::make_unique<D3B3NormalPresentAdapter>(
        descriptor.present, std::move(runtime));
    if (!adapter->constructOperations())
        throw std::invalid_argument("D3B3 normal adapter prerequisites are incomplete");

    core = std::move(localCore);
    isRetirementReady = std::move(retirement);
    normalAdapter = std::move(adapter);
}

VkSwapchainKHR D3B3PerSwapchainRuntimeOwner::swapchain() const noexcept {
    return normalAdapter ? normalAdapter->context().swapchain : VK_NULL_HANDLE;
}

uint64_t D3B3PerSwapchainRuntimeOwner::generation() const noexcept {
    return normalAdapter ? normalAdapter->swapchainGeneration() : 0;
}

bool D3B3PerSwapchainRuntimeOwner::structurallyReady() const noexcept {
    return core && core->structurallyReady() && normalAdapter
        && normalAdapter->structurallyReady();
}

bool D3B3PerSwapchainRuntimeOwner::retirementReady() const noexcept {
    try {
        return isRetirementReady && isRetirementReady();
    } catch (...) {
        return false;
    }
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::create(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    auto candidate = std::make_shared<Owner>(std::move(descriptor), assembly);
    const auto key = candidate->swapchain();
    std::scoped_lock lock(mutex);
    const auto [iterator, inserted] = owners.emplace(key, candidate);
    if (!inserted)
        throw std::logic_error("D3B3 swapchain runtime already exists");
    return iterator->second;
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::recreate(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    auto candidate = std::make_shared<Owner>(std::move(descriptor), assembly);
    const auto key = candidate->swapchain();
    std::scoped_lock lock(mutex);
    auto iterator = owners.find(key);
    if (iterator == owners.end())
        throw std::logic_error("D3B3 swapchain runtime does not exist");
    if (candidate->generation() <= iterator->second->generation())
        throw std::invalid_argument("D3B3 swapchain generation did not advance");
    if (!iterator->second->retirementReady())
        throw std::logic_error("D3B3 previous swapchain generation is still active");
    iterator->second = candidate;
    return candidate;
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::find(VkSwapchainKHR swapchain) const {
    std::scoped_lock lock(mutex);
    const auto iterator = owners.find(swapchain);
    return iterator == owners.end() ? nullptr : iterator->second;
}

bool D3B3PerSwapchainRuntimeFactory::destroy(VkSwapchainKHR swapchain) {
    std::scoped_lock lock(mutex);
    const auto iterator = owners.find(swapchain);
    if (iterator == owners.end() || !iterator->second->retirementReady())
        return false;
    owners.erase(iterator);
    return true;
}

size_t D3B3PerSwapchainRuntimeFactory::size() const noexcept {
    std::scoped_lock lock(mutex);
    return owners.size();
}

}
