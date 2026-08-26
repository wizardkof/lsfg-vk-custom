/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_normal_present_adapter.hpp"
#include "d3b3_production_core_owner.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace lsfgvk::layer {

/// Immutable input used to assemble one D3B3 runtime generation.  The normal
/// present route does not create or consult this object until A6B selects it.
struct D3B3PerSwapchainRuntimeDescriptor {
    D3B3NormalPresentContext present;
    D3B3ProductionCoreOwner::Dependencies core;
};

/// Production composition seam.  The callback binds the already-qualified
/// finite operations to this owner's real core; it is not an execution API.
struct D3B3PerSwapchainRuntimeAssembly {
    std::function<D3B3FiniteProductionOperations(
        D3B3ProductionCoreOwner&, const D3B3PerSwapchainRuntimeDescriptor&)>
        bindFiniteOperations;
    std::function<std::function<bool()>(D3B3ProductionCoreOwner&)>
        bindRetirementReady;
};

class D3B3PerSwapchainRuntimeOwner {
public:
    D3B3PerSwapchainRuntimeOwner(D3B3PerSwapchainRuntimeDescriptor,
        const D3B3PerSwapchainRuntimeAssembly&);
    D3B3PerSwapchainRuntimeOwner(const D3B3PerSwapchainRuntimeOwner&) = delete;
    D3B3PerSwapchainRuntimeOwner& operator=(const D3B3PerSwapchainRuntimeOwner&) = delete;
    ~D3B3PerSwapchainRuntimeOwner() = default;

    [[nodiscard]] VkSwapchainKHR swapchain() const noexcept;
    [[nodiscard]] uint64_t generation() const noexcept;
    [[nodiscard]] bool structurallyReady() const noexcept;
    [[nodiscard]] bool retirementReady() const noexcept;
    [[nodiscard]] D3B3ProductionCoreOwner& coreOwner() noexcept { return *core; }
    [[nodiscard]] D3B3NormalPresentAdapter& adapter() noexcept { return *normalAdapter; }

private:
    // Destruction is reverse declaration order: adapter/session/callbacks die
    // before the GPU core to which their finite bindings may refer.
    std::unique_ptr<D3B3ProductionCoreOwner> core;
    std::function<bool()> isRetirementReady;
    std::unique_ptr<D3B3NormalPresentAdapter> normalAdapter;
};

/// Thread-safe ownership registry.  Construction happens outside the lock and
/// a complete owner is committed in one map operation.  shared_ptr lookup
/// keeps an in-flight future A6B caller alive across destroy/recreate.
class D3B3PerSwapchainRuntimeFactory {
public:
    using Owner = D3B3PerSwapchainRuntimeOwner;

    [[nodiscard]] std::shared_ptr<Owner> create(
        D3B3PerSwapchainRuntimeDescriptor,
        const D3B3PerSwapchainRuntimeAssembly&);
    [[nodiscard]] std::shared_ptr<Owner> recreate(
        D3B3PerSwapchainRuntimeDescriptor,
        const D3B3PerSwapchainRuntimeAssembly&);
    [[nodiscard]] std::shared_ptr<Owner> find(VkSwapchainKHR) const;
    bool destroy(VkSwapchainKHR);
    [[nodiscard]] size_t size() const noexcept;

private:
    mutable std::mutex mutex;
    std::unordered_map<VkSwapchainKHR, std::shared_ptr<Owner>> owners;
};

}
