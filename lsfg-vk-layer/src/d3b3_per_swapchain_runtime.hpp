/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_normal_present_adapter.hpp"
#include "d3b3_production_core_owner.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace lsfgvk::layer {

enum class D3B3AsyncProgress : uint8_t {
    ACCEPTED, PENDING, FINISHED, DEVICE_LOST, FAILED
};

enum class D3B3PrePairPhase : uint8_t {
    IDLE, WARMUP_SUBMITTED, GENERATE_SUBMITTED, B_RETURN_SUBMITTED,
    A_RETURN_SUBMITTED, TERMINAL_READY, PAIR_SUBMITTED, PAIR_RETIRED, FAILED
};

struct D3B3AsyncFiniteOperations {
    // Returns the accepted ingest epoch; zero means rejection.
    std::function<uint64_t(uint64_t, backend::TemporalSourceSlot, bool)> submitIngest;
    std::function<D3B3RetirementStatus()> tryRetireIngest;
    std::function<bool(uint64_t)> presentWarmupOriginal;
    std::function<bool(backend::RuntimeTemporalPairIdentity)> submitGenerate;
    std::function<backend::RuntimeRetirementStatus()> tryRetireGenerate;
    std::function<void()> submitBReturn;
    std::function<backend::RuntimeRetirementStatus()> tryRetireBReturn;
    std::function<void(backend::RuntimeTemporalPairIdentity)> submitAReturn;
    std::function<backend::RuntimeRetirementStatus()> releaseGeneratedOutput;
    std::function<backend::ReturnedGeneratedOperation()> takeReturnedOperation;
    std::function<D3B3OriginalSourceAuthority(
        const backend::RuntimeTemporalPairIdentity&)> makeOriginal;
    std::function<D3B3PairTerminalDispatch(
        const backend::RuntimeTemporalPairIdentity&)> makeTerminal;
};

struct D3B3AsyncFiniteCounters {
    uint32_t frames{};
    uint32_t generateSubmits{};
    uint32_t bReturnSubmits{};
    uint32_t aReturnSubmits{};
    uint32_t pairConstructions{};
    uint32_t terminalSubmits{};
    uint32_t pairRetirements{};
    uint32_t pendingPolls{};
};

/// Persistent finite A-E production composition.  Every accepted driver
/// submit has a durable phase before this function returns; retries only poll
/// existing authorities and never resubmit an earlier phase.
class D3B3AsyncFiniteComposition {
public:
    explicit D3B3AsyncFiniteComposition(D3B3AsyncFiniteOperations);
    D3B3AsyncFiniteComposition(const D3B3AsyncFiniteComposition&) = delete;
    D3B3AsyncFiniteComposition& operator=(const D3B3AsyncFiniteComposition&) = delete;

    [[nodiscard]] D3B3AsyncProgress processFrame(uint64_t frameId) noexcept;
    [[nodiscard]] D3B3PrePairPhase phase() const noexcept { return current; }
    [[nodiscard]] D3B3AsyncFiniteCounters counters() const noexcept { return totals; }
    [[nodiscard]] bool retirementReady() const noexcept;
    [[nodiscard]] const D3B3PairOperation* activePair() const noexcept {
        return pair ? &*pair : nullptr;
    }

private:
    [[nodiscard]] D3B3AsyncProgress beginFrame(uint64_t);
    [[nodiscard]] D3B3AsyncProgress advanceActive();
    void fail() noexcept { current = D3B3PrePairPhase::FAILED; }

    D3B3AsyncFiniteOperations operations;
    std::array<D3B3TemporalProductionSlot, 2> slots{};
    std::optional<D3B3PairOperation> pair;
    backend::RuntimeTemporalPairIdentity activeIdentity{};
    uint64_t activeFrame{};
    uint64_t generation{};
    D3B3PrePairPhase current{D3B3PrePairPhase::IDLE};
    D3B3AsyncFiniteCounters totals{};
    bool terminalRetired{};
    bool aReturnRetired{};
    bool generateRetired{};
    bool bReturnRetired{};
    bool generatedOutputReleased{};
};

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
    std::function<D3B3AsyncFiniteOperations(
        D3B3ProductionCoreOwner&, const D3B3PerSwapchainRuntimeDescriptor&)>
        bindAsyncFiniteOperations;
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
    [[nodiscard]] D3B3AsyncFiniteComposition& asyncComposition() noexcept {
        return *asyncFinite;
    }

private:
    // Destruction is reverse declaration order: adapter/session/callbacks die
    // before the GPU core to which their finite bindings may refer.
    std::unique_ptr<D3B3ProductionCoreOwner> core;
    std::function<bool()> isRetirementReady;
    std::unique_ptr<D3B3AsyncFiniteComposition> asyncFinite;
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
