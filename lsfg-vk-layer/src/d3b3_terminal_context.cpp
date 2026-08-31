/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_terminal_context.hpp"

#include <atomic>
#include <algorithm>
#include <stdexcept>

namespace lsfgvk::layer {

namespace {
std::atomic<uint64_t> nextDeviceLifetimeIdentity{1};
std::atomic<uint64_t> nextTerminalContextIdentity{1};
}

D3B3DeviceLifetimeQuarantine::D3B3DeviceLifetimeQuarantine() :
    identityValue(nextDeviceLifetimeIdentity.fetch_add(1, std::memory_order_relaxed)) {}
D3B3DeviceLifetimeQuarantine::~D3B3DeviceLifetimeQuarantine() = default;

size_t D3B3DeviceLifetimeQuarantine::recordCount() const noexcept {
    const std::scoped_lock lock(mutex);
    return retainedRecordCount;
}

bool D3B3DeviceLifetimeQuarantine::reserve(uint64_t contextIdentity,
        uint64_t leaseIdentity) noexcept {
    if (contextIdentity == 0 || leaseIdentity == 0) return false;
    const std::scoped_lock lock(mutex);
    for (const auto& slot : records) {
        if (slot && slot->contextIdentity == contextIdentity
                && slot->leaseIdentity == leaseIdentity)
            return true;
    }
    for (auto& slot : records) {
        if (slot) continue;
        slot.emplace(Record{contextIdentity, leaseIdentity, {}});
        return true;
    }
    return false;
}

void D3B3DeviceLifetimeQuarantine::releaseReservation(
        uint64_t contextIdentity, uint64_t leaseIdentity) noexcept {
    const std::scoped_lock lock(mutex);
    for (auto& slot : records) {
        if (!slot || slot->contextIdentity != contextIdentity
                || slot->leaseIdentity != leaseIdentity)
            continue;
        // Once physical backing has been retained for a lost device, ordinary
        // lease destruction must not release it.  The record then lives until
        // the real device-lifetime teardown.
        if (!slot->backing) slot.reset();
        return;
    }
}

bool D3B3DeviceLifetimeQuarantine::retain(uint64_t contextIdentity,
        uint64_t leaseIdentity, std::shared_ptr<void> backing) noexcept {
    if (!backing) return false;
    const std::scoped_lock lock(mutex);
    for (auto& slot : records) {
        if (!slot || slot->contextIdentity != contextIdentity
                || slot->leaseIdentity != leaseIdentity)
            continue;
        if (slot->backing) return true;
        // The Record itself was reserved before submit. shared_ptr assignment is
        // allocation-free and converts that reservation into device-lifetime
        // ownership of the already-accepted backing.
        slot->backing = std::move(backing);
        ++retainedRecordCount;
        return true;
    }
    return false;
}

struct D3B3PerSwapchainTerminalContext::State {
    State(const vk::Vulkan& value, uint32_t family, uint64_t generation,
            std::shared_ptr<D3B3DeviceLifetimeQuarantine> owner) :
        vk(&value), destinationFamily(family), lifecycle(generation),
        contextIdentity(nextTerminalContextIdentity.fetch_add(
            1, std::memory_order_relaxed)),
        deviceLifetime(std::move(owner)) {
        if (family == VK_QUEUE_FAMILY_IGNORED || generation == 0
                || deviceLifetime.expired())
            throw std::invalid_argument("invalid D3B3 terminal context");
    }
    const vk::Vulkan* vk;
    uint32_t destinationFamily;
    uint64_t lifecycle;
    uint64_t contextIdentity;
    std::atomic<uint32_t> activeLeases{0};
    std::atomic<uint64_t> nextLease{1};
    std::atomic<bool> deviceLost{false};
    std::weak_ptr<D3B3DeviceLifetimeQuarantine> deviceLifetime;
    std::mutex acceptedMutex;
    static constexpr size_t MAX_TRACKED_BACKINGS = 4096;
    std::array<std::weak_ptr<void>, MAX_TRACKED_BACKINGS> acceptedBackings{};
};

struct D3B3AReturnHandoffLease::State {
    State(std::shared_ptr<D3B3PerSwapchainTerminalContext::State> value,
        uint64_t generation, uint64_t identity) : context(std::move(value)), pairGeneration(generation), lease(identity),
        semaphore(*context->vk) { context->activeLeases.fetch_add(1, std::memory_order_acq_rel); }
    ~State() {
        if (quarantineReservation) {
            if (const auto owner = context->deviceLifetime.lock())
                owner->releaseReservation(context->contextIdentity, lease);
        }
        context->activeLeases.fetch_sub(1, std::memory_order_acq_rel);
    }
    std::shared_ptr<D3B3PerSwapchainTerminalContext::State> context;
    uint64_t pairGeneration;
    uint64_t lease;
    bool waitAuthorityIssued{};
    bool quarantineReservation{};
    vk::Semaphore semaphore;
    enum class Phase : uint8_t { RESERVED, SIGNAL_SUBMITTED, WAIT_SUBMITTED, WAIT_RETIRED } phase{Phase::RESERVED};
};

struct ReturnedForGraphicsWaitAuthority::State {
    std::shared_ptr<D3B3AReturnHandoffLease::State> backing;
    bool terminalSubmitAttempted{};
};

D3B3PerSwapchainTerminalContext::D3B3PerSwapchainTerminalContext(
        const vk::Vulkan& value, uint32_t family, uint64_t generation) {
    standaloneDeviceLifetime =
        std::make_shared<D3B3DeviceLifetimeQuarantine>();
    state = std::make_shared<State>(
        value, family, generation, standaloneDeviceLifetime);
}
D3B3PerSwapchainTerminalContext::D3B3PerSwapchainTerminalContext(
        const vk::Vulkan& value, uint32_t family, uint64_t generation,
        std::shared_ptr<D3B3DeviceLifetimeQuarantine> owner) :
    state(std::make_shared<State>(value, family, generation, std::move(owner))) {}
D3B3PerSwapchainTerminalContext::~D3B3PerSwapchainTerminalContext() = default;
uint32_t D3B3PerSwapchainTerminalContext::destinationQueueFamily() const noexcept { return state->destinationFamily; }
uint64_t D3B3PerSwapchainTerminalContext::lifecycleGeneration() const noexcept { return state->lifecycle; }
uint64_t D3B3PerSwapchainTerminalContext::contextIdentity() const noexcept {
    return state ? state->contextIdentity : 0;
}
uint64_t D3B3PerSwapchainTerminalContext::deviceIdentity() const noexcept {
    if (!state) return 0;
    const auto owner = state->deviceLifetime.lock();
    return owner ? owner->identity() : 0;
}
bool D3B3PerSwapchainTerminalContext::retirementReady() const noexcept {
    const auto owner = state ? state->deviceLifetime.lock() : nullptr;
    return state && (!owner || !owner->devicePoisoned())
        && !state->deviceLost.load(std::memory_order_acquire)
        && state->activeLeases.load(std::memory_order_acquire) == 0;
}
bool D3B3PerSwapchainTerminalContext::deviceLostQuarantined() const noexcept {
    return state && state->deviceLost.load(std::memory_order_acquire);
}
void D3B3PerSwapchainTerminalContext::quarantineDeviceLost() noexcept {
    if (!state) return;
    const auto deviceLifetime = state->deviceLifetime.lock();
    if (!deviceLifetime) return;
    deviceLifetime->poison();
    {
        const std::scoped_lock lock(state->acceptedMutex);
        for (const auto& weak : state->acceptedBackings) {
            if (auto backing = weak.lock()) {
                const auto lease =
                    std::static_pointer_cast<D3B3AReturnHandoffLease::State>(backing);
                if (lease->phase == D3B3AReturnHandoffLease::State::Phase::RESERVED)
                    continue;
                if (!deviceLifetime->retain(
                        state->contextIdentity, lease->lease, backing)) {
                    // A failed ownership transfer is observable as a poisoned
                    // device lifetime. Keep the accepted backing reachable
                    // through its existing lease/context graph; never report
                    // this device reusable.
                    state->deviceLost.store(true, std::memory_order_release);
                }
            }
        }
    }
    state->deviceLost.store(true, std::memory_order_release);
}

D3B3AReturnHandoffLease::D3B3AReturnHandoffLease(std::shared_ptr<State> value) noexcept : state(std::move(value)) {}
bool D3B3AReturnHandoffLease::valid() const noexcept { return state != nullptr; }
uint64_t D3B3AReturnHandoffLease::generation() const noexcept { return state ? state->pairGeneration : 0; }
uint64_t D3B3AReturnHandoffLease::leaseIdentity() const noexcept { return state ? state->lease : 0; }
uint64_t D3B3AReturnHandoffLease::contextIdentity() const noexcept { return state ? state->context->contextIdentity : 0; }
TerminalSubmitProvenance D3B3AReturnHandoffLease::provenance(TerminalSubmitRole role) const {
    if (!state) throw std::logic_error("invalid handoff lease provenance");
    return {role, state->context->contextIdentity, state->lease, state->semaphore.handle()};
}
ReturnedForGraphicsWaitAuthority D3B3AReturnHandoffLease::deriveWaitAuthority() {
    if (!state || state->phase != State::Phase::SIGNAL_SUBMITTED || state->waitAuthorityIssued)
        throw std::logic_error("wait authority requires accepted producer signal");
    state->waitAuthorityIssued = true;
    return ReturnedForGraphicsWaitAuthority(std::make_shared<ReturnedForGraphicsWaitAuthority::State>(
        ReturnedForGraphicsWaitAuthority::State{state}));
}
VkSemaphore D3B3AReturnHandoffLease::signalSemaphore() const { if (!state) throw std::logic_error("invalid handoff lease"); return state->semaphore.handle(); }
vk::RuntimeForeignImageHandoffInfo D3B3AReturnHandoffLease::handoffInfo() const {
    if (!state) throw std::logic_error("invalid handoff lease");
    return {state->context->destinationFamily, state->semaphore.handle()};
}
void D3B3AReturnHandoffLease::signalSubmitted() {
    if (!state || state->phase != State::Phase::RESERVED)
        throw std::logic_error("invalid A-return lease transition");
    // Tracking storage is reserved by acquire() before any QueueSubmit can be
    // accepted.  This transition is therefore allocation-free on the critical
    // post-submit path.
    state->phase = State::Phase::SIGNAL_SUBMITTED;
}
void D3B3AReturnHandoffLease::terminalWaitSubmitted() { if (!state || state->phase != State::Phase::SIGNAL_SUBMITTED) throw std::logic_error("invalid terminal lease transition"); state->phase = State::Phase::WAIT_SUBMITTED; }
void D3B3AReturnHandoffLease::terminalWaitRetired() { if (!state || state->phase != State::Phase::WAIT_SUBMITTED) throw std::logic_error("invalid terminal retirement transition"); state->phase = State::Phase::WAIT_RETIRED; }

D3B3AReturnHandoffBinding::D3B3AReturnHandoffBinding(std::shared_ptr<D3B3PerSwapchainTerminalContext> value) : context(std::move(value)) {
    if (!context || !context->state) throw std::invalid_argument("missing D3B3 terminal context");
}
D3B3AReturnHandoffLease D3B3AReturnHandoffBinding::acquire(uint64_t pairGeneration) {
    if (pairGeneration == 0) throw std::invalid_argument("invalid D3B3 pair generation");
    const auto identity = context->state->nextLease.fetch_add(1, std::memory_order_relaxed);
    auto backing = std::make_shared<D3B3AReturnHandoffLease::State>(
        context->state, pairGeneration, identity);

    const auto deviceLifetime = context->state->deviceLifetime.lock();
    if (!deviceLifetime || !deviceLifetime->reserve(
            context->state->contextIdentity, identity))
        throw std::runtime_error("D3B3 device quarantine reservation exhausted");
    backing->quarantineReservation = true;

    // Reserve a fixed tracking slot before the lease can participate in an
    // accepted submit.  No allocation is permitted after QueueSubmit success,
    // because device-loss quarantine must always be able to discover accepted
    // backing while an authority is alive.
    bool tracked{};
    {
        const std::scoped_lock lock(context->state->acceptedMutex);
        for (auto& slot : context->state->acceptedBackings) {
            if (!slot.expired()) continue;
            slot = backing;
            tracked = true;
            break;
        }
    }
    if (!tracked)
        throw std::runtime_error("D3B3 terminal accepted-backing tracking exhausted");

    return D3B3AReturnHandoffLease(std::move(backing));
}
bool D3B3AReturnHandoffBinding::retirementReady() const noexcept { return context->retirementReady(); }

void submitReturnedForGraphicsTerminalWait(
        ReturnedForGraphicsWaitAuthority& authority, TerminalSubmitRole role,
        uint64_t expectedContextIdentity,
        const vk::CommandBuffer& command, const vk::Vulkan& vk, VkQueue queue,
        std::vector<VkSemaphore> waits, std::vector<VkSemaphore> signals,
        VkFence fence, VkPipelineStageFlags waitStage,
        const vk::SubmitObserver& observer,
        const vk::SubmitResultObserver& resultObserver) {
    if (role != TerminalSubmitRole::D3B1TerminalConsumption
            && role != TerminalSubmitRole::D3B2TerminalWait)
        throw std::invalid_argument("invalid terminal submit role");
    if (expectedContextIdentity == 0
            || authority.contextIdentity() != expectedContextIdentity)
        throw std::invalid_argument("terminal wait authority context mismatch");
    const auto semaphore = authority.semaphore();
    if (std::count(waits.begin(), waits.end(), semaphore) != 1)
        throw std::invalid_argument("terminal wait does not uniquely consume its authority");
    if (authority.state->terminalSubmitAttempted)
        throw std::logic_error("terminal wait submit already attempted");
    authority.state->terminalSubmitAttempted = true;
    command.submit(vk, queue, std::move(waits), VK_NULL_HANDLE, 0,
        std::move(signals), VK_NULL_HANDLE, 0, fence, waitStage,
        observer, resultObserver);
    authority.terminalWaitSubmitted();
}

ReturnedForGraphicsWaitAuthority::ReturnedForGraphicsWaitAuthority(std::shared_ptr<State> value) noexcept
    : state(std::move(value)) {}
bool ReturnedForGraphicsWaitAuthority::valid() const noexcept { return state && state->backing; }
VkSemaphore ReturnedForGraphicsWaitAuthority::semaphore() const {
    if (!valid()) throw std::logic_error("invalid terminal wait authority");
    return state->backing->semaphore.handle();
}
uint64_t ReturnedForGraphicsWaitAuthority::contextIdentity() const noexcept {
    return valid() ? state->backing->context->contextIdentity : 0;
}
uint64_t ReturnedForGraphicsWaitAuthority::leaseIdentity() const noexcept {
    return valid() ? state->backing->lease : 0;
}
TerminalSubmitProvenance ReturnedForGraphicsWaitAuthority::provenance(TerminalSubmitRole role) const {
    if (!valid()) throw std::logic_error("invalid terminal wait provenance");
    return {role, state->backing->context->contextIdentity, state->backing->lease,
        state->backing->semaphore.handle()};
}
void ReturnedForGraphicsWaitAuthority::terminalWaitSubmitted() {
    if (!valid() || state->backing->phase != D3B3AReturnHandoffLease::State::Phase::SIGNAL_SUBMITTED)
        throw std::logic_error("invalid terminal wait transition");
    state->backing->phase = D3B3AReturnHandoffLease::State::Phase::WAIT_SUBMITTED;
}
void ReturnedForGraphicsWaitAuthority::terminalWaitRetired() {
    if (!valid() || state->backing->phase != D3B3AReturnHandoffLease::State::Phase::WAIT_SUBMITTED)
        throw std::logic_error("invalid terminal retirement transition");
    state->backing->phase = D3B3AReturnHandoffLease::State::Phase::WAIT_RETIRED;
}

}
