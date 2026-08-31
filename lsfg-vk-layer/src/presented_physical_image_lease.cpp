/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "presented_physical_image_lease.hpp"

#include <atomic>


namespace lsfgvk::layer {
namespace {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
std::atomic_bool failPresentedLeaseReserve{};
#endif
bool samePresentedOperation(const PresentedPhysicalImageIdentity& left,
        const PresentedPhysicalImageIdentity& right) noexcept {
    return left.deviceLifetimeIdentity == right.deviceLifetimeIdentity
        && left.swapchainLifecycleIdentity == right.swapchainLifecycleIdentity
        && left.physicalSwapchainIdentity == right.physicalSwapchainIdentity
        && left.physicalImageIndex == right.physicalImageIndex
        && left.presentOperationIdentity == right.presentOperationIdentity;
}

}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace test {
void failNextPresentedPhysicalLeaseReserve() noexcept {
    failPresentedLeaseReserve.store(true, std::memory_order_release);
}
}
#endif

size_t PresentedPhysicalImageLeaseRegistry::PhysicalImageKeyHash::operator()(
        const PhysicalImageKey& key) const noexcept {
    size_t value = std::hash<uint64_t>{}(key.deviceLifetimeIdentity);
    const auto combine = [&value](size_t next) {
        value ^= next + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    };
    combine(std::hash<uint64_t>{}(key.swapchainLifecycleIdentity));
    combine(std::hash<uintptr_t>{}(key.physicalSwapchainIdentity));
    combine(std::hash<uint32_t>{}(key.physicalImageIndex));
    return value;
}

PresentedPhysicalImageLeaseRegistry::PhysicalImageKey
PresentedPhysicalImageLeaseRegistry::imageKey(
        const PresentedPhysicalImageIdentity& identity) noexcept {
    return {identity.deviceLifetimeIdentity,
        identity.swapchainLifecycleIdentity,
        identity.physicalSwapchainIdentity,
        identity.physicalImageIndex};
}

bool PresentedPhysicalImageIdentity::valid() const noexcept {
    return deviceLifetimeIdentity != 0 && swapchainLifecycleIdentity != 0
        && physicalSwapchainIdentity != 0 && presentOperationIdentity != 0;
}

std::optional<PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease>
PresentedPhysicalImageLeaseRegistry::prepare(
        PresentedPhysicalImageIdentity identity,
        std::shared_ptr<void> wsiResources, PrepareFailure* failure,
        std::function<VkResult()> internalFenceWait) noexcept {
    auto reserved = reserve(identity, std::move(wsiResources), failure,
        std::move(internalFenceWait));
    if (!reserved) return std::nullopt;
    auto bound = bindPhysicalImage(*reserved, identity, failure);
    if (!bound) abort(*reserved);
    return bound;
}

std::optional<PresentedPhysicalImageLeaseRegistry::ReservedPresentedLease>
PresentedPhysicalImageLeaseRegistry::reserve(
        PresentedPhysicalImageIdentity identity, std::shared_ptr<void> resources,
        PrepareFailure* failure,
        std::function<VkResult()> internalFenceWait) noexcept {
    const std::scoped_lock lock(mutex);
    if (failure) *failure = PrepareFailure::None;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    if (failPresentedLeaseReserve.exchange(false, std::memory_order_acq_rel)) {
        if (failure) *failure = PrepareFailure::OutOfHostMemory;
        return std::nullopt;
    }
#endif
    if (!identity.valid() || !resources
            || entries.contains(identity.presentOperationIdentity)) {
        if (failure) *failure = PrepareFailure::InvalidOrConflict;
        return std::nullopt;
    }
    try {
        const auto operation = identity.presentOperationIdentity;
        identity.physicalImageIndex = 0;
        if (!entries.emplace(operation, Entry{identity, std::move(resources),
                0, false, false, false, std::move(internalFenceWait), false}).second) {
            if (failure) *failure = PrepareFailure::InvalidOrConflict;
            return std::nullopt;
        }
        return ReservedPresentedLease{operation};
    } catch (...) {
        if (failure) *failure = PrepareFailure::OutOfHostMemory;
        return std::nullopt;
    }
}

std::optional<PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease>
PresentedPhysicalImageLeaseRegistry::bindPhysicalImage(
        ReservedPresentedLease& reserved, PresentedPhysicalImageIdentity identity,
        PrepareFailure* failure) noexcept {
    const std::scoped_lock lock(mutex);
    if (failure) *failure = PrepareFailure::None;
    if (!reserved.valid() || !identity.valid()
            || reserved.presentOperationIdentity != identity.presentOperationIdentity) {
        if (failure) *failure = PrepareFailure::InvalidOrConflict;
        return std::nullopt;
    }
    auto it = entries.find(reserved.presentOperationIdentity);
    if (it == entries.end() || it->second.bound || it->second.committed) {
        if (failure) *failure = PrepareFailure::InvalidOrConflict;
        return std::nullopt;
    }
    const auto key = imageKey(identity);
    for (const auto& [operation, entry] : entries)
        if (operation != reserved.presentOperationIdentity && entry.bound
                && entry.acquireOperationIdentity == 0
                && imageKey(entry.identity) == key) {
            if (failure) *failure = PrepareFailure::InvalidOrConflict;
            return std::nullopt;
        }
    it->second.identity = identity;
    it->second.bound = true;
    reserved = {};
    return PreparedPresentedLease{identity};
}

void PresentedPhysicalImageLeaseRegistry::abort(
        ReservedPresentedLease& reserved) noexcept {
    const std::scoped_lock lock(mutex);
    if (!reserved.valid()) return;
    const auto it = entries.find(reserved.presentOperationIdentity);
    if (it != entries.end() && !it->second.bound && !it->second.committed)
        entries.erase(it);
    reserved = {};
}

bool PresentedPhysicalImageLeaseRegistry::commit(
        PreparedPresentedLease& prepared) noexcept {
    const std::scoped_lock lock(mutex);
    if (!prepared.valid()) return false;
    const auto it = entries.find(prepared.identity.presentOperationIdentity);
    if (it == entries.end() || !it->second.bound || it->second.committed
            || !samePresentedOperation(it->second.identity, prepared.identity))
        return false;
    it->second.committed = true;
    prepared = {};
    return true;
}

void PresentedPhysicalImageLeaseRegistry::abort(
        PreparedPresentedLease& prepared) noexcept {
    const std::scoped_lock lock(mutex);
    if (!prepared.valid()) return;
    const auto it = entries.find(prepared.identity.presentOperationIdentity);
    if (it != entries.end() && !it->second.committed
            && samePresentedOperation(it->second.identity, prepared.identity)) {
        entries.erase(it);
    }
    prepared = {};
}

bool PresentedPhysicalImageLeaseRegistry::install(
        PresentedPhysicalImageIdentity identity,
        std::shared_ptr<void> wsiResources) noexcept {
    auto prepared = prepare(identity, std::move(wsiResources));
    return prepared && commit(*prepared);
}

std::optional<PresentedPhysicalImageAcquireToken>
PresentedPhysicalImageLeaseRegistry::reacquired(
        uint64_t deviceLifetimeIdentity,
        uint64_t swapchainLifecycleIdentity,
        uintptr_t physicalSwapchainIdentity,
        uint32_t physicalImageIndex,
        uint64_t acquireOperationIdentity) noexcept {
    const std::scoped_lock lock(mutex);
    if (acquireOperationIdentity == 0) return std::nullopt;
    const PhysicalImageKey key{deviceLifetimeIdentity,
        swapchainLifecycleIdentity, physicalSwapchainIdentity,
        physicalImageIndex};
    auto iterator = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it)
        if (it->second.bound && it->second.committed
                && it->second.acquireOperationIdentity == 0
                && imageKey(it->second.identity) == key) { iterator = it; break; }
    if (iterator == entries.end()) return std::nullopt;
    auto& entry = iterator->second;
    if (entry.identity.deviceLifetimeIdentity != deviceLifetimeIdentity
            || entry.identity.swapchainLifecycleIdentity != swapchainLifecycleIdentity
            || entry.identity.physicalSwapchainIdentity != physicalSwapchainIdentity
            || entry.acquireOperationIdentity != 0)
        return std::nullopt;
    entry.acquireOperationIdentity = acquireOperationIdentity;
    return PresentedPhysicalImageAcquireToken{entry.identity, acquireOperationIdentity};
}

bool PresentedPhysicalImageLeaseRegistry::markAcquireConsumed(
        const PresentedPhysicalImageAcquireToken& token) noexcept {
    const std::scoped_lock lock(mutex);
    if (!token.valid()) return false;
    const auto iterator = entries.find(token.presented.presentOperationIdentity);
    if (iterator == entries.end() || !iterator->second.committed) return false;
    auto& entry = iterator->second;
    if (!samePresentedOperation(entry.identity, token.presented)
            || entry.acquireOperationIdentity != token.acquireOperationIdentity
            || entry.acquireConsumed)
        return false;
    entry.acquireConsumed = true;
    return true;
}

bool PresentedPhysicalImageLeaseRegistry::retireAfterAcquireCompletion(
        const PresentedPhysicalImageAcquireToken& token) noexcept {
    const std::scoped_lock lock(mutex);
    if (!token.valid()) return false;
    const auto iterator = entries.find(token.presented.presentOperationIdentity);
    if (iterator == entries.end() || !iterator->second.committed) return false;
    const auto& entry = iterator->second;
    if (!samePresentedOperation(entry.identity, token.presented)
            || entry.acquireOperationIdentity != token.acquireOperationIdentity
            || entry.acquireConsumed)
        return false;
    entries.erase(iterator);
    return true;
}

bool PresentedPhysicalImageLeaseRegistry::retireAfterSafeCompletion(
        const PresentedPhysicalImageAcquireToken& token) noexcept {
    const std::scoped_lock lock(mutex);
    if (!token.valid()) return false;
    const auto iterator = entries.find(token.presented.presentOperationIdentity);
    if (iterator == entries.end() || !iterator->second.committed) return false;
    const auto& entry = iterator->second;
    if (!samePresentedOperation(entry.identity, token.presented)
            || entry.acquireOperationIdentity != token.acquireOperationIdentity
            || !entry.acquireConsumed)
        return false;
    entries.erase(iterator);
    return true;
}

bool PresentedPhysicalImageLeaseRegistry::retireAfterPresentFence(
        const PresentedPhysicalImageIdentity& identity) noexcept {
    const std::scoped_lock lock(mutex);
    if (!identity.valid()) return false;
    const auto iterator = entries.find(identity.presentOperationIdentity);
    if (iterator == entries.end() || !iterator->second.committed
            || !samePresentedOperation(iterator->second.identity, identity))
        return false;
    entries.erase(iterator);
    return true;
}

bool PresentedPhysicalImageLeaseRegistry::contains(
        const PresentedPhysicalImageIdentity& identity) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto iterator = entries.find(identity.presentOperationIdentity);
    return iterator != entries.end() && iterator->second.committed
        && samePresentedOperation(iterator->second.identity, identity);
}

PresentedPhysicalImageLeaseRegistry::PreparationState
PresentedPhysicalImageLeaseRegistry::preparationState(
        const PresentedPhysicalImageIdentity& identity) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto iterator = entries.find(identity.presentOperationIdentity);
    if (iterator == entries.end()
            || !samePresentedOperation(iterator->second.identity, identity))
        return PreparationState::Missing;
    if (iterator->second.committed) return PreparationState::Committed;
    if (iterator->second.bound) return PreparationState::BoundUncommitted;
    return PreparationState::Reserved;
}

size_t PresentedPhysicalImageLeaseRegistry::size() const noexcept {
    const std::scoped_lock lock(mutex);
    return entries.size();
}

size_t PresentedPhysicalImageLeaseRegistry::countForSwapchainLifecycle(
        uint64_t lifecycle) const noexcept {
    const std::scoped_lock lock(mutex);
    size_t count{};
    for (const auto& [operation, entry] : entries) {
        (void)operation;
        if (entry.committed && entry.identity.swapchainLifecycleIdentity == lifecycle) ++count;
    }
    return count;
}

size_t PresentedPhysicalImageLeaseRegistry::drainInternalFencesForTeardown() noexcept {
    size_t retired{};
    while (true) {
        PresentedPhysicalImageIdentity identity{};
        std::function<VkResult()> wait;
        {
            const std::scoped_lock lock(mutex);
            auto candidate = entries.end();
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (it->second.committed && it->second.internalFenceWait
                        && !it->second.teardownDrainInProgress) {
                    candidate = it;
                    break;
                }
            }
            if (candidate == entries.end()) break;
            identity = candidate->second.identity;
            candidate->second.teardownDrainInProgress = true;
            wait = std::move(candidate->second.internalFenceWait);
        }
        if (wait() == VK_SUCCESS) {
            if (retireAfterPresentFence(identity)) ++retired;
            continue;
        }
        const std::scoped_lock lock(mutex);
        const auto it = entries.find(identity.presentOperationIdentity);
        if (it != entries.end()
                && samePresentedOperation(it->second.identity, identity)) {
            it->second.internalFenceWait = std::move(wait);
            it->second.teardownDrainInProgress = false;
        }
        break;
    }
    return retired;
}

size_t PresentedPhysicalImageLeaseRegistry::releasePreparedProducerBackingsAfterDeviceIdle()
        noexcept {
    const std::scoped_lock lock(mutex);
    size_t released{};
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->second.committed) {
            ++it;
            continue;
        }
        it = entries.erase(it);
        ++released;
    }
    return released;
}

PresentedPhysicalImageRetirementAuthority::PresentedPhysicalImageRetirementAuthority(
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> registryValue,
        PresentedPhysicalImageIdentity identity) noexcept
    : registry(std::move(registryValue)), identityValue(identity) {}

bool PresentedPhysicalImageRetirementAuthority::valid() const noexcept {
    return !retired && registry && identityValue.valid()
        && registry->contains(identityValue);
}

bool PresentedPhysicalImageRetirementAuthority::matches(
        uint64_t deviceLifetimeIdentity,
        uint64_t swapchainLifecycleIdentity,
        uintptr_t physicalSwapchainIdentity,
        uint32_t physicalImageIndex) const noexcept {
    return valid()
        && identityValue.deviceLifetimeIdentity == deviceLifetimeIdentity
        && identityValue.swapchainLifecycleIdentity == swapchainLifecycleIdentity
        && identityValue.physicalSwapchainIdentity == physicalSwapchainIdentity
        && identityValue.physicalImageIndex == physicalImageIndex;
}

bool PresentedPhysicalImageRetirementAuthority::retire() noexcept {
    if (!valid()) return false;
    if (!registry->retireAfterPresentFence(identityValue)) return false;
    retired = true;
    return true;
}

}
