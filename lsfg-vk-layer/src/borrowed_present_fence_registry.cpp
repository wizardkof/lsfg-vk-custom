/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "borrowed_present_fence_registry.hpp"

#include <algorithm>
#include <atomic>

namespace lsfgvk::layer {
namespace {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
std::atomic_bool failBorrowedFenceReserve{};
#endif
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace test {
void failNextBorrowedPresentFenceReserve() noexcept {
    failBorrowedFenceReserve.store(true, std::memory_order_release);
}
}
#endif

BorrowedPresentFenceRegistry::BorrowedPresentFenceRegistry(
        uint64_t identity) noexcept : deviceLifetimeIdentity(identity) {}

BorrowedFenceLifecycleIdentity BorrowedPresentFenceRegistry::created(
        VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    if (!deviceLifetimeIdentity || fence == VK_NULL_HANDLE || fences.contains(fence))
        return {};
    BorrowedFenceLifecycleIdentity identity{
        deviceLifetimeIdentity, nextFenceObjectId++, 1, fence};
    try {
        if (!fences.emplace(fence, FenceGeneration{identity, 1, {}, false}).second)
            return {};
    } catch (...) { return {}; }
    return identity;
}

BorrowedFenceLifecycleIdentity BorrowedPresentFenceRegistry::identity(
        VkFence fence) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = fences.find(fence);
    return it == fences.end() ? BorrowedFenceLifecycleIdentity{}
                              : it->second.identity;
}

std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
BorrowedPresentFenceRegistry::prepare(VkFence fence,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases,
        PresentedPhysicalImageIdentity presented, PrepareFailure* failure) noexcept {
    auto reserved = reserve(fence, std::move(leases), failure);
    if (!reserved) return std::nullopt;
    auto prepared = bindPresentedIdentity(*reserved, presented, failure);
    if (!prepared) abort(*reserved);
    return prepared;
}

std::optional<BorrowedPresentFenceRegistry::ReservedBorrowedPresentFence>
BorrowedPresentFenceRegistry::reserve(VkFence fence,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases,
        PrepareFailure* failure) noexcept {
    const std::scoped_lock lock(mutex);
    if (failure) *failure = PrepareFailure::None;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    if (failBorrowedFenceReserve.exchange(false, std::memory_order_acq_rel)) {
        if (failure) *failure = PrepareFailure::OutOfHostMemory;
        return std::nullopt;
    }
#endif
    const auto it = fences.find(fence);
    if (it == fences.end() || !leases)
    {
        if (failure) *failure = PrepareFailure::InvalidOrUnknown;
        return std::nullopt;
    }
    try {
        const auto generation = it->second.nextPresentGeneration++;
        it->second.associations.push_back({
            it->second.identity.fenceObjectId,
            it->second.identity.fenceLifecycleEpoch,
            generation, std::move(leases), {},
            Association::State::ReservedUnbound, false});
        return ReservedBorrowedPresentFence{fence,
            it->second.identity.fenceObjectId,
            it->second.identity.fenceLifecycleEpoch, generation};
    } catch (...) {
        if (failure) *failure = PrepareFailure::OutOfHostMemory;
        return std::nullopt;
    }
}

std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
BorrowedPresentFenceRegistry::bindPresentedIdentity(
        ReservedBorrowedPresentFence& reserved,
        PresentedPhysicalImageIdentity presented, PrepareFailure* failure) noexcept {
    const std::scoped_lock lock(mutex);
    if (failure) *failure = PrepareFailure::None;
    if (!reserved.valid() || !presented.valid()
            || presented.deviceLifetimeIdentity != deviceLifetimeIdentity) {
        if (failure) *failure = PrepareFailure::InvalidOrUnknown;
        return std::nullopt;
    }
    const auto fence = fences.find(reserved.fence);
    if (fence == fences.end()) return std::nullopt;
    for (auto& association : fence->second.associations) {
        if (association.fenceObjectId != reserved.fenceObjectId
                || association.fenceLifecycleEpoch != reserved.fenceLifecycleEpoch
                || association.presentGeneration != reserved.presentGeneration)
            continue;
        if (association.state != Association::State::ReservedUnbound)
            return std::nullopt;
        association.presented = presented;
        association.state = Association::State::BoundUncommitted;
        const PreparedBorrowedPresentFence prepared{reserved.fence,
            reserved.fenceObjectId, reserved.fenceLifecycleEpoch,
            reserved.presentGeneration};
        reserved = {};
        return prepared;
    }
    return std::nullopt;
}

void BorrowedPresentFenceRegistry::abort(
        ReservedBorrowedPresentFence& reserved) noexcept {
    const std::scoped_lock lock(mutex);
    if (!reserved.valid()) return;
    const auto fence = fences.find(reserved.fence);
    if (fence != fences.end())
        for (auto it = fence->second.associations.begin();
                it != fence->second.associations.end(); ++it)
            if (it->fenceObjectId == reserved.fenceObjectId
                    && it->fenceLifecycleEpoch == reserved.fenceLifecycleEpoch
                    && it->presentGeneration == reserved.presentGeneration
                    && it->state == Association::State::ReservedUnbound) {
                fence->second.associations.erase(it);
                break;
            }
    reserved = {};
}

size_t BorrowedPresentFenceRegistry::commit(
        PreparedBorrowedPresentFence& prepared) noexcept {
    const std::scoped_lock lock(mutex);
    if (!prepared.valid()) return 0;
    const auto fence = fences.find(prepared.fence);
    if (fence == fences.end()) return 0;
    for (auto it = fence->second.associations.begin();
            it != fence->second.associations.end(); ++it) {
        if (it->fenceObjectId != prepared.fenceObjectId
                || it->fenceLifecycleEpoch != prepared.fenceLifecycleEpoch
                || it->presentGeneration != prepared.presentGeneration)
            continue;
        if (it->state != Association::State::BoundUncommitted) return 0;
        it->state = Association::State::Committed;
        prepared = {};
        if (!it->completionObserved) return 0;
        size_t retired{};
        if (const auto leases = it->leases.lock(); leases
                && leases->retireAfterPresentFence(it->presented))
            retired = 1;
        fence->second.associations.erase(it);
        return retired;
    }
    return 0;
}

void BorrowedPresentFenceRegistry::abort(
        PreparedBorrowedPresentFence& prepared) noexcept {
    const std::scoped_lock lock(mutex);
    if (!prepared.valid()) return;
    const auto fence = fences.find(prepared.fence);
    if (fence != fences.end()) {
        for (auto it = fence->second.associations.begin();
                it != fence->second.associations.end(); ++it) {
            if (it->fenceObjectId == prepared.fenceObjectId
                    && it->fenceLifecycleEpoch == prepared.fenceLifecycleEpoch
                    && it->presentGeneration == prepared.presentGeneration
                    && it->state == Association::State::BoundUncommitted) {
                fence->second.associations.erase(it);
                break;
            }
        }
    }
    prepared = {};
}

bool BorrowedPresentFenceRegistry::associate(VkFence fence,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases,
        PresentedPhysicalImageIdentity presented) noexcept {
    auto prepared = prepare(fence, std::move(leases), presented);
    if (!prepared) return false;
    static_cast<void>(commit(*prepared));
    return !prepared->valid();
}

size_t BorrowedPresentFenceRegistry::observeLocked(
        FenceGeneration& generation) noexcept {
    size_t retired{};
    for (auto it = generation.associations.begin();
            it != generation.associations.end();) {
        if (it->fenceObjectId != generation.identity.fenceObjectId
                || it->fenceLifecycleEpoch
                    != generation.identity.fenceLifecycleEpoch) {
            ++it;
            continue;
        }
        it->completionObserved = true;
        if (it->state != Association::State::Committed) {
            ++it;
            continue;
        }
        if (const auto leases = it->leases.lock(); leases
                && leases->retireAfterPresentFence(it->presented))
            ++retired;
        it = generation.associations.erase(it);
    }
    return retired;
}

size_t BorrowedPresentFenceRegistry::retireObserved(VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = fences.find(fence);
    return it == fences.end() ? 0 : observeLocked(it->second);
}

size_t BorrowedPresentFenceRegistry::resetBoundary(VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = fences.find(fence);
    if (it == fences.end()) return 0;
    const auto oldEpoch = it->second.identity.fenceLifecycleEpoch;
    const auto retired = retireCommittedLocked(it->second, oldEpoch);
    // A prepared-but-uncommitted association at this boundary violates the
    // two-phase protocol. Preserve it and the epoch instead of fabricating
    // completion or making it unreachable.
    for (const auto& association : it->second.associations)
        if (association.fenceLifecycleEpoch == oldEpoch)
            return retired;
    ++it->second.identity.fenceLifecycleEpoch;
    it->second.nextPresentGeneration = 1;
    return retired;
}

size_t BorrowedPresentFenceRegistry::importBoundary(VkFence fence) noexcept {
    return resetBoundary(fence);
}

size_t BorrowedPresentFenceRegistry::destroyBoundary(VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = fences.find(fence);
    if (it == fences.end()) return 0;
    const auto retired = retireCommittedLocked(it->second, std::nullopt);
    // Uncommitted entries are invariant failures, not completed work. Erasing
    // their metadata is safe only because the application object is now dead;
    // their leases deliberately remain retained for the device-level invariant.
    fences.erase(it);
    return retired;
}

bool BorrowedPresentFenceRegistry::contains(VkFence fence) const noexcept {
    const std::scoped_lock lock(mutex);
    return fences.contains(fence);
}

size_t BorrowedPresentFenceRegistry::associationCount() const noexcept {
    const std::scoped_lock lock(mutex);
    size_t count{};
    for (const auto& [fence, generation] : fences) {
        (void)fence;
        count += generation.associations.size();
    }
    return count;
}

BorrowedPresentFenceRegistry::PreparationState
BorrowedPresentFenceRegistry::preparationState(
        const PreparedBorrowedPresentFence& prepared) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto generation = fences.find(prepared.fence);
    if (generation == fences.end()) return PreparationState::Missing;
    const auto& associations = generation->second.associations;
    const auto found = std::find_if(associations.begin(), associations.end(),
        [&](const Association& value) {
            return value.fenceObjectId == prepared.fenceObjectId
                && value.fenceLifecycleEpoch == prepared.fenceLifecycleEpoch
                && value.presentGeneration == prepared.presentGeneration;
        });
    if (found == associations.end()) return PreparationState::Missing;
    switch (found->state) {
    case Association::State::ReservedUnbound:
        return PreparationState::ReservedUnbound;
    case Association::State::BoundUncommitted:
        return PreparationState::BoundUncommitted;
    case Association::State::Committed:
        return PreparationState::Committed;
    }
    return PreparationState::Missing;
}

BorrowedPresentFenceRegistry::PreparationState
BorrowedPresentFenceRegistry::preparationState(VkFence fence,
        const PresentedPhysicalImageIdentity& identity) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto generation = fences.find(fence);
    if (generation == fences.end()) return PreparationState::Missing;
    const auto found = std::find_if(generation->second.associations.begin(),
        generation->second.associations.end(), [&](const Association& value) {
            return value.presented.presentOperationIdentity
                    == identity.presentOperationIdentity
                && value.presented.deviceLifetimeIdentity
                    == identity.deviceLifetimeIdentity
                && value.presented.swapchainLifecycleIdentity
                    == identity.swapchainLifecycleIdentity;
        });
    if (found == generation->second.associations.end())
        return PreparationState::Missing;
    switch (found->state) {
    case Association::State::ReservedUnbound:
        return PreparationState::ReservedUnbound;
    case Association::State::BoundUncommitted:
        return PreparationState::BoundUncommitted;
    case Association::State::Committed:
        return PreparationState::Committed;
    }
    return PreparationState::Missing;
}

size_t BorrowedPresentFenceRegistry::retireCommittedLocked(
        FenceGeneration& generation, std::optional<uint64_t> epoch) noexcept {
    size_t retired{};
    for (auto it = generation.associations.begin();
            it != generation.associations.end();) {
        if ((epoch && it->fenceLifecycleEpoch != *epoch)
                || it->state != Association::State::Committed) {
            ++it;
            continue;
        }
        if (const auto leases = it->leases.lock(); leases
                && leases->retireAfterPresentFence(it->presented))
            ++retired;
        it = generation.associations.erase(it);
    }
    return retired;
}

std::optional<VkFence> BorrowedPresentFenceRegistry::beginTeardownDrain() noexcept {
    const std::scoped_lock lock(mutex);
    for (auto& [fence, generation] : fences) {
        const bool hasCommitted = std::any_of(
            generation.associations.begin(), generation.associations.end(),
            [](const Association& association) {
                return association.state == Association::State::Committed;
            });
        if (hasCommitted && !generation.teardownDrainInProgress) {
            generation.teardownDrainInProgress = true;
            return fence;
        }
    }
    return std::nullopt;
}

void BorrowedPresentFenceRegistry::cancelTeardownDrain(VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = fences.find(fence);
    if (it != fences.end()) it->second.teardownDrainInProgress = false;
}

}
