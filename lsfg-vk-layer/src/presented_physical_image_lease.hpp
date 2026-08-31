/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace test { void failNextPresentedPhysicalLeaseReserve() noexcept; }
#endif

struct PresentedPhysicalImageIdentity {
    uint64_t deviceLifetimeIdentity{};
    uint64_t swapchainLifecycleIdentity{};
    uintptr_t physicalSwapchainIdentity{};
    uint32_t physicalImageIndex{};
    uint64_t presentOperationIdentity{};

    [[nodiscard]] bool valid() const noexcept;
};

struct PresentedPhysicalImageAcquireToken {
    PresentedPhysicalImageIdentity presented;
    uint64_t acquireOperationIdentity{};
    [[nodiscard]] bool valid() const noexcept {
        return presented.valid() && acquireOperationIdentity != 0;
    }
};

/// Owns only layer resources referenced by WSI. In the no-internal-fence case,
/// a lease becomes retireable only after the same physical image is reacquired,
/// that acquire synchronization is consumed by real GPU work, and the work's
/// safe completion event is observed.
class PresentedPhysicalImageLeaseRegistry final {
public:
    enum class PrepareFailure : uint8_t { None, OutOfHostMemory, InvalidOrConflict };
    enum class PreparationState : uint8_t { Missing, Reserved, BoundUncommitted, Committed };
    struct PreparedPresentedLease {
        PresentedPhysicalImageIdentity identity;
        [[nodiscard]] bool valid() const noexcept { return identity.valid(); }
    };
    struct ReservedPresentedLease {
        uint64_t presentOperationIdentity{};
        [[nodiscard]] bool valid() const noexcept {
            return presentOperationIdentity != 0;
        }
    };

    [[nodiscard]] std::optional<ReservedPresentedLease> reserve(
        PresentedPhysicalImageIdentity identityWithoutImage,
        std::shared_ptr<void> wsiResources,
        PrepareFailure* failure = nullptr,
        std::function<VkResult()> internalFenceWait = {}) noexcept;
    [[nodiscard]] std::optional<PreparedPresentedLease> bindPhysicalImage(
        ReservedPresentedLease&, PresentedPhysicalImageIdentity,
        PrepareFailure* failure = nullptr) noexcept;
    void abort(ReservedPresentedLease&) noexcept;

    [[nodiscard]] std::optional<PreparedPresentedLease> prepare(
        PresentedPhysicalImageIdentity identity,
        std::shared_ptr<void> wsiResources,
        PrepareFailure* failure = nullptr,
        std::function<VkResult()> internalFenceWait = {}) noexcept;
    [[nodiscard]] bool commit(PreparedPresentedLease&) noexcept;
    void abort(PreparedPresentedLease&) noexcept;
    /// Convenience for callers which have no external side effect between
    /// preparation and publication. Production present paths use prepare/commit.
    [[nodiscard]] bool install(PresentedPhysicalImageIdentity identity,
        std::shared_ptr<void> wsiResources) noexcept;
    [[nodiscard]] std::optional<PresentedPhysicalImageAcquireToken> reacquired(
        uint64_t deviceLifetimeIdentity,
        uint64_t swapchainLifecycleIdentity,
        uintptr_t physicalSwapchainIdentity,
        uint32_t physicalImageIndex,
        uint64_t acquireOperationIdentity) noexcept;
    [[nodiscard]] bool markAcquireConsumed(
        const PresentedPhysicalImageAcquireToken&) noexcept;
    /// Exact-once retirement after an Acquire fence proves the reacquire signal
    /// operation complete but no device submission consumed its semaphore.
    [[nodiscard]] bool retireAfterAcquireCompletion(
        const PresentedPhysicalImageAcquireToken&) noexcept;
    [[nodiscard]] bool retireAfterSafeCompletion(
        const PresentedPhysicalImageAcquireToken&) noexcept;
    /// Alternate exact-once retirement authority, used by a borrowed
    /// application present fence after the application observes its signal.
    [[nodiscard]] bool retireAfterPresentFence(
        const PresentedPhysicalImageIdentity&) noexcept;
    [[nodiscard]] bool contains(const PresentedPhysicalImageIdentity&) const noexcept;
    [[nodiscard]] PreparationState preparationState(
        const PresentedPhysicalImageIdentity&) const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] size_t countForSwapchainLifecycle(uint64_t) const noexcept;
    [[nodiscard]] size_t drainInternalFencesForTeardown() noexcept;
    [[nodiscard]] size_t releasePreparedProducerBackingsAfterDeviceIdle() noexcept;

private:
    struct PhysicalImageKey {
        uint64_t deviceLifetimeIdentity{};
        uint64_t swapchainLifecycleIdentity{};
        uintptr_t physicalSwapchainIdentity{};
        uint32_t physicalImageIndex{};

        bool operator==(const PhysicalImageKey&) const noexcept = default;
    };
    struct PhysicalImageKeyHash {
        [[nodiscard]] size_t operator()(const PhysicalImageKey&) const noexcept;
    };
    [[nodiscard]] static PhysicalImageKey imageKey(
        const PresentedPhysicalImageIdentity&) noexcept;
    struct Entry {
        PresentedPhysicalImageIdentity identity;
        std::shared_ptr<void> resources;
        uint64_t acquireOperationIdentity{};
        bool acquireConsumed{};
        bool bound{};
        bool committed{};
        std::function<VkResult()> internalFenceWait;
        bool teardownDrainInProgress{};
    };
    std::unordered_map<uint64_t, Entry> entries;
    mutable std::mutex mutex;
};

/// Move-only authority retaining the registry and the exact committed physical
/// image generation used by one logical present.  Retirement is delegated to
/// the registry, which supplies generation validation and exact-once removal.
class PresentedPhysicalImageRetirementAuthority final {
public:
    PresentedPhysicalImageRetirementAuthority() = default;
    PresentedPhysicalImageRetirementAuthority(
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry>,
        PresentedPhysicalImageIdentity) noexcept;
    PresentedPhysicalImageRetirementAuthority(
        const PresentedPhysicalImageRetirementAuthority&) = delete;
    PresentedPhysicalImageRetirementAuthority& operator=(
        const PresentedPhysicalImageRetirementAuthority&) = delete;
    PresentedPhysicalImageRetirementAuthority(
        PresentedPhysicalImageRetirementAuthority&&) noexcept = default;
    PresentedPhysicalImageRetirementAuthority& operator=(
        PresentedPhysicalImageRetirementAuthority&&) noexcept = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool matches(uint64_t deviceLifetimeIdentity,
        uint64_t swapchainLifecycleIdentity,
        uintptr_t physicalSwapchainIdentity,
        uint32_t physicalImageIndex) const noexcept;
    [[nodiscard]] bool retire() noexcept;
    [[nodiscard]] const PresentedPhysicalImageIdentity& identity() const noexcept {
        return identityValue;
    }

private:
    std::shared_ptr<PresentedPhysicalImageLeaseRegistry> registry;
    PresentedPhysicalImageIdentity identityValue;
    bool retired{};
};

}
