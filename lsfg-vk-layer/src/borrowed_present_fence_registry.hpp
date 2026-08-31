/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "presented_physical_image_lease.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <list>
#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace test { void failNextBorrowedPresentFenceReserve() noexcept; }
#endif

struct BorrowedFenceLifecycleIdentity {
    uint64_t deviceLifetimeIdentity{};
    uint64_t fenceObjectId{};
    uint64_t fenceLifecycleEpoch{};
    VkFence fence{VK_NULL_HANDLE};
    [[nodiscard]] bool valid() const noexcept {
        return deviceLifetimeIdentity != 0 && fenceObjectId != 0
            && fenceLifecycleEpoch != 0
            && fence != VK_NULL_HANDLE;
    }
};

/// Observes application-owned fences without changing their payload or API
/// semantics. The registry owns association metadata and layer WSI leases only.
class BorrowedPresentFenceRegistry final {
public:
    enum class PrepareFailure : uint8_t { None, OutOfHostMemory, InvalidOrUnknown };
    enum class PreparationState : uint8_t {
        Missing, ReservedUnbound, BoundUncommitted, Committed
    };
    struct ReservedBorrowedPresentFence {
        VkFence fence{VK_NULL_HANDLE};
        uint64_t fenceObjectId{};
        uint64_t fenceLifecycleEpoch{};
        uint64_t presentGeneration{};
        [[nodiscard]] bool valid() const noexcept {
            return fence != VK_NULL_HANDLE && fenceObjectId != 0
                && fenceLifecycleEpoch != 0 && presentGeneration != 0;
        }
    };
    struct PreparedBorrowedPresentFence {
        VkFence fence{VK_NULL_HANDLE};
        uint64_t fenceObjectId{};
        uint64_t fenceLifecycleEpoch{};
        uint64_t presentGeneration{};
        [[nodiscard]] bool valid() const noexcept {
            return fence != VK_NULL_HANDLE && fenceObjectId != 0
                && fenceLifecycleEpoch != 0 && presentGeneration != 0;
        }
    };
    explicit BorrowedPresentFenceRegistry(uint64_t deviceLifetimeIdentity = 0) noexcept;

    [[nodiscard]] BorrowedFenceLifecycleIdentity created(VkFence) noexcept;
    [[nodiscard]] BorrowedFenceLifecycleIdentity identity(VkFence) const noexcept;
    [[nodiscard]] std::optional<ReservedBorrowedPresentFence> reserve(
        VkFence, std::shared_ptr<PresentedPhysicalImageLeaseRegistry>,
        PrepareFailure* failure = nullptr) noexcept;
    [[nodiscard]] std::optional<PreparedBorrowedPresentFence>
        bindPresentedIdentity(ReservedBorrowedPresentFence&,
            PresentedPhysicalImageIdentity,
            PrepareFailure* failure = nullptr) noexcept;
    void abort(ReservedBorrowedPresentFence&) noexcept;
    [[nodiscard]] std::optional<PreparedBorrowedPresentFence> prepare(
        VkFence,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry>,
        PresentedPhysicalImageIdentity,
        PrepareFailure* failure = nullptr) noexcept;
    [[nodiscard]] size_t commit(PreparedBorrowedPresentFence&) noexcept;
    void abort(PreparedBorrowedPresentFence&) noexcept;
    [[nodiscard]] bool associate(VkFence,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry>,
        PresentedPhysicalImageIdentity) noexcept;
    [[nodiscard]] size_t retireObserved(VkFence) noexcept;
    [[nodiscard]] size_t resetBoundary(VkFence) noexcept;
    [[nodiscard]] size_t importBoundary(VkFence) noexcept;
    [[nodiscard]] size_t destroyBoundary(VkFence) noexcept;
    [[nodiscard]] bool contains(VkFence) const noexcept;
    [[nodiscard]] size_t associationCount() const noexcept;
    [[nodiscard]] PreparationState preparationState(
        const PreparedBorrowedPresentFence&) const noexcept;
    [[nodiscard]] PreparationState preparationState(
        VkFence, const PresentedPhysicalImageIdentity&) const noexcept;
    /// Device-destroy-only observation boundary. This never resets or destroys
    /// application fences; it only observes, then waits when still pending.
    template<class Status, class Wait>
    [[nodiscard]] size_t drainForTeardown(Status&& getStatus, Wait&& wait) noexcept {
        size_t retired{};
        while (const auto fence = beginTeardownDrain()) {
            auto result = getStatus(*fence);
            if (result == VK_NOT_READY) result = wait(*fence);
            if (result == VK_SUCCESS) {
                retired += retireObserved(*fence);
                continue;
            }
            cancelTeardownDrain(*fence);
            break;
        }
        return retired;
    }

private:
    struct Association {
        enum class State : uint8_t {
            ReservedUnbound, BoundUncommitted, Committed
        };
        uint64_t fenceObjectId{};
        uint64_t fenceLifecycleEpoch{};
        uint64_t presentGeneration{};
        std::weak_ptr<PresentedPhysicalImageLeaseRegistry> leases;
        PresentedPhysicalImageIdentity presented;
        State state{State::ReservedUnbound};
        bool completionObserved{};
    };
    struct FenceGeneration {
        BorrowedFenceLifecycleIdentity identity;
        uint64_t nextPresentGeneration{1};
        std::list<Association> associations;
        bool teardownDrainInProgress{};
    };
    [[nodiscard]] size_t observeLocked(FenceGeneration&) noexcept;
    [[nodiscard]] size_t retireCommittedLocked(
        FenceGeneration&, std::optional<uint64_t> epoch) noexcept;
    [[nodiscard]] std::optional<VkFence> beginTeardownDrain() noexcept;
    void cancelTeardownDrain(VkFence) noexcept;

    uint64_t deviceLifetimeIdentity{};
    uint64_t nextFenceObjectId{1};
    std::unordered_map<VkFence, FenceGeneration> fences;
    mutable std::mutex mutex;
};

template<class Downstream>
VkResult observeBorrowedGetFenceStatus(BorrowedPresentFenceRegistry& registry,
        VkFence fence, Downstream&& downstream) {
    const auto result = downstream();
    if (result == VK_SUCCESS) static_cast<void>(registry.retireObserved(fence));
    return result;
}

template<class Downstream, class Status>
VkResult observeBorrowedWaitForFences(BorrowedPresentFenceRegistry& registry,
        uint32_t count, const VkFence* fences, VkBool32 waitAll,
        Downstream&& downstream, Status&& status) {
    const auto result = downstream();
    if (fences && result == VK_SUCCESS && waitAll) {
        for (uint32_t i = 0; i < count; ++i)
            static_cast<void>(registry.retireObserved(fences[i]));
    } else if (fences && (result == VK_SUCCESS || result == VK_TIMEOUT)) {
        for (uint32_t i = 0; i < count; ++i)
            if (status(fences[i]) == VK_SUCCESS)
                static_cast<void>(registry.retireObserved(fences[i]));
    }
    return result;
}

template<class Downstream>
VkResult observeBorrowedWaitForFences(BorrowedPresentFenceRegistry& registry,
        uint32_t count, const VkFence* fences, VkBool32 waitAll,
        Downstream&& downstream) {
    return observeBorrowedWaitForFences(registry, count, fences, waitAll,
        std::forward<Downstream>(downstream), [](VkFence) { return VK_NOT_READY; });
}

template<class Downstream, class Status>
VkResult observeBorrowedResetFences(BorrowedPresentFenceRegistry& registry,
        uint32_t count, const VkFence* fences, Downstream&& downstream,
        Status&& status) {
    if (fences)
        for (uint32_t i = 0; i < count; ++i)
            if (status(fences[i]) == VK_SUCCESS)
                static_cast<void>(registry.retireObserved(fences[i]));
    const auto result = downstream();
    if (result == VK_SUCCESS && fences)
        for (uint32_t i = 0; i < count; ++i)
            static_cast<void>(registry.resetBoundary(fences[i]));
    return result;
}

template<class Downstream>
VkResult observeBorrowedResetFences(BorrowedPresentFenceRegistry& registry,
        uint32_t count, const VkFence* fences, Downstream&& downstream) {
    return observeBorrowedResetFences(registry, count, fences,
        std::forward<Downstream>(downstream), [](VkFence) { return VK_NOT_READY; });
}

template<class Downstream>
void observeBorrowedDestroyFence(BorrowedPresentFenceRegistry& registry,
        VkFence fence, Downstream&& downstream) {
    static_cast<void>(registry.destroyBoundary(fence));
    downstream();
}

}
