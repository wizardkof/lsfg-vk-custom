/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "adaptive_1x_recovery_authority.hpp"
#include "borrowed_present_fence_registry.hpp"
#include "presented_physical_image_lease.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace lsfgvk::layer {

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
#define LSFGVK_ADAPTIVE_PREPARED_VISIBILITY __attribute__((visibility("default")))
#else
#define LSFGVK_ADAPTIVE_PREPARED_VISIBILITY
#endif
class LSFGVK_ADAPTIVE_PREPARED_VISIBILITY Adaptive1xPreparedLogicalFinal final {
public:
    enum class State : uint8_t {
        Ready, LogicalPresentCalled, Finalized, Abandoned, Retained
    };
    struct Operations final {
        std::function<bool(const PresentedPhysicalImageIdentity&)>
            armInternalPresentFence;
        std::function<VkResult(int*)> exportProducerCompletion;
        std::function<VkResult(int,
            std::vector<PresentedPhysicalImageAcquireToken>&&)> publishCompletion;
        std::function<bool(std::shared_ptr<void>)> retainConservatively;
    };

    Adaptive1xPreparedLogicalFinal(
        VkSwapchainKHR physicalSwapchain, uint32_t physicalImageIndex,
        VkSemaphore presentReadySemaphore, VkFence applicationPresentFence,
        VkFence internalPresentFence, PresentedPhysicalImageIdentity,
        Adaptive1xRecoveryAuthority,
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry>,
        PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease,
        std::shared_ptr<BorrowedPresentFenceRegistry>,
        std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>,
        std::vector<PresentedPhysicalImageAcquireToken>,
        std::shared_ptr<void> strongBacking, Operations) noexcept;
    Adaptive1xPreparedLogicalFinal(const Adaptive1xPreparedLogicalFinal&) = delete;
    Adaptive1xPreparedLogicalFinal& operator=(const Adaptive1xPreparedLogicalFinal&) = delete;
    Adaptive1xPreparedLogicalFinal(Adaptive1xPreparedLogicalFinal&&) noexcept = default;
    Adaptive1xPreparedLogicalFinal& operator=(Adaptive1xPreparedLogicalFinal&&) noexcept = default;
    ~Adaptive1xPreparedLogicalFinal() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] State state() const noexcept { return stateValue; }
    [[nodiscard]] VkSwapchainKHR physicalSwapchain() const noexcept {
        return physicalSwapchainValue;
    }
    [[nodiscard]] uint32_t physicalImageIndex() const noexcept { return imageIndex; }
    [[nodiscard]] VkSemaphore presentReadySemaphore() const noexcept {
        return readySemaphore;
    }
    [[nodiscard]] bool requiresInternalPresentFence() const noexcept {
        return applicationFence == VK_NULL_HANDLE;
    }
    [[nodiscard]] VkFence internalPresentFence() const noexcept {
        return internalFence;
    }
    [[nodiscard]] VkFence applicationPresentFence() const noexcept {
        return applicationFence;
    }
    [[nodiscard]] VkFence producerFence() const noexcept {
        return recovery.producerFence();
    }
    [[nodiscard]] VkFence acquireFence() const noexcept {
        return recovery.acquireFence();
    }
    [[nodiscard]] const PresentedPhysicalImageIdentity& presentedIdentity()
            const noexcept {
        return presented;
    }
    [[nodiscard]] uint64_t generation() const noexcept {
        return recovery.generation();
    }

    [[nodiscard]] bool markLogicalPresentCalled() noexcept;
    [[nodiscard]] VkResult finalizeLogicalPresent(VkResult) noexcept;
    [[nodiscard]] VkResult abandonWithoutLogicalPresent() noexcept;

private:
    void abortReservations() noexcept;
    [[nodiscard]] VkResult retainBacking(VkResult) noexcept;
    [[nodiscard]] bool consumeReacquires() noexcept;
    [[nodiscard]] bool retireReacquires() noexcept;

    VkSwapchainKHR physicalSwapchainValue{};
    uint32_t imageIndex{};
    VkSemaphore readySemaphore{};
    VkFence applicationFence{};
    VkFence internalFence{};
    PresentedPhysicalImageIdentity presented;
    Adaptive1xRecoveryAuthority recovery;
    std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases;
    PresentedPhysicalImageLeaseRegistry::PreparedPresentedLease preparedLease;
    std::shared_ptr<BorrowedPresentFenceRegistry> borrowedFences;
    std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
        preparedBorrowedFence;
    std::vector<PresentedPhysicalImageAcquireToken> consumedAcquires;
    std::shared_ptr<void> backing;
    Operations ops;
    State stateValue{State::Ready};
};
#undef LSFGVK_ADAPTIVE_PREPARED_VISIBILITY

} // namespace lsfgvk::layer
