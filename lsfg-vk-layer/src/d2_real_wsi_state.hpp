/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

enum class D2ApplicationExit : uint8_t { None, Present, Release };
enum class D2HiddenAcquireOwnership : uint8_t {
    None, AcquiredUntouched, UsedByGpu, PresentEnqueued, Released, Terminal
};
enum class D2HiddenAcquireClassification : uint8_t {
    EligibleCarrier, IneligibleCarrier, LocalStateConflict
};
enum class D2RestoreDisposition : uint8_t {
    NotClobbered, Required, InvalidShadowAuthority
};

struct D2PhysicalImageState {
    bool appAcquired{};
    bool firstAppAcquireSeen{};
    bool releasePrepared{};
    bool batchPresentPrepared{};
    bool shadowValid{};
    bool hiddenClobbered{};
    bool hiddenAcquireOwned{};
    bool hiddenReleasePrepared{};
    bool publicAcquirePending{};
    bool carrierEligible{};
    uint64_t applicationGeneration{};
    uint64_t shadowGeneration{};
    D2HiddenAcquireOwnership hiddenOwnership{D2HiddenAcquireOwnership::None};
    D2ApplicationExit lastApplicationExit{D2ApplicationExit::None};
    std::optional<uint32_t> lastApplicationPresentFamily;
};

/// Pass-4A state only. Its storage is fully allocated at swapchain creation;
/// release preparation/commit never inserts or allocates.
class D2RealWsiState final {
public:
    explicit D2RealWsiState(size_t imageCount) : images(imageCount) {}

    void configureCarrierCapability(bool copySupported,
            bool maintenance1Available,
            std::optional<uint32_t> executionQueueFamily = std::nullopt) noexcept {
        const std::scoped_lock lock(mutex);
        carrierCopySupported = copySupported;
        maintenanceAvailable = maintenance1Available;
        carrierExecutionFamily = executionQueueFamily;
        for (auto& image : images)
            image.carrierEligible = carrierEligibleLocked(image);
    }

    void retireForLsfgWork() noexcept {
        const std::scoped_lock lock(mutex);
        retired = true;
        for (auto& image : images) image.carrierEligible = false;
    }
    [[nodiscard]] bool retiredForLsfgWork() const noexcept {
        const std::scoped_lock lock(mutex);
        return retired;
    }

    [[nodiscard]] bool acquired(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || images[index].appAcquired
                || images[index].releasePrepared
                || images[index].batchPresentPrepared
                || images[index].hiddenAcquireOwned
                || images[index].publicAcquirePending)
            return false;
        images[index].appAcquired = true;
        images[index].firstAppAcquireSeen = true;
        images[index].carrierEligible = false;
        return true;
    }

    [[nodiscard]] bool recordPublicAcquireSuccess(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || images[index].appAcquired
                || images[index].releasePrepared
                || images[index].batchPresentPrepared
                || images[index].hiddenAcquireOwned
                || images[index].publicAcquirePending) {
            latchTerminalForLsfgWorkLocked();
            return false;
        }
        images[index].publicAcquirePending = true;
        images[index].carrierEligible = false;
        return true;
    }

    [[nodiscard]] bool commitPublicAcquire(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !images[index].publicAcquirePending)
            return false;
        images[index].publicAcquirePending = false;
        images[index].appAcquired = true;
        images[index].firstAppAcquireSeen = true;
        images[index].carrierEligible = false;
        return true;
    }

    void failPublicAcquire(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index < images.size() && images[index].publicAcquirePending)
            images[index].hiddenOwnership = D2HiddenAcquireOwnership::Terminal;
        latchTerminalForLsfgWorkLocked();
    }

    [[nodiscard]] bool recoverPublicAcquire(uint32_t index,
            bool restored) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !images[index].publicAcquirePending)
            return false;
        auto& image = images[index];
        image.publicAcquirePending = false;
        if (restored) {
            image.hiddenClobbered = false;
            image.hiddenAcquireOwned = false;
            image.hiddenOwnership = D2HiddenAcquireOwnership::Released;
        }
        image.carrierEligible = carrierEligibleLocked(image);
        return true;
    }

    [[nodiscard]] bool prepareRelease(std::span<const uint32_t> indices) noexcept {
        const std::scoped_lock lock(mutex);
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= images.size() || !images[indices[i]].appAcquired
                    || images[indices[i]].releasePrepared
                    || images[indices[i]].batchPresentPrepared)
                return false;
            for (size_t j = 0; j < i; ++j)
                if (indices[j] == indices[i]) return false;
        }
        for (const auto index : indices) images[index].releasePrepared = true;
        return true;
    }

    [[nodiscard]] bool canPresent(uint32_t index) const noexcept {
        const std::scoped_lock lock(mutex);
        return index < images.size() && images[index].appAcquired
            && !images[index].releasePrepared
            && !images[index].batchPresentPrepared;
    }

    [[nodiscard]] std::optional<uint64_t> reserveBatchPresent(
            uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return std::nullopt;
        auto& image = images[index];
        if (!image.appAcquired || image.releasePrepared
                || image.batchPresentPrepared || image.publicAcquirePending)
            return std::nullopt;
        image.batchPresentPrepared = true;
        return image.applicationGeneration;
    }

    [[nodiscard]] bool abortBatchPresent(
            uint32_t index, uint64_t generation) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.batchPresentPrepared
                || image.applicationGeneration != generation)
            return false;
        image.batchPresentPrepared = false;
        return true;
    }

    [[nodiscard]] bool commitBatchPresent(uint32_t index,
            uint64_t generation,
            std::optional<uint32_t> queueFamily = std::nullopt) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.batchPresentPrepared || !image.appAcquired
                || image.applicationGeneration != generation)
            return false;
        image.batchPresentPrepared = false;
        image.appAcquired = false;
        image.releasePrepared = false;
        image.lastApplicationExit = D2ApplicationExit::Present;
        ++image.applicationGeneration;
        image.shadowValid = false;
        image.hiddenClobbered = false;
        image.carrierEligible = false;
        image.lastApplicationPresentFamily = queueFamily;
        image.carrierEligible = carrierEligibleLocked(image);
        return true;
    }

    void finishRelease(std::span<const uint32_t> indices, bool success) noexcept {
        const std::scoped_lock lock(mutex);
        for (const auto index : indices) {
            if (index >= images.size() || !images[index].releasePrepared) continue;
            images[index].releasePrepared = false;
            if (!success) continue;
            images[index].appAcquired = false;
            images[index].lastApplicationExit = D2ApplicationExit::Release;
            images[index].shadowValid = false;
            images[index].hiddenClobbered = false;
            images[index].carrierEligible = false;
            images[index].lastApplicationPresentFamily.reset();
        }
    }

    void presented(uint32_t index,
            std::optional<uint32_t> queueFamily = std::nullopt) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return;
        images[index].appAcquired = false;
        images[index].releasePrepared = false;
        images[index].lastApplicationExit = D2ApplicationExit::Present;
        ++images[index].applicationGeneration;
        images[index].shadowValid = false;
        images[index].hiddenClobbered = false;
        images[index].carrierEligible = false;
        images[index].lastApplicationPresentFamily = queueFamily;
        images[index].carrierEligible = carrierEligibleLocked(images[index]);
    }

    [[nodiscard]] bool carrierEligible(uint32_t index) const noexcept {
        const std::scoped_lock lock(mutex);
        return index < images.size() && carrierEligibleLocked(images[index]);
    }

    [[nodiscard]] bool hiddenAcquireOpportunityAllowed() const noexcept {
        const std::scoped_lock lock(mutex);
        if (retired || terminalForLsfgWorkState || !carrierCopySupported
                || !maintenanceAvailable)
            return false;
        for (const auto& image : images)
            if (carrierEligibleLocked(image)) return true;
        return false;
    }

    [[nodiscard]] bool publicAcquireRelayRequired() const noexcept {
        const std::scoped_lock lock(mutex);
        for (const auto& image : images)
            if (image.hiddenClobbered) return true;
        return false;
    }

    [[nodiscard]] D2HiddenAcquireClassification recordHiddenAcquireSuccess(
            uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) {
            terminalHiddenAcquireIndex = index;
            latchTerminalForLsfgWorkLocked();
            return D2HiddenAcquireClassification::LocalStateConflict;
        }
        auto& image = images[index];
        const bool conflict = image.appAcquired || image.releasePrepared
            || image.batchPresentPrepared || image.hiddenAcquireOwned
            || image.hiddenReleasePrepared;
        image.hiddenAcquireOwned = true;
        image.carrierEligible = false;
        if (conflict) {
            image.hiddenOwnership = D2HiddenAcquireOwnership::Terminal;
            image.hiddenReleasePrepared = false;
            latchTerminalForLsfgWorkLocked();
            return D2HiddenAcquireClassification::LocalStateConflict;
        }
        image.hiddenOwnership = D2HiddenAcquireOwnership::AcquiredUntouched;
        return carrierEligibleLockedIgnoringOwnership(image)
            ? D2HiddenAcquireClassification::EligibleCarrier
            : D2HiddenAcquireClassification::IneligibleCarrier;
    }

    [[nodiscard]] bool commitShadowSave(uint32_t index,
            uint64_t generation) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.hiddenAcquireOwned
                || generation != image.applicationGeneration)
            return false;
        if (image.shadowValid && image.shadowGeneration == generation)
            return true;
        if (image.hiddenOwnership
                != D2HiddenAcquireOwnership::AcquiredUntouched)
            return false;
        if (!image.shadowValid) {
            image.shadowGeneration = generation;
            image.shadowValid = true;
        } else if (image.shadowGeneration != generation) {
            return false;
        }
        image.hiddenOwnership = D2HiddenAcquireOwnership::UsedByGpu;
        return true;
    }

    [[nodiscard]] bool shadowSaveRequired(uint32_t index,
            uint64_t generation) const noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        const auto& image = images[index];
        return image.hiddenAcquireOwned
            && image.applicationGeneration == generation
            && (!image.shadowValid || image.shadowGeneration != generation);
    }

    /// Records the irreversible boundary at which a hidden-acquired physical
    /// image has been referenced by an accepted GPU submission.  This does
    /// not claim that a shadow copy completed (or is even usable); it only
    /// prevents the untouched-image Release recovery from being used after
    /// device work has been accepted.
    [[nodiscard]] bool markHiddenGpuUseAccepted(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.hiddenAcquireOwned
                || image.hiddenOwnership
                    != D2HiddenAcquireOwnership::AcquiredUntouched)
            return false;
        image.hiddenOwnership = D2HiddenAcquireOwnership::UsedByGpu;
        image.carrierEligible = false;
        return true;
    }

    [[nodiscard]] bool markHiddenAcquireTerminal(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !images[index].hiddenAcquireOwned)
            return false;
        auto& image = images[index];
        image.hiddenOwnership = D2HiddenAcquireOwnership::Terminal;
        image.hiddenReleasePrepared = false;
        image.carrierEligible = false;
        latchTerminalForLsfgWorkLocked();
        return true;
    }

    [[nodiscard]] bool markHiddenClobbered(uint32_t index,
            uint64_t generation) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.hiddenAcquireOwned || !image.shadowValid
                || image.shadowGeneration != generation
                || image.applicationGeneration != generation)
            return false;
        image.hiddenClobbered = true;
        image.carrierEligible = false;
        image.hiddenOwnership = D2HiddenAcquireOwnership::UsedByGpu;
        return true;
    }

    [[nodiscard]] D2RestoreDisposition restoreDisposition(
            uint32_t index) const noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size())
            return D2RestoreDisposition::InvalidShadowAuthority;
        const auto& image = images[index];
        if (!image.hiddenClobbered)
            return D2RestoreDisposition::NotClobbered;
        return image.shadowValid
                && image.shadowGeneration == image.applicationGeneration
            ? D2RestoreDisposition::Required
            : D2RestoreDisposition::InvalidShadowAuthority;
    }

    void invalidateShadowAuthority(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return;
        images[index].shadowValid = false;
    }

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    [[nodiscard]] bool modelCarrierReturnedToWsiForTesting(
            uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !images[index].hiddenAcquireOwned
                || !images[index].hiddenClobbered)
            return false;
        images[index].hiddenAcquireOwned = false;
        images[index].hiddenOwnership =
            D2HiddenAcquireOwnership::PresentEnqueued;
        return true;
    }
#endif

    [[nodiscard]] bool commitRestore(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return false;
        auto& image = images[index];
        if (!image.hiddenClobbered || !image.shadowValid
                || image.shadowGeneration != image.applicationGeneration)
            return false;
        image.hiddenClobbered = false;
        image.hiddenAcquireOwned = false;
        image.hiddenOwnership = D2HiddenAcquireOwnership::None;
        image.carrierEligible = carrierEligibleLocked(image);
        return true;
    }

    [[nodiscard]] bool prepareHiddenRelease(uint32_t index) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !maintenanceAvailable) return false;
        auto& image = images[index];
        if (!image.hiddenAcquireOwned
                || image.hiddenOwnership
                    != D2HiddenAcquireOwnership::AcquiredUntouched
                || image.hiddenReleasePrepared)
            return false;
        image.hiddenReleasePrepared = true;
        return true;
    }

    void finishHiddenRelease(uint32_t index, VkResult result) noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size() || !images[index].hiddenReleasePrepared)
            return;
        auto& image = images[index];
        image.hiddenReleasePrepared = false;
        if (result != VK_SUCCESS) return;
        image.hiddenAcquireOwned = false;
        image.hiddenOwnership = D2HiddenAcquireOwnership::Released;
        image.carrierEligible = carrierEligibleLocked(image);
    }

    [[nodiscard]] std::optional<uint32_t> terminalHiddenAcquireAuthority()
            const noexcept {
        const std::scoped_lock lock(mutex);
        return terminalHiddenAcquireIndex;
    }

    [[nodiscard]] bool terminalForLsfgWork() const noexcept {
        const std::scoped_lock lock(mutex);
        return terminalForLsfgWorkState;
    }

    void markTerminalForLsfgWork() noexcept {
        const std::scoped_lock lock(mutex);
        latchTerminalForLsfgWorkLocked();
    }

    [[nodiscard]] std::optional<D2PhysicalImageState> state(
            uint32_t index) const noexcept {
        const std::scoped_lock lock(mutex);
        if (index >= images.size()) return std::nullopt;
        return images[index];
    }

private:
    void latchTerminalForLsfgWorkLocked() noexcept {
        terminalForLsfgWorkState = true;
        for (auto& image : images) image.carrierEligible = false;
    }

    [[nodiscard]] bool carrierEligibleLockedIgnoringOwnership(
            const D2PhysicalImageState& image) const noexcept {
        return carrierCopySupported && maintenanceAvailable && !retired
            && !terminalForLsfgWorkState
            && !image.appAcquired && image.firstAppAcquireSeen
            && image.lastApplicationExit == D2ApplicationExit::Present
            && image.lastApplicationPresentFamily.has_value()
            && (!carrierExecutionFamily.has_value()
                || image.lastApplicationPresentFamily == carrierExecutionFamily)
            && !image.hiddenClobbered && !image.publicAcquirePending;
    }

    [[nodiscard]] bool carrierEligibleLocked(
            const D2PhysicalImageState& image) const noexcept {
        return carrierEligibleLockedIgnoringOwnership(image)
            && !image.batchPresentPrepared && !image.hiddenAcquireOwned
            && !image.hiddenReleasePrepared;
    }

    mutable std::mutex mutex;
    std::vector<D2PhysicalImageState> images;
    bool retired{};
    bool carrierCopySupported{};
    bool maintenanceAvailable{};
    std::optional<uint32_t> carrierExecutionFamily;
    bool terminalForLsfgWorkState{};
    std::optional<uint32_t> terminalHiddenAcquireIndex;
};

enum class D2HiddenAcquireOutcome : uint8_t {
    NoImage, OwnedEligible, OwnedIneligible, OwnedTerminal
};

struct D2HiddenAcquireResult {
    D2HiddenAcquireOutcome outcome{D2HiddenAcquireOutcome::NoImage};
    VkResult downstreamResult{VK_ERROR_UNKNOWN};
    uint32_t physicalIndex{};
    // SUCCESS/SUBOPTIMAL signals this LSFG-owned semaphore independently of
    // physical-image release; Pass 4B1B must retire or consume that signal.
    bool internalAcquireSyncRetirementRequired{};
};

/// Exactly one timeout-zero hidden acquire using only LSFG-owned sync.
template<class Acquire>
[[nodiscard]] D2HiddenAcquireResult attemptD2HiddenAcquire(
        D2RealWsiState& state, VkSwapchainKHR swapchain,
        VkSemaphore internalSemaphore, VkFence internalFence,
        Acquire&& acquire) noexcept {
    if (!state.hiddenAcquireOpportunityAllowed())
        return {D2HiddenAcquireOutcome::NoImage, VK_NOT_READY, 0, false};
    uint32_t index{};
    const auto result = acquire(
        swapchain, uint64_t{0}, internalSemaphore, internalFence, &index);
    if (result == VK_NOT_READY)
        return {D2HiddenAcquireOutcome::NoImage, result, 0, false};
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        return {D2HiddenAcquireOutcome::NoImage, result, 0, false};
    const auto classification = state.recordHiddenAcquireSuccess(index);
    const auto outcome = classification
            == D2HiddenAcquireClassification::EligibleCarrier
        ? D2HiddenAcquireOutcome::OwnedEligible
        : classification == D2HiddenAcquireClassification::IneligibleCarrier
        ? D2HiddenAcquireOutcome::OwnedIneligible
        : D2HiddenAcquireOutcome::OwnedTerminal;
    return {outcome, result, index, true};
}

template<class Acquire>
[[nodiscard]] D2HiddenAcquireResult attemptD2HiddenAcquire(
        D2RealWsiState& state, VkSwapchainKHR swapchain,
        VkSemaphore internalSemaphore, Acquire&& acquire) noexcept {
    return attemptD2HiddenAcquire(state, swapchain, internalSemaphore,
        VK_NULL_HANDLE, std::forward<Acquire>(acquire));
}

}
