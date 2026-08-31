/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "aborted_present_semantics.hpp"

namespace lsfgvk::layer {

/// Allocation-complete snapshot and state authority for one application
/// vkQueuePresentKHR call.  This object deliberately does not retain pNext or
/// pResults pointers: those are caller-owned and may only be used synchronously
/// by the eventual single final downstream present.
class BatchPresentTransaction final {
public:
    enum class Path : uint8_t { Native, D2, VirtualPrepared, Unsupported };
    enum class State : uint8_t {
        Entry,
        Validated,
        StorageReserved,
        WaitBridgePrepared,
        WaitBridgeAccepted,
        EntriesPreparing,
        EntriesReady,
        FinalPresentPrepared,
        FinalPresentCalled,
        FinalPresentNormal,
        FinalPresentEnqueuedRejection,
        FinalPresentPreEnqueueFailure,
        FinalPresentIndeterminate,
        RetirementPending,
        Retired,
        DeviceLost
    };

    struct Entry final {
        VkSwapchainKHR applicationSwapchain{};
        uint32_t applicationImageIndex{};
        Path path{Path::Unsupported};
        VkSwapchainKHR physicalSwapchain{};
        uint32_t physicalImageIndex{};
        VkSemaphore completionSemaphore{};
        uint64_t deviceLifetimeIdentity{};
        uint64_t lifecycleGeneration{};
        uint64_t presentOperationIdentity{};
        bool preparationAccepted{};
    };

    enum class CreateFailureInjection : uint8_t {
        None, WaitStorage, EntryStorage, FinalStorage
    };

    [[nodiscard]] static std::optional<BatchPresentTransaction> create(
        VkDevice device,
        VkQueue queue,
        uint64_t transactionId,
        const VkPresentInfoKHR& info,
        CreateFailureInjection = CreateFailureInjection::None) noexcept;

    [[nodiscard]] State state() const noexcept { return stateValue; }
    [[nodiscard]] VkDevice device() const noexcept { return deviceValue; }
    [[nodiscard]] VkQueue queue() const noexcept { return queueValue; }
    [[nodiscard]] uint64_t transactionId() const noexcept { return idValue; }
    [[nodiscard]] uint32_t applicationSwapchainCount() const noexcept {
        return static_cast<uint32_t>(entriesValue.size());
    }
    [[nodiscard]] bool applicationHadPNext() const noexcept { return hadPNext; }
    [[nodiscard]] const std::vector<VkSemaphore>& applicationWaits() const noexcept {
        return applicationWaitsValue;
    }
    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entriesValue;
    }
    [[nodiscard]] Entry& entry(uint32_t index) noexcept { return entriesValue[index]; }
    [[nodiscard]] const std::vector<VkSwapchainKHR>& finalSwapchains() const noexcept {
        return finalSwapchainsValue;
    }
    [[nodiscard]] const std::vector<uint32_t>& finalImageIndices() const noexcept {
        return finalImageIndicesValue;
    }
    [[nodiscard]] const std::vector<VkSemaphore>& finalWaits() const noexcept {
        return finalWaitsValue;
    }

    [[nodiscard]] bool markWaitBridgePrepared() noexcept;
    [[nodiscard]] bool markWaitBridgeAccepted() noexcept;
    [[nodiscard]] bool markEntriesPreparing() noexcept;
    [[nodiscard]] bool markEntriesReady() noexcept;
    [[nodiscard]] bool setFinalEntry(uint32_t index, VkSwapchainKHR,
        uint32_t imageIndex) noexcept;
    [[nodiscard]] bool setFinalWait(uint32_t index, VkSemaphore) noexcept;
    [[nodiscard]] bool appendFinalWait(VkSemaphore) noexcept;
    [[nodiscard]] std::optional<VkPresentInfoKHR> prepareFinalPresentInfo(
        const void* pNext, VkResult* pResults, uint32_t waitCount) noexcept;
    [[nodiscard]] bool markFinalPresentPrepared() noexcept;
    [[nodiscard]] bool markFinalPresentCalled() noexcept;
    [[nodiscard]] bool markFinalPresentResult(VkResult result) noexcept;
    [[nodiscard]] bool markRetirementPending() noexcept;
    [[nodiscard]] bool markRetired() noexcept;
    void markDeviceLost() noexcept { stateValue = State::DeviceLost; }

    [[nodiscard]] bool hasQueueSideEffect() const noexcept;
    [[nodiscard]] bool applicationWaitsConsumed() const noexcept;
    [[nodiscard]] VkResult finalResult() const noexcept { return finalResultValue; }
    [[nodiscard]] PresentResultClass finalResultClass() const noexcept {
        return finalResultClassValue;
    }

private:
    [[nodiscard]] bool transition(State from, State to) noexcept;

    VkDevice deviceValue{};
    VkQueue queueValue{};
    uint64_t idValue{};
    bool hadPNext{};
    bool bridgeAcceptedValue{};
    State stateValue{State::Entry};
    VkResult finalResultValue{VK_NOT_READY};
    PresentResultClass finalResultClassValue{PresentResultClass::Indeterminate};
    std::vector<VkSemaphore> applicationWaitsValue;
    std::vector<Entry> entriesValue;
    std::vector<VkSwapchainKHR> finalSwapchainsValue;
    std::vector<uint32_t> finalImageIndicesValue;
    std::vector<VkSemaphore> finalWaitsValue;
    uint32_t finalWaitCountValue{};
};

} // namespace lsfgvk::layer
