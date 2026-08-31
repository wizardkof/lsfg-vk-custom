/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d2_real_wsi_state.hpp"
#include "acquire_fence_proxy_registry.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk { class Vulkan; class Image; class CommandBuffer; }

namespace lsfgvk::layer {

[[nodiscard]] constexpr bool d2AcquireSemaphoreSyncFdEligible(
        VkExternalSemaphoreFeatureFlags features) noexcept {
    constexpr auto required = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT
        | VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    return (features & required) == required;
}

[[nodiscard]] constexpr bool d2InternalPresentFenceSyncFdEligible(
        VkExternalFenceFeatureFlags features) noexcept {
    return (features & VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT) != 0;
}

[[nodiscard]] constexpr bool d2ExternalSyncFdFeaturesEligible(
        VkExternalSemaphoreFeatureFlags semaphoreFeatures,
        VkExternalFenceFeatureFlags fenceFeatures) noexcept {
    return d2AcquireSemaphoreSyncFdEligible(semaphoreFeatures)
        && d2InternalPresentFenceSyncFdEligible(fenceFeatures);
}

class D2VulkanShadowRuntime final {
public:
    using DownstreamAcquire = std::function<VkResult(
        VkSemaphore, VkFence, uint32_t*)>;
    using DownstreamRelease = std::function<VkResult(uint32_t)>;
    using HiddenDownstreamAcquire = std::function<VkResult(VkSwapchainKHR,
        uint64_t, VkSemaphore, VkFence, uint32_t*)>;

    D2VulkanShadowRuntime(const vk::Vulkan& vulkan, VkQueue queue,
        std::shared_ptr<std::mutex> queueMutex, std::span<const VkImage> realImages,
        VkFormat format, VkExtent2D extent);
    ~D2VulkanShadowRuntime();

    D2VulkanShadowRuntime(const D2VulkanShadowRuntime&) = delete;
    D2VulkanShadowRuntime& operator=(const D2VulkanShadowRuntime&) = delete;

    [[nodiscard]] size_t shadowCount() const noexcept;
    [[nodiscard]] VkImage shadowImage(uint32_t index) const noexcept;
    [[nodiscard]] size_t retainedTransactionCount() const noexcept {
        return retainedTransactions.size();
    }

    [[nodiscard]] VkResult publicAcquire(D2RealWsiState& state,
        VkSemaphore applicationSemaphore, VkFence applicationFence,
        uint32_t* imageIndex, const DownstreamAcquire& downstream,
        const DownstreamRelease& release = {},
        AcquireFenceProxyRegistry* proxies = nullptr) noexcept;

    [[nodiscard]] VkResult consumeAndReleaseIneligible(
        D2RealWsiState& state, uint32_t imageIndex,
        VkSemaphore internalAcquireSemaphore,
        const DownstreamRelease& downstream) noexcept;

    [[nodiscard]] VkResult submitShadowSave(D2RealWsiState& state,
        uint32_t imageIndex, uint64_t generation,
        VkSemaphore internalAcquireSemaphore) noexcept;
    [[nodiscard]] D2HiddenAcquireResult attemptHiddenAcquire(
        D2RealWsiState& state, VkSwapchainKHR swapchain,
        const HiddenDownstreamAcquire& downstreamAcquire,
        const DownstreamRelease& downstreamRelease) noexcept;
    [[nodiscard]] VkResult submitPreparedShadowSave(D2RealWsiState& state,
        uint32_t imageIndex, uint64_t generation,
        const DownstreamRelease& downstreamRelease) noexcept;

private:
    struct Transaction;
    [[nodiscard]] std::shared_ptr<Transaction> prepareTransaction(
        bool exportSemaphore, bool exportFence) const;
    [[nodiscard]] VkResult recordCopy(Transaction& transaction,
        VkImage source, VkImage destination, bool save) const;
    [[nodiscard]] VkResult submit(Transaction& transaction,
        VkSemaphore waitSemaphore, bool copy) const;
    [[nodiscard]] VkResult handoff(Transaction& transaction,
        VkSemaphore applicationSemaphore, VkFence applicationFence) const;
    [[nodiscard]] VkResult releaseUntouchedHidden(D2RealWsiState& state,
        uint32_t imageIndex, Transaction& transaction,
        VkSemaphore acquireSemaphore,
        const DownstreamRelease& downstream) noexcept;
    void reapCompletedTransactions() noexcept;

    const vk::Vulkan* vulkan{};
    VkQueue queue{VK_NULL_HANDLE};
    std::shared_ptr<std::mutex> queueMutex;
    std::vector<VkImage> realImages;
    std::vector<std::unique_ptr<vk::Image>> shadows;
    std::vector<bool> shadowInitialized;
    std::vector<std::shared_ptr<Transaction>> retainedTransactions;
    std::vector<Transaction*> preparedHiddenByImage;
};

}
