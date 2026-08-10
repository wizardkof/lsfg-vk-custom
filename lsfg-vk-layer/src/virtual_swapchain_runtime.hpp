/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "virtual_swapchain_image_spec.hpp"
#include "virtual_swapchain_state.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

    /// Owns the application-visible images of a virtual swapchain.
    ///
    /// Stage 3C2B2A intentionally keeps presentation synchronous. This object
    /// lives outside Root/Swapchain contexts so hot-reloading a profile cannot
    /// destroy VkImage handles that were already returned to the application.
    class VirtualSwapchainRuntime {
    public:
        VirtualSwapchainRuntime(const vk::Vulkan& vk,
            VkQueue offloadQueue,
            std::shared_ptr<std::mutex> offloadMutex,
            size_t imageCount,
            const VirtualSwapchainImageSpec& spec);

        [[nodiscard]] VkResult getImages(uint32_t* count, VkImage* images) const noexcept;

        [[nodiscard]] VkResult acquire(uint64_t timeout,
            VkSemaphore semaphore, VkFence fence, uint32_t* imageIndex) noexcept;

        [[nodiscard]] bool beginPresent(uint32_t imageIndex) noexcept;
        void completePresent(uint32_t imageIndex) noexcept;
        void stop() noexcept;

        [[nodiscard]] std::vector<VkImage> imageHandles() const;

        VirtualSwapchainRuntime(const VirtualSwapchainRuntime&) = delete;
        VirtualSwapchainRuntime& operator=(const VirtualSwapchainRuntime&) = delete;
        VirtualSwapchainRuntime(VirtualSwapchainRuntime&&) = delete;
        VirtualSwapchainRuntime& operator=(VirtualSwapchainRuntime&&) = delete;

    private:
        [[nodiscard]] VkResult signalAcquire(VkSemaphore semaphore, VkFence fence) const noexcept;

        ls::R<const vk::Vulkan> vk;
        VkQueue offloadQueue{VK_NULL_HANDLE};
        std::shared_ptr<std::mutex> offloadMutex;
        std::vector<vk::Image> images;
        VirtualSwapchainState state;
        std::atomic<uint64_t> presentSerial{1};
    };

}
