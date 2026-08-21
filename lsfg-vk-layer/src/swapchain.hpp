/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fixed_frame_scheduler.hpp"
#include "fixed_output_pacer.hpp"
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/timeline_semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "runtime_dma_buf_backing.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <chrono>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

    enum class CrossDeviceRuntimeMode : uint8_t {
        BLOCKED,
        CAPTURE_ONLY,
        FRAME_TRANSPORT
    };

    /// swapchain info struct
    struct SwapchainInfo {
        // Images exposed to the application. In the Fixed virtual path these
        // are persistent virtual images; otherwise they are the real WSI images.
        std::vector<VkImage> images;
        // Underlying WSI images. Empty/non-distinct for the legacy path.
        std::vector<VkImage> realImages;
        VkFormat format;
        VkColorSpaceKHR colorSpace;
        VkExtent2D extent;
        VkImageUsageFlags usage{};
        VkSharingMode sharingMode{VK_SHARING_MODE_EXCLUSIVE};
        std::vector<uint32_t> queueFamilyIndices;
        bool surfaceSupportsTransferSrc{};
        // Present mode currently selected by the internal context.
        VkPresentModeKHR presentMode;
        // Stable modes declared for the hidden real WSI swapchain. These are
        // used only when dynamicPresentModeEligible is true.
        VkPresentModeKHR adaptivePresentMode{VK_PRESENT_MODE_FIFO_KHR};
        VkPresentModeKHR fixedPresentMode{VK_PRESENT_MODE_FIFO_KHR};
        bool virtualized{};
        // True when LSFG injected a compatible FIFO + MAILBOX/IMMEDIATE
        // declaration and can select between them per present without
        // recreating the application-visible virtual topology.
        bool dynamicPresentModeEligible{};
        // Tracks the mode of the Root-owned Swapchain context. The virtual
        // VkImage handles and VirtualSwapchainRuntime remain stable.
        bool fixedContext{};
    };

    /// modify the swapchain create info based on the profile pre-swapchain creation
    /// @param profile active game profile
    /// @param maxImages maximum number of images supported by the surface
    /// @param createInfo swapchain create info to modify
    void context_ModifySwapchainCreateInfo(const ls::GameConf& profile, uint32_t maxImages,
        VkSwapchainCreateInfoKHR& createInfo);

    /// swapchain context for a layer instance
    class Swapchain {
    public:
        /// create a new swapchain context
        /// @param vk vulkan instance
        /// @param backend lsfg-vk backend instance
        /// @param devicePair immutable render/generation physical-device role binding
        /// @param profile active game profile
        /// @param info swapchain info
        Swapchain(const vk::Vulkan& vk, backend::Instance& backend,
            vk::RuntimeDevicePair devicePair,
            ls::GameConf profile, SwapchainInfo info);

        /// Runtime render/generation role binding captured for this swapchain.
        /// P3C exposes identity/ownership only; transport is connected later.
        [[nodiscard]] const vk::RuntimeDevicePair& runtimeDevicePair() const noexcept {
            return this->devicePair;
        }

        /// present a frame
        /// @param vk vulkan instance
        /// @param queue presentation queue
        /// @param next_chain next chain pointer for the present info (WARN: shared!)
        /// @param imageIdx swapchain image index to present to
        /// @param semaphores semaphores to wait on before presenting
        /// @throws ls::vulkan_error on vulkan errors
        VkResult present(const vk::Vulkan& vk,
            VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
            VkSwapchainKHR swapchain,
            void* next_chain, uint32_t imageIdx,
            const std::vector<VkSemaphore>& semaphores,
            std::stop_token stopToken = {},
            std::optional<std::chrono::steady_clock::time_point> sourcePresentTime = std::nullopt);
    private:
        std::vector<vk::Image> sourceImages;
        std::vector<vk::Image> destinationImages;
        ls::lazy<vk::TimelineSemaphore> syncSemaphore;

        ls::lazy<vk::CommandBuffer> renderCommandBuffer;
        ls::lazy<vk::Fence> renderFence;
        struct RenderPass {
            vk::CommandBuffer commandBuffer;
            vk::Semaphore acquireSemaphore;
        };
        std::vector<RenderPass> passes;
        std::vector<std::pair<vk::Semaphore, vk::Semaphore>> postCopySemaphores;

        // Final real-frame copy resources used only when the application renders
        // into virtual swapchain images.
        ls::lazy<vk::CommandBuffer> virtualFinalCommandBuffer;
        ls::lazy<vk::Semaphore> virtualFinalAcquireSemaphore;
        ls::lazy<vk::Semaphore> virtualFinalPresentSemaphore;

        ls::R<backend::Instance> instance;
        vk::RuntimeDevicePair devicePair;
        ls::owned_ptr<ls::R<backend::Context>> ctx;
        size_t idx{1};
        size_t fidx{0}; // real frame index
        std::array<std::optional<uint64_t>, 2> sourceReturnValues{};

        FixedFrameScheduler fixedScheduler;
        FixedOutputPacer fixedOutputPacer;
        std::optional<std::chrono::steady_clock::time_point> lastSourcePresent;

        CrossDeviceRuntimeMode crossDeviceMode{CrossDeviceRuntimeMode::BLOCKED};
        uint32_t captureOnlyPhase{};
        RuntimeDmaBufBacking frameTransportBacking;
        vk::RuntimeImageEndpoint frameTransportA;
        vk::RuntimeImageEndpoint frameTransportB;
        bool frameTransportReady{};
        ls::owned_ptr<ls::R<backend::RuntimePrepassSession>> runtimePrepassSession;

        void captureRealFrameOnce(const vk::Vulkan& vk, VkImage sourceImage,
            uint32_t imageIndex, const std::vector<VkSemaphore>& bridgeSemaphores);

        ls::GameConf profile;
        SwapchainInfo info;
    };

}
