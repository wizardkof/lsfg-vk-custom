/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b1_present_path.hpp"
#include "d3b2_insertion_path.hpp"
#include "d3b3_production_seams.hpp"

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
#include "generated_output_return_diagnostic.hpp"
#include "graphics_final_queue.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <chrono>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

enum class SwapchainReleaseBackend : uint8_t { None, Khr, Ext };

enum class GeneratedOutputTerminalConsumer : uint8_t {
    DiagnosticOnly,
    D3B1,
    D3B2,
    D3B3
};

[[nodiscard]] constexpr GeneratedOutputTerminalConsumer
selectGeneratedOutputTerminalConsumer(
        bool d3b1Eligible, bool d3b2Eligible, bool d3b3Eligible = false) noexcept {
    if (d3b3Eligible)
        return GeneratedOutputTerminalConsumer::D3B3;
    if (d3b2Eligible)
        return GeneratedOutputTerminalConsumer::D3B2;
    if (d3b1Eligible)
        return GeneratedOutputTerminalConsumer::D3B1;
    return GeneratedOutputTerminalConsumer::DiagnosticOnly;
}

    [[nodiscard]] inline bool d3b1PresentationDiagnosticEnabled() noexcept {
        const char* value = std::getenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
    }

    [[nodiscard]] inline bool d3b2InsertionDiagnosticEnabled() noexcept {
        const char* value = std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
    }

    [[nodiscard]] constexpr bool captureDiagnosticFormatSupported(VkFormat format) noexcept {
        return format == VK_FORMAT_B8G8R8A8_UNORM
            || format == VK_FORMAT_R8G8B8A8_UNORM
            || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32
            || format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    }

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
        std::vector<uint32_t> surfacePresentFamilies;
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
        SwapchainReleaseBackend releaseBackend{SwapchainReleaseBackend::None};
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
            ls::GameConf profile, SwapchainInfo info,
            uint32_t offloadQueueFamily = VK_QUEUE_FAMILY_IGNORED);

        /// Runtime render/generation role binding captured for this swapchain.
        /// P3C exposes identity/ownership only; transport is connected later.
        [[nodiscard]] const vk::RuntimeDevicePair& runtimeDevicePair() const noexcept {
            return this->devicePair;
        }

        [[nodiscard]] PrePresentGateResult prePresentGate() noexcept;

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
            std::optional<std::chrono::steady_clock::time_point> sourcePresentTime = std::nullopt,
            bool d3bSingleSwapchainEligible = false,
            const GraphicsFinalQueueInfo* graphicsFinalQueue = nullptr,
            BorrowedGraphicsQueueLease* graphicsLease = nullptr,
            bool* stopAfterCompletion = nullptr);
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
        ls::owned_ptr<VkCommandPool> virtualFinalCommandPool;
        ls::lazy<vk::CommandBuffer> virtualFinalCommandBuffer;
        uint32_t virtualFinalCommandFamily{VK_QUEUE_FAMILY_IGNORED};
        ls::lazy<vk::Semaphore> virtualFinalAcquireSemaphore;
        ls::lazy<vk::Semaphore> virtualFinalPresentSemaphore;
        ls::lazy<vk::Semaphore> d3b2OriginalAcquireSemaphore;
        ls::lazy<vk::Semaphore> d3b2GeneratedPresentSemaphore;
        ls::lazy<vk::Semaphore> d3b2OriginalPresentSemaphore;

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
        uint32_t offloadQueueFamily{VK_QUEUE_FAMILY_IGNORED};
        ls::owned_ptr<ls::R<backend::RuntimePrepassSession>> runtimePrepassSession;
        ls::owned_ptr<ls::R<backend::RuntimeGenerateSession>>
            runtimeGenerateDiagnosticSession;
        std::unique_ptr<GeneratedOutputReturnSession>
            generatedOutputReturnDiagnosticSession;
        std::optional<vk::Semaphore> returnedForGraphics;
        D3B1PresentationState d3b1State{D3B1PresentationState::IDLE};
        D3B2InsertionState d3b2State{D3B2InsertionState::INACTIVE};
        std::unique_ptr<D3B3ProductionState> d3b3ProductionState;

        void ensureGraphicsFinalResources(const vk::Vulkan& vk, uint32_t family);

        void captureRealFrameOnce(const vk::Vulkan& vk, VkImage sourceImage,
            uint32_t imageIndex, const std::vector<VkSemaphore>& bridgeSemaphores);

        ls::GameConf profile;
        SwapchainInfo info;
    };

}
