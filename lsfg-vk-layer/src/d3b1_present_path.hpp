#pragma once

#include "lsfg-vk-common/vulkan/command_buffer.hpp"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {
    enum class D3B1PresentationState : uint8_t {
        IDLE, A_HANDOFF_SUBMITTED, OFFLOAD_SUBMITTED, PRESENT_SUBMITTED,
        PRESENT_SUCCESS, OFFLOAD_COMPLETED, PASS, FAILED
    };

    [[nodiscard]] constexpr bool d3b1PresentEligible(bool selector,
            bool singleSwapchain, bool hasPNext, bool virtualized,
            bool resourcesReady, D3B1PresentationState state) noexcept {
        return selector && singleSwapchain && !hasPNext && virtualized
            && resourcesReady && state == D3B1PresentationState::IDLE;
    }

    [[nodiscard]] constexpr bool d3b1OneShotRequested(const char* value) noexcept {
        return value && value[0] == '1' && value[1] == '\0';
    }

    [[nodiscard]] constexpr bool d3b1StopAfterTerminalPass(
            bool oneShot, D3B1PresentationState state) noexcept {
        return oneShot && state == D3B1PresentationState::PASS;
    }

    struct D3B1ReturnedSource {
        VkImage image{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkExtent2D extent{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        uint64_t modifier{};
        uint32_t sourceQueueFamily{VK_QUEUE_FAMILY_IGNORED};
        uint32_t destinationQueueFamily{VK_QUEUE_FAMILY_IGNORED};
        bool ownershipAcquireRequired{};
    };

    struct D3B1HiddenImage {
        VkImage image{};
        uint32_t index{};
        VkExtent2D extent{};
    };

    struct D3B1SubmitPresentResult {
        VkResult presentResult{VK_ERROR_UNKNOWN};
        bool offloadSubmitted{};
    };

    struct D3B1PresentPath {
        D3B1ReturnedSource source;
        VkSemaphore hiddenAcquire{};
        VkSemaphore returnedForPresent{};
        VkSemaphore finalPresentSemaphore{};
        VkFence renderFence{};
        uint32_t commandPoolFamily{VK_QUEUE_FAMILY_IGNORED};
        uint32_t submitQueueFamily{VK_QUEUE_FAMILY_IGNORED};
        VkQueueFlags submitQueueFlags{};
        bool sourceBlitSupported{};
        bool destinationBlitSupported{};
        D3B1PresentationState* state{};

        std::function<D3B1HiddenImage()> acquireHidden;
        std::function<void(const std::vector<vk::Barrier>&, VkImage, VkImage,
            VkExtent2D, VkExtent2D, const std::vector<vk::Barrier>&)> recordBlit;
        std::function<D3B1SubmitPresentResult(const std::vector<VkSemaphore>&,
            const std::vector<VkPipelineStageFlags>&, VkSemaphore, VkFence,
            uint32_t)> submitAndPresent;
        std::function<bool()> waitRenderFence;
        std::function<bool()> completeDiagnostics;
        std::function<void()> retire;
        std::function<void()> emitMarker;
    };

    VkResult executeD3B1PresentPath(D3B1PresentPath&);
}
