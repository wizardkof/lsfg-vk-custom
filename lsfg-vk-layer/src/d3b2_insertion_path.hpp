#pragma once

#include "lsfg-vk-common/vulkan/command_buffer.hpp"

#include <cstdint>
#include <functional>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

enum class D3B2InsertionState : uint8_t {
    INACTIVE, PREFLIGHT, GENERATED_READY, TWO_ACQUIRES_READY,
    INSERT_SUBMITTED, GENERATED_PRESENTED, ORIGINAL_PRESENTED,
    VALIDATING, PASS, FAILED
};

enum class D3B2ImageState : uint8_t {
    NOT_ACQUIRED, ACQUIRED_IDLE, SUBMITTED_FOR_GRAPHICS,
    PRESENT_CALL_ISSUED, PRESENT_ACQUISITION_RELEASED,
    RELEASED_WITH_MAINTENANCE1
};

struct D3B2PresentFencePolicy {
    VkFence generated{};
    VkFence original{};
    [[nodiscard]] bool enabled() const noexcept { return generated && original; }
};

[[nodiscard]] constexpr bool d3b2InsertionRequested(const char* value) noexcept {
    return value && value[0] == '1' && value[1] == '\0';
}

struct D3B2Source {
    VkImage image{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t sourceQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    uint32_t destinationQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    bool ownershipAcquireRequired{};
    bool blitSourceSupported{};
};

struct D3B2HiddenImage {
    VkImage image{};
    uint32_t index{};
    VkExtent2D extent{};
};

struct D3B2InsertionPath {
    D3B2Source generated;
    D3B2Source original;
    VkSemaphore originalReady{};
    VkSemaphore returnedForGraphics{};
    VkSemaphore acquireGenerated{};
    VkSemaphore acquireOriginal{};
    VkSemaphore generatedPresentReady{};
    VkSemaphore originalPresentReady{};
    VkFence graphicsFence{};
    uint32_t commandPoolFamily{VK_QUEUE_FAMILY_IGNORED};
    uint32_t submitQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    VkQueueFlags submitQueueFlags{};
    bool singleSwapchain{true};
    bool hasPNext{};
    bool fifo{};
    bool hiddenBlitDestinationSupported{};
    bool maintenanceReleaseCapable{};
    D3B2PresentFencePolicy presentFences{};
    bool distinctHiddenImages{};
    D3B2InsertionState* state{};

    std::function<D3B2HiddenImage(VkSemaphore)> acquire;
    std::function<void(const D3B2HiddenImage&, const D3B2HiddenImage&)> record;
    std::function<VkResult()> submit;
    std::function<VkResult(const D3B2HiddenImage&, VkSemaphore)> present;
    std::function<VkResult(const D3B2HiddenImage&, VkSemaphore, VkFence)> presentWithFence;
    std::function<bool()> waitGraphicsFence;
    std::function<bool()> generatedIntegrity;
    std::function<bool()> originalIdentity;
    std::function<void()> retire;
    std::function<void()> emitMarker;
    // Must dispatch vkReleaseSwapchainImagesKHR/EXT on this swapchain.
    std::function<VkResult(const std::vector<uint32_t>&)> releaseAcquiredImages;

    // The first terminal consumer is the transfer blit recorded by the shared
    // path.  Keep this explicit so a PairOperation can verify the wait stage
    // from the command contract instead of from semaphore naming.
    VkPipelineStageFlags returnedForGraphicsWaitStage{
        VK_PIPELINE_STAGE_TRANSFER_BIT};
    std::function<VkResult(VkPipelineStageFlags)> submitWithWaitStage;

    // Optional lifecycle observations used by D3B3PairOperation.  They are
    // deliberately callbacks: the D3B2 algorithm remains the sole owner of
    // hidden-image acquisition, recording, submission and present ordering.
    std::function<void()> onDestinationsAcquired;
    std::function<void()> onRecordBegin;
    std::function<void()> onRecordEnd;
    std::function<void()> onTerminalSubmitAccepted;
    std::function<void()> onGeneratedPresentAccepted;
    std::function<void()> onOriginalPresentAccepted;
    std::function<void()> onGraphicsFenceRetired;
};

[[nodiscard]] constexpr VkPipelineStageFlags d3b2FirstTerminalConsumerStage() noexcept {
    return VK_PIPELINE_STAGE_TRANSFER_BIT;
}

void validateD3B2InsertionPreflight(const D3B2InsertionPath&);
VkResult executeD3B2Insertion(D3B2InsertionPath&);

}
