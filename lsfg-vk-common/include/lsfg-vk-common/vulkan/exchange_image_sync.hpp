/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace vk {

    [[nodiscard]] constexpr VkImageSubresourceRange exchangeImageSubresourceRange() {
        return {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier makeImageBarrier(
            VkImage image,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            VkImageSubresourceRange subresourceRange = exchangeImageSubresourceRange()) {
        return {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = srcAccessMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = srcQueueFamilyIndex,
            .dstQueueFamilyIndex = dstQueueFamilyIndex,
            .image = image,
            .subresourceRange = subresourceRange
        };
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier sourceReleaseToBackend(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            localFamily, VK_QUEUE_FAMILY_EXTERNAL);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier sourceAcquireFromLayer(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_EXTERNAL, localFamily);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier sourceReleaseToLayer(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            VK_ACCESS_SHADER_READ_BIT, 0,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            localFamily, VK_QUEUE_FAMILY_EXTERNAL);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier sourceAcquireFromBackend(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_EXTERNAL, localFamily);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationInitialReleaseToBackend(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, 0,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            localFamily, VK_QUEUE_FAMILY_EXTERNAL);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationInitialAcquireFromLayer(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_EXTERNAL, localFamily);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationReleaseToLayer(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            VK_ACCESS_SHADER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            localFamily, VK_QUEUE_FAMILY_EXTERNAL);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationAcquireFromBackend(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_EXTERNAL, localFamily);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationReleaseToBackend(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            VK_ACCESS_TRANSFER_READ_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            localFamily, VK_QUEUE_FAMILY_EXTERNAL);
    }

    [[nodiscard]] constexpr VkImageMemoryBarrier destinationAcquireFromLayer(
            VkImage image, uint32_t localFamily) {
        return makeImageBarrier(image,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_EXTERNAL, localFamily);
    }

    struct ExchangeTimelineFrame {
        uint64_t sourceReady{};
        uint64_t sourceReturn{};
        uint64_t nextBase{};
        size_t generatedFrames{};

        [[nodiscard]] constexpr uint64_t destinationReady(size_t index) const {
            return this->sourceReady + 1 + index;
        }
    };

    /// Values for one real source frame. With generated frames, the final
    /// destination-ready signal is also the source-return ordering point. With
    /// zero generated frames, the prepass emits one dedicated GPU timeline
    /// signal so the next layer-side reuse can wait for source ownership.
    [[nodiscard]] constexpr ExchangeTimelineFrame makeExchangeTimelineFrame(
            uint64_t sourceReady, size_t generatedFrames) {
        const uint64_t sourceReturn = sourceReady
            + (generatedFrames == 0 ? 1 : generatedFrames);
        return {
            .sourceReady = sourceReady,
            .sourceReturn = sourceReturn,
            .nextBase = sourceReturn + 1,
            .generatedFrames = generatedFrames
        };
    }

}
