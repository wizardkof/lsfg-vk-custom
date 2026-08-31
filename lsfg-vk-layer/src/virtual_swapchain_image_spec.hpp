/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/vulkan/image.hpp"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

    /// Vulkan image properties that can safely mirror a swapchain image.
    ///
    /// unsupportedSwapchainFlags is intentionally retained so the runtime
    /// integration can fall back to the synchronous path instead of silently
    /// creating an incompatible virtual image.
    struct VirtualSwapchainImageSpec {
        VkImageCreateFlags imageFlags{};
        VkSwapchainCreateFlagsKHR unsupportedSwapchainFlags{};
        VkExtent2D extent{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        uint32_t arrayLayers{1};
        VkImageUsageFlags2KHR effectiveUsage{};
        VkImageUsageFlags usage{};
        bool usageFitsLegacy{true};
        bool hasRequiredTransferUsage{};
        VkSharingMode sharingMode{VK_SHARING_MODE_EXCLUSIVE};
        std::vector<uint32_t> queueFamilyIndices;
        bool hasFormatList{};
        std::vector<VkFormat> viewFormats;

        [[nodiscard]] VkImageFormatListCreateInfo makeFormatListInfo() const {
            VkImageFormatListCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
            info.viewFormatCount = static_cast<uint32_t>(this->viewFormats.size());
            info.pViewFormats = this->viewFormats.empty() ? nullptr : this->viewFormats.data();
            return info;
        }

        [[nodiscard]] vk::ImageCreateOptions imageOptions(const void* pNext = nullptr) const {
            return vk::ImageCreateOptions {
                .flags = this->imageFlags,
                .arrayLayers = this->arrayLayers,
                .sharingMode = this->sharingMode,
                .queueFamilyIndices = std::span<const uint32_t>(this->queueFamilyIndices),
                .pNext = pNext
            };
        }

        [[nodiscard]] bool supported() const {
            // The current FG exchange images are structurally single-layer.
            // Preserve the real descriptor above, but keep multilayer
            // swapchains on the logical pass-through path until both sides of
            // the exchange can preserve every layer.
            return this->arrayLayers == 1
                && this->unsupportedSwapchainFlags == 0
                && this->usageFitsLegacy
                && this->hasRequiredTransferUsage;
        }
    };

    /// Extract image-creation details from a swapchain create info without
    /// touching the real swapchain. Image-affecting flags are mirrored when
    /// supported; presentation-only WSI flags can remain on the real swapchain
    /// without becoming VkImageCreateFlags. Other flags are surfaced as
    /// unsupported so runtime integration can fall back safely.
    [[nodiscard]] VirtualSwapchainImageSpec
    makeVirtualSwapchainImageSpec(const VkSwapchainCreateInfoKHR& info);

    [[nodiscard]] VkImageUsageFlags2KHR
    effectiveSwapchainImageUsage(const VkSwapchainCreateInfoKHR& info) noexcept;

}
