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
        VkImageUsageFlags usage{};
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
                .sharingMode = this->sharingMode,
                .queueFamilyIndices = std::span<const uint32_t>(this->queueFamilyIndices),
                .pNext = pNext
            };
        }

        [[nodiscard]] bool supported() const {
            return this->unsupportedSwapchainFlags == 0;
        }
    };

    /// Extract image-creation details from a swapchain create info without
    /// touching the real swapchain. Only mutable-format is supported for now;
    /// other swapchain creation flags are surfaced as unsupported.
    [[nodiscard]] VirtualSwapchainImageSpec
    makeVirtualSwapchainImageSpec(const VkSwapchainCreateInfoKHR& info);

}
