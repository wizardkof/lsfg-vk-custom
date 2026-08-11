/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "../lsfg-vk-layer/src/virtual_swapchain_image_spec.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

int main() {
    {
        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
        info.imageExtent = { 1920, 1080 };
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const auto spec = makeVirtualSwapchainImageSpec(info);
        assert(spec.supported());
        assert(spec.imageFlags == 0);
        assert(spec.extent.width == 1920);
        assert(spec.extent.height == 1080);
        assert(spec.format == VK_FORMAT_B8G8R8A8_UNORM);
        assert(spec.usage == info.imageUsage);
        assert(spec.sharingMode == VK_SHARING_MODE_EXCLUSIVE);
        assert(spec.queueFamilyIndices.empty());
        assert(!spec.hasFormatList);
        assert(spec.viewFormats.empty());

        const auto options = spec.imageOptions();
        assert(options.flags == 0);
        assert(options.sharingMode == VK_SHARING_MODE_EXCLUSIVE);
        assert(options.queueFamilyIndices.empty());
        assert(options.pNext == nullptr);
    }

    {
        const std::array<VkFormat, 2> viewFormats {
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_B8G8R8A8_SRGB
        };
        VkImageFormatListCreateInfo formatList{};
        formatList.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
        formatList.viewFormatCount = static_cast<uint32_t>(viewFormats.size());
        formatList.pViewFormats = viewFormats.data();
        const std::array<uint32_t, 2> queueFamilies { 2, 5 };
        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.pNext = &formatList;
        info.flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
        info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
        info.imageExtent = { 2560, 1440 };
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size());
        info.pQueueFamilyIndices = queueFamilies.data();

        const auto spec = makeVirtualSwapchainImageSpec(info);
        assert(spec.supported());
        assert((spec.imageFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0);
        assert(spec.hasFormatList);
        assert(spec.viewFormats.size() == 2);
        assert(spec.viewFormats.at(0) == VK_FORMAT_B8G8R8A8_UNORM);
        assert(spec.viewFormats.at(1) == VK_FORMAT_B8G8R8A8_SRGB);
        assert(spec.viewFormats.data() != viewFormats.data());
        assert(spec.queueFamilyIndices.size() == 2);
        assert(spec.queueFamilyIndices.at(0) == 2);
        assert(spec.queueFamilyIndices.at(1) == 5);

        const auto copiedFormatList = spec.makeFormatListInfo();
        assert(copiedFormatList.viewFormatCount == 2);
        assert(copiedFormatList.pViewFormats == spec.viewFormats.data());

        const auto options = spec.imageOptions(&copiedFormatList);
        assert(options.pNext == &copiedFormatList);
        assert(options.queueFamilyIndices.size() == 2);
    }

    {
        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.flags = VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR
            | VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR
            | VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const auto spec = makeVirtualSwapchainImageSpec(info);
        assert(spec.supported());
        assert(spec.unsupportedSwapchainFlags == 0);
        assert(spec.imageFlags == 0);
    }

    {
        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.flags = VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const auto spec = makeVirtualSwapchainImageSpec(info);
        assert(!spec.supported());
        assert((spec.unsupportedSwapchainFlags
            & VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR) != 0);
        assert((spec.imageFlags & VK_IMAGE_CREATE_PROTECTED_BIT) == 0);
    }

    std::cout << "All VirtualSwapchainImageSpec tests passed.\n";
}
