/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "virtual_swapchain_image_spec.hpp"

#include <cstdint>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

namespace {
    const VkImageFormatListCreateInfo* findFormatList(const void* chain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(chain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO)
                return reinterpret_cast<const VkImageFormatListCreateInfo*>(current);
            current = current->pNext;
        }
        return nullptr;
    }
}

VirtualSwapchainImageSpec lsfgvk::layer::makeVirtualSwapchainImageSpec(
        const VkSwapchainCreateInfoKHR& info) {
    VirtualSwapchainImageSpec spec{};
    spec.extent = info.imageExtent;
    spec.format = info.imageFormat;
    spec.usage = info.imageUsage;
    spec.sharingMode = info.imageSharingMode;

    constexpr VkSwapchainCreateFlagsKHR supportedFlags =
        VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
    spec.unsupportedSwapchainFlags = info.flags & ~supportedFlags;

    if (info.flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR)
        spec.imageFlags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

    if (info.imageSharingMode == VK_SHARING_MODE_CONCURRENT
            && info.queueFamilyIndexCount
            && info.pQueueFamilyIndices) {
        spec.queueFamilyIndices.assign(
            info.pQueueFamilyIndices,
            info.pQueueFamilyIndices + info.queueFamilyIndexCount);
    }

    if (const auto* formatList = findFormatList(info.pNext)) {
        spec.hasFormatList = true;
        if (formatList->viewFormatCount && formatList->pViewFormats) {
            spec.viewFormats.assign(
                formatList->pViewFormats,
                formatList->pViewFormats + formatList->viewFormatCount);
        }
    }

    return spec;
}
