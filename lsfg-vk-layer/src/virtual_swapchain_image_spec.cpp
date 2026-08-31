/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "virtual_swapchain_image_spec.hpp"

#include <cstdint>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

namespace {
    const VkImageUsageFlags2CreateInfoKHR* findUsageFlags2(const void* chain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(chain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_IMAGE_USAGE_FLAGS_2_CREATE_INFO_KHR)
                return reinterpret_cast<const VkImageUsageFlags2CreateInfoKHR*>(current);
            current = current->pNext;
        }
        return nullptr;
    }

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
    spec.arrayLayers = info.imageArrayLayers;
    spec.effectiveUsage = effectiveSwapchainImageUsage(info);
    spec.usageFitsLegacy = (spec.effectiveUsage & ~VkImageUsageFlags2KHR{UINT32_MAX}) == 0;
    spec.usage = static_cast<VkImageUsageFlags>(spec.effectiveUsage);
    constexpr VkImageUsageFlags2KHR requiredTransferUsage =
        VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT_KHR
        | VK_IMAGE_USAGE_2_TRANSFER_DST_BIT_KHR;
    spec.hasRequiredTransferUsage =
        (spec.effectiveUsage & requiredTransferUsage) == requiredTransferUsage;
    spec.sharingMode = info.imageSharingMode;

    // These WSI-only flags control presentation behavior of the real swapchain
    // and do not imply VkImageCreateFlags for the application-visible virtual
    // images. Keep them on the real WSI swapchain, but do not reject the
    // virtual bridge because of them.
    constexpr VkSwapchainCreateFlagsKHR supportedFlags =
        VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR
        | VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR
        | VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR
        | VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
    spec.unsupportedSwapchainFlags = info.flags & ~supportedFlags;

    if (info.flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR)
        spec.imageFlags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
            | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;

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

VkImageUsageFlags2KHR lsfgvk::layer::effectiveSwapchainImageUsage(
        const VkSwapchainCreateInfoKHR& info) noexcept {
    if (const auto* usage2 = findUsageFlags2(info.pNext))
        return usage2->usage;
    return info.imageUsage;
}
