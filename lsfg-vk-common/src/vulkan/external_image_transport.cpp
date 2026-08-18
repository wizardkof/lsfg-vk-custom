/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "lsfg-vk-common/vulkan/external_image_transport.hpp"

#include <algorithm>
#include <libdrm/drm_fourcc.h>

std::optional<uint32_t> vk::drmFourccForVkFormat(VkFormat format) noexcept {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return DRM_FORMAT_ABGR8888;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return DRM_FORMAT_ARGB8888;
    case VK_FORMAT_R8_UNORM:
        return DRM_FORMAT_R8;
    default:
        return std::nullopt;
    }
}

bool vk::validateImagePlaneMetadata(uint32_t requestedFourcc, uint64_t requestedModifier,
        uint32_t requestedPlanes, uint32_t actualFourcc, uint64_t actualModifier,
        uint32_t actualPlanes, const std::vector<VkSubresourceLayout>& layouts) noexcept {
    if (requestedFourcc != actualFourcc || requestedModifier != actualModifier
            || requestedPlanes != actualPlanes || layouts.size() != actualPlanes)
        return false;
    return std::all_of(layouts.begin(), layouts.end(), [](const auto& layout) {
        return layout.rowPitch > 0;
    });
}

std::vector<vk::ImageModifierInfo> vk::intersectImageModifiers(
        const std::vector<ImageModifierInfo>& a,
        const std::vector<ImageModifierInfo>& b) noexcept {
    std::vector<ImageModifierInfo> result;
    for (const auto& left : a) {
        const auto right = std::find_if(b.begin(), b.end(),
            [&left](const auto& item) { return item.modifier == left.modifier; });
        if (right != b.end() && right->planeCount == left.planeCount)
            result.push_back({left.modifier, left.planeCount,
                left.features & right->features});
    }
    return result;
}

std::vector<vk::ImageModifierInfo> vk::filterImageModifiers(
        const std::vector<ImageModifierInfo>& modifiers,
        VkFormatFeatureFlags2 required) noexcept {
    std::vector<ImageModifierInfo> result;
    for (const auto& modifier : modifiers)
        if ((modifier.features & required) == required)
            result.push_back(modifier);
    return result;
}

std::optional<vk::ImageModifierInfo> vk::selectImageModifier(
        std::vector<ImageModifierInfo> modifiers) noexcept {
    if (modifiers.empty())
        return std::nullopt;
    std::ranges::sort(modifiers, {}, &ImageModifierInfo::modifier);
    return modifiers.front();
}

bool vk::ImageTransportCapability::usable(VkExternalMemoryHandleTypeFlags handle) const noexcept {
    return result == VK_SUCCESS
        && (features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0
        && (features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0
        && (compatibleHandleTypes & handle) != 0;
}

std::optional<vk::ImageTransportContract> vk::chooseImageTransportContract(
        const std::vector<ImageTransportContract>& candidates) noexcept {
    for (const auto& candidate : candidates) {
        if (candidate.a.usable(candidate.candidate.handleType)
                && candidate.b.usable(candidate.candidate.handleType)
                && candidate.a.limits.maxExtent.width >= candidate.candidate.extent.width
                && candidate.a.limits.maxExtent.height >= candidate.candidate.extent.height
                && candidate.b.limits.maxExtent.width >= candidate.candidate.extent.width
                && candidate.b.limits.maxExtent.height >= candidate.candidate.extent.height)
            return candidate;
    }
    return std::nullopt;
}
