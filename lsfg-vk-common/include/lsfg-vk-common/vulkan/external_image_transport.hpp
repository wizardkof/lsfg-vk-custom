/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {
    struct ExplicitDrmImageLayout {
        uint64_t modifier{};
        std::vector<VkSubresourceLayout> planes;
    };

    [[nodiscard]] bool explicitDrmImageLayoutMatches(
        const ExplicitDrmImageLayout& requested,
        uint64_t actualModifier,
        const std::vector<VkSubresourceLayout>& actualPlanes) noexcept;

    [[nodiscard]] std::optional<uint32_t> drmFourccForVkFormat(VkFormat format) noexcept;
    [[nodiscard]] bool validateImagePlaneMetadata(
        uint32_t requestedFourcc, uint64_t requestedModifier,
        uint32_t requestedPlanes, uint32_t actualFourcc,
        uint64_t actualModifier, uint32_t actualPlanes,
        const std::vector<VkSubresourceLayout>& layouts) noexcept;
    [[nodiscard]] std::vector<VkSubresourceLayout> normalizeExplicitImagePlaneLayouts(
        const std::vector<VkSubresourceLayout>& queriedLayouts,
        uint32_t arrayLayers, uint32_t depth);
    struct ImageModifierInfo {
        uint64_t modifier{};
        uint32_t planeCount{};
        VkFormatFeatureFlags2 features{};
    };

    [[nodiscard]] std::vector<ImageModifierInfo> intersectImageModifiers(
        const std::vector<ImageModifierInfo>& a,
        const std::vector<ImageModifierInfo>& b) noexcept;
    [[nodiscard]] std::vector<ImageModifierInfo> filterImageModifiers(
        const std::vector<ImageModifierInfo>& modifiers,
        VkFormatFeatureFlags2 required) noexcept;
    [[nodiscard]] std::optional<ImageModifierInfo> selectImageModifier(
        std::vector<ImageModifierInfo> modifiers) noexcept;

    struct ImageTransportCandidate {
        VkFormat format{};
        VkImageTiling tiling{};
        VkImageUsageFlags usage{};
        VkExtent3D extent{};
        VkExternalMemoryHandleTypeFlags handleType{};
        uint64_t modifier{};
        uint32_t planeCount{};
    };

    struct ImageTransportCapability {
        VkResult result{VK_ERROR_FORMAT_NOT_SUPPORTED};
        VkExternalMemoryFeatureFlags features{};
        VkExternalMemoryHandleTypeFlags compatibleHandleTypes{};
        VkImageFormatProperties limits{};

        [[nodiscard]] bool usable(VkExternalMemoryHandleTypeFlags handle) const noexcept;
    };

    struct ImageTransportContract {
        ImageTransportCandidate candidate{};
        ImageTransportCapability a{};
        ImageTransportCapability b{};
    };

    [[nodiscard]] constexpr bool imageNeedsDedicatedAllocation(
        VkExternalMemoryFeatureFlags externalFeatures,
        VkMemoryDedicatedRequirements dedicated) noexcept {
        return (externalFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0
            || dedicated.requiresDedicatedAllocation == VK_TRUE;
    }

    [[nodiscard]] std::optional<ImageTransportContract> chooseImageTransportContract(
        const std::vector<ImageTransportContract>& candidates) noexcept;
}
