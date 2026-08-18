/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace vk {

    struct VulkanInstanceInventoryFuncs;

    // These value types preserve one exact Vulkan query and its raw result.
    // They intentionally contain no LSFG-specific transport policy.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    struct ExternalImageDescriptor {
        VkFormat format{};
        VkImageType imageType{};
        VkImageTiling tiling{};
        VkImageUsageFlags usage{};
        VkImageCreateFlags flags{};
        VkExtent3D extent{};
        uint32_t mipLevels{};
        uint32_t arrayLayers{};
        VkSampleCountFlagBits samples{};
    };

    struct ExternalImageCapabilityQuery {
        ExternalImageDescriptor image;
        VkExternalMemoryHandleTypeFlagBits handleType{};
    };

    struct ExternalImageCapability {
        ExternalImageCapabilityQuery query;
        VkResult result{};
        VkImageFormatProperties limits{};
        VkExternalMemoryFeatureFlags externalMemoryFeatures{};
        VkExternalMemoryHandleTypeFlags exportFromImportedHandleTypes{};
        VkExternalMemoryHandleTypeFlags compatibleHandleTypes{};

        [[nodiscard]] bool querySucceeded() const;
        [[nodiscard]] bool exportable() const;
        [[nodiscard]] bool importable() const;
        [[nodiscard]] bool dedicatedOnly() const;
        [[nodiscard]] bool queriedHandleCompatible() const;
        [[nodiscard]] bool descriptorFitsLimits() const;
    };

    struct ExternalSemaphoreCapability {
        VkExternalSemaphoreHandleTypeFlagBits handleType{};
        VkExternalSemaphoreFeatureFlags externalSemaphoreFeatures{};
        VkExternalSemaphoreHandleTypeFlags exportFromImportedHandleTypes{};
        VkExternalSemaphoreHandleTypeFlags compatibleHandleTypes{};

        [[nodiscard]] bool exportable() const;
        [[nodiscard]] bool importable() const;
        [[nodiscard]] bool queriedHandleCompatible() const;
    };

    struct ExternalBufferCapabilityQuery {
        VkBufferCreateFlags flags{};
        VkBufferUsageFlags usage{};
        VkExternalMemoryHandleTypeFlagBits handleType{};
    };

    struct ExternalBufferCapability {
        ExternalBufferCapabilityQuery query;
        VkExternalMemoryFeatureFlags externalMemoryFeatures{};
        VkExternalMemoryHandleTypeFlags exportFromImportedHandleTypes{};
        VkExternalMemoryHandleTypeFlags compatibleHandleTypes{};

        [[nodiscard]] bool exportable() const;
        [[nodiscard]] bool importable() const;
        [[nodiscard]] bool dedicatedOnly() const;
        [[nodiscard]] bool queriedHandleCompatible() const;
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    /// Query raw external-memory capabilities for one exact image descriptor.
    [[nodiscard]] ExternalImageCapability queryExternalImageCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        const ExternalImageCapabilityQuery& query
    );

    /// Query raw external-semaphore capabilities for one handle type.
    [[nodiscard]] ExternalSemaphoreCapability queryExternalSemaphoreCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        VkExternalSemaphoreHandleTypeFlagBits handleType
    );

    [[nodiscard]] ExternalBufferCapability queryExternalBufferCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        const ExternalBufferCapabilityQuery& query
    );

}
