/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/external_capabilities.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cstdint>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    [[nodiscard]] bool hasFlag(VkFlags value, VkFlags flag) {
        return (value & flag) == flag;
    }
}

bool ExternalImageCapability::querySucceeded() const {
    return this->result == VK_SUCCESS;
}

bool ExternalImageCapability::exportable() const {
    return this->querySucceeded()
        && hasFlag(this->externalMemoryFeatures,
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
}

bool ExternalImageCapability::importable() const {
    return this->querySucceeded()
        && hasFlag(this->externalMemoryFeatures,
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
}

bool ExternalImageCapability::dedicatedOnly() const {
    return this->querySucceeded()
        && hasFlag(this->externalMemoryFeatures,
            VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT);
}

bool ExternalImageCapability::queriedHandleCompatible() const {
    return this->querySucceeded()
        && this->query.handleType != 0
        && hasFlag(this->compatibleHandleTypes, this->query.handleType);
}

bool ExternalImageCapability::descriptorFitsLimits() const {
    if (!this->querySucceeded())
        return false;

    const auto& image = this->query.image;
    const auto sampleCount = static_cast<uint32_t>(image.samples);
    const bool singleSampleCount = sampleCount != 0
        && (sampleCount & (sampleCount - 1)) == 0;

    return image.extent.width > 0
        && image.extent.width <= this->limits.maxExtent.width
        && image.extent.height > 0
        && image.extent.height <= this->limits.maxExtent.height
        && image.extent.depth > 0
        && image.extent.depth <= this->limits.maxExtent.depth
        && image.mipLevels > 0
        && image.mipLevels <= this->limits.maxMipLevels
        && image.arrayLayers > 0
        && image.arrayLayers <= this->limits.maxArrayLayers
        && singleSampleCount
        && hasFlag(this->limits.sampleCounts, image.samples);
}

bool ExternalSemaphoreCapability::exportable() const {
    return hasFlag(this->externalSemaphoreFeatures,
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT);
}

bool ExternalSemaphoreCapability::importable() const {
    return hasFlag(this->externalSemaphoreFeatures,
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT);
}

bool ExternalSemaphoreCapability::queriedHandleCompatible() const {
    return this->handleType != 0
        && hasFlag(this->compatibleHandleTypes, this->handleType);
}

bool ExternalBufferCapability::exportable() const {
    return hasFlag(this->externalMemoryFeatures,
        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
}

bool ExternalBufferCapability::importable() const {
    return hasFlag(this->externalMemoryFeatures,
        VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
}

bool ExternalBufferCapability::dedicatedOnly() const {
    return hasFlag(this->externalMemoryFeatures,
        VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT);
}

bool ExternalBufferCapability::queriedHandleCompatible() const {
    return this->query.handleType != 0
        && hasFlag(this->compatibleHandleTypes, this->query.handleType);
}

ExternalImageCapability vk::queryExternalImageCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        const ExternalImageCapabilityQuery& query) {
    const VkPhysicalDeviceExternalImageFormatInfo externalInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
        .handleType = query.handleType
    };
    const VkPhysicalDeviceImageFormatInfo2 imageInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext = &externalInfo,
        .format = query.image.format,
        .type = query.image.imageType,
        .tiling = query.image.tiling,
        .usage = query.image.usage,
        .flags = query.image.flags
    };
    VkExternalImageFormatProperties externalProperties{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES
    };
    VkImageFormatProperties2 imageProperties{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = &externalProperties
    };

    ExternalImageCapability capability{
        .query = query,
        .result = funcs.GetPhysicalDeviceImageFormatProperties2(
            device, &imageInfo, &imageProperties)
    };
    if (!capability.querySucceeded())
        return capability;

    capability.limits = imageProperties.imageFormatProperties;
    capability.externalMemoryFeatures =
        externalProperties.externalMemoryProperties.externalMemoryFeatures;
    capability.exportFromImportedHandleTypes =
        externalProperties.externalMemoryProperties.exportFromImportedHandleTypes;
    capability.compatibleHandleTypes =
        externalProperties.externalMemoryProperties.compatibleHandleTypes;
    return capability;
}

ExternalSemaphoreCapability vk::queryExternalSemaphoreCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        VkExternalSemaphoreHandleTypeFlagBits handleType) {
    const VkPhysicalDeviceExternalSemaphoreInfo externalInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        .handleType = handleType
    };
    VkExternalSemaphoreProperties externalProperties{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES
    };
    funcs.GetPhysicalDeviceExternalSemaphoreProperties(
        device, &externalInfo, &externalProperties);

    return {
        .handleType = handleType,
        .externalSemaphoreFeatures = externalProperties.externalSemaphoreFeatures,
        .exportFromImportedHandleTypes =
            externalProperties.exportFromImportedHandleTypes,
        .compatibleHandleTypes = externalProperties.compatibleHandleTypes
    };
}

ExternalBufferCapability vk::queryExternalBufferCapability(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device,
        const ExternalBufferCapabilityQuery& query) {
    const VkPhysicalDeviceExternalBufferInfo info{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .flags = query.flags,
        .usage = query.usage,
        .handleType = query.handleType
    };
    VkExternalBufferProperties properties{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES
    };
    funcs.GetPhysicalDeviceExternalBufferProperties(device, &info, &properties);
    return {
        .query = query,
        .externalMemoryFeatures = properties.externalMemoryProperties.externalMemoryFeatures,
        .exportFromImportedHandleTypes =
            properties.externalMemoryProperties.exportFromImportedHandleTypes,
        .compatibleHandleTypes = properties.externalMemoryProperties.compatibleHandleTypes
    };
}
