/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/external_capabilities.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cassert>
#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace {
    VkResult imageResult{VK_SUCCESS};
    VkImageFormatProperties imageLimits{};
    VkExternalMemoryFeatureFlags memoryFeatures{};
    VkExternalMemoryHandleTypeFlags memoryExportFromImported{};
    VkExternalMemoryHandleTypeFlags memoryCompatible{};

    VkPhysicalDeviceImageFormatInfo2 observedImageInfo{};
    VkExternalMemoryHandleTypeFlagBits observedMemoryHandle{};
    bool sawImageInputChain{};
    bool sawImageOutputChain{};

    VkExternalSemaphoreFeatureFlags semaphoreFeatures{};
    VkExternalSemaphoreHandleTypeFlags semaphoreExportFromImported{};
    VkExternalSemaphoreHandleTypeFlags semaphoreCompatible{};
    VkExternalSemaphoreHandleTypeFlagBits observedSemaphoreHandle{};

    VKAPI_ATTR VkResult VKAPI_CALL getPhysicalDeviceImageFormatProperties2(
            VkPhysicalDevice,
            const VkPhysicalDeviceImageFormatInfo2* info,
            VkImageFormatProperties2* properties) {
        assert(info != nullptr);
        assert(info->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2);
        const auto* externalInfo = reinterpret_cast<
            const VkPhysicalDeviceExternalImageFormatInfo*>(info->pNext);
        assert(externalInfo != nullptr);
        assert(externalInfo->sType
            == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
        sawImageInputChain = true;
        observedImageInfo = *info;
        observedImageInfo.pNext = nullptr;
        observedMemoryHandle = externalInfo->handleType;

        assert(properties != nullptr);
        assert(properties->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2);
        auto* externalProperties = reinterpret_cast<
            VkExternalImageFormatProperties*>(properties->pNext);
        assert(externalProperties != nullptr);
        assert(externalProperties->sType
            == VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES);
        sawImageOutputChain = true;

        properties->imageFormatProperties = imageLimits;
        externalProperties->externalMemoryProperties = {
            .externalMemoryFeatures = memoryFeatures,
            .exportFromImportedHandleTypes = memoryExportFromImported,
            .compatibleHandleTypes = memoryCompatible
        };
        return imageResult;
    }

    VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceExternalSemaphoreProperties(
            VkPhysicalDevice,
            const VkPhysicalDeviceExternalSemaphoreInfo* info,
            VkExternalSemaphoreProperties* properties) {
        assert(info != nullptr);
        assert(info->sType
            == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO);
        assert(info->pNext == nullptr);
        observedSemaphoreHandle = info->handleType;

        assert(properties != nullptr);
        assert(properties->sType == VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES);
        assert(properties->pNext == nullptr);
        properties->externalSemaphoreFeatures = semaphoreFeatures;
        properties->exportFromImportedHandleTypes = semaphoreExportFromImported;
        properties->compatibleHandleTypes = semaphoreCompatible;
    }

    vk::VulkanInstanceInventoryFuncs fakeFunctions() {
        return {
            .GetPhysicalDeviceImageFormatProperties2 =
                getPhysicalDeviceImageFormatProperties2,
            .GetPhysicalDeviceExternalSemaphoreProperties =
                getPhysicalDeviceExternalSemaphoreProperties
        };
    }

    vk::ExternalImageCapabilityQuery imageQuery(
            VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
            VkExternalMemoryHandleTypeFlagBits handleType =
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
        return {
            .image = {
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .imageType = VK_IMAGE_TYPE_2D,
                .tiling = tiling,
                .usage = usage,
                .flags = 0,
                .extent = {1920, 1080, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT
            },
            .handleType = handleType
        };
    }

    void resetImageResponse() {
        imageResult = VK_SUCCESS;
        imageLimits = {
            .maxExtent = {4096, 4096, 1},
            .maxMipLevels = 4,
            .maxArrayLayers = 2,
            .sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
            .maxResourceSize = 1024ULL * 1024 * 1024
        };
        memoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT
            | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
        memoryExportFromImported = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        memoryCompatible = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        sawImageInputChain = false;
        sawImageOutputChain = false;
    }

    vk::ExternalImageCapability queryImage(
            const vk::ExternalImageCapabilityQuery& query) {
        return vk::queryExternalImageCapability(
            fakeFunctions(), reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}), query);
    }

    vk::ExternalSemaphoreCapability querySemaphore(
            VkExternalSemaphoreHandleTypeFlagBits handleType) {
        return vk::queryExternalSemaphoreCapability(
            fakeFunctions(), reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}), handleType);
    }
}

int main() {
    resetImageResponse();
    auto capability = queryImage(imageQuery());
    assert(sawImageInputChain);
    assert(sawImageOutputChain);
    assert(capability.querySucceeded());
    assert(capability.exportable());
    assert(capability.importable());
    assert(!capability.dedicatedOnly());
    assert(capability.queriedHandleCompatible());
    assert(capability.descriptorFitsLimits());
    assert(capability.limits.maxExtent.width == imageLimits.maxExtent.width);
    assert(capability.exportFromImportedHandleTypes == memoryExportFromImported);

    imageResult = VK_ERROR_FORMAT_NOT_SUPPORTED;
    capability = queryImage(imageQuery());
    assert(capability.result == VK_ERROR_FORMAT_NOT_SUPPORTED);
    assert(!capability.querySucceeded());
    assert(!capability.exportable());
    assert(!capability.descriptorFitsLimits());
    assert(capability.limits.maxExtent.width == 0);

    imageResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    capability = queryImage(imageQuery());
    assert(capability.result == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(!capability.querySucceeded());

    resetImageResponse();
    memoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
    capability = queryImage(imageQuery());
    assert(capability.exportable());
    assert(!capability.importable());

    memoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    capability = queryImage(imageQuery());
    assert(!capability.exportable());
    assert(capability.importable());

    memoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT
        | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
        | VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT;
    capability = queryImage(imageQuery());
    assert(capability.exportable());
    assert(capability.importable());
    assert(capability.dedicatedOnly());

    memoryCompatible = 0;
    capability = queryImage(imageQuery());
    assert(!capability.queriedHandleCompatible());

    resetImageResponse();
    const auto dmaBufQuery = imageQuery(
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    memoryCompatible = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    capability = queryImage(dmaBufQuery);
    assert(observedMemoryHandle == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assert(capability.query.handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assert(capability.queriedHandleCompatible());

    const auto transferQuery = imageQuery(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    static_cast<void>(queryImage(transferQuery));
    assert(observedImageInfo.usage == VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    const auto storageQuery = imageQuery(VK_IMAGE_USAGE_STORAGE_BIT);
    static_cast<void>(queryImage(storageQuery));
    assert(observedImageInfo.usage == VK_IMAGE_USAGE_STORAGE_BIT);

    static_cast<void>(queryImage(imageQuery(
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_TILING_LINEAR)));
    assert(observedImageInfo.tiling == VK_IMAGE_TILING_LINEAR);
    static_cast<void>(queryImage(imageQuery(
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_TILING_OPTIMAL)));
    assert(observedImageInfo.tiling == VK_IMAGE_TILING_OPTIMAL);

    auto limitsQuery = imageQuery();
    limitsQuery.image.extent = {4096, 4096, 1};
    assert(queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.extent.width = 4097;
    assert(!queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.extent = {1920, 1080, 1};
    limitsQuery.image.mipLevels = 4;
    assert(queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.mipLevels = 5;
    assert(!queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.mipLevels = 1;
    limitsQuery.image.arrayLayers = 2;
    assert(queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.arrayLayers = 3;
    assert(!queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.arrayLayers = 1;
    limitsQuery.image.samples = VK_SAMPLE_COUNT_4_BIT;
    assert(queryImage(limitsQuery).descriptorFitsLimits());
    limitsQuery.image.samples = VK_SAMPLE_COUNT_2_BIT;
    assert(!queryImage(limitsQuery).descriptorFitsLimits());

    semaphoreFeatures = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT
        | VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    semaphoreExportFromImported = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    semaphoreCompatible = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto semaphoreCapability = querySemaphore(
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
    assert(observedSemaphoreHandle == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
    assert(semaphoreCapability.exportable());
    assert(semaphoreCapability.importable());
    assert(semaphoreCapability.queriedHandleCompatible());
    assert(semaphoreCapability.exportFromImportedHandleTypes
        == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);

    semaphoreCompatible = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    semaphoreCapability = querySemaphore(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
    assert(observedSemaphoreHandle == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
    assert(semaphoreCapability.exportable());
    assert(semaphoreCapability.importable());
    assert(semaphoreCapability.queriedHandleCompatible());

    semaphoreFeatures = 0;
    semaphoreCapability = querySemaphore(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
    assert(!semaphoreCapability.exportable());
    assert(!semaphoreCapability.importable());

    semaphoreCompatible = 0;
    semaphoreCapability = querySemaphore(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
    assert(!semaphoreCapability.queriedHandleCompatible());

    return 0;
}
