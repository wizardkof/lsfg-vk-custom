#include "lsfg-vk-common/vulkan/vulkan_native_external_image_backing.hpp"
#include "lsfg-vk-common/vulkan/external_image_transport.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <libdrm/drm_fourcc.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>

using namespace vk;

namespace {
    constexpr VkImageUsageFlags reverseUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    constexpr VkImageAspectFlagBits planeAspects[]{
        VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
        VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
        VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
        VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT};

    ExplicitDrmImageLayout discoverConsumerLayout(const RuntimeExchangeEndpoint& consumer,
            VkExtent2D size, VkFormat format) {
        if (!consumer.CreateImage || !consumer.DestroyImage
                || !consumer.GetImageDrmFormatModifierPropertiesEXT
                || !consumer.GetImageSubresourceLayout
                || !consumer.GetPhysicalDeviceFormatProperties2
                || !consumer.GetPhysicalDeviceImageFormatProperties2)
            throw std::invalid_argument("incomplete D3A1 layout consumer endpoint");
        VkDrmFormatModifierPropertiesListEXT list{
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
        VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        properties.pNext = &list;
        consumer.GetPhysicalDeviceFormatProperties2(consumer.physicalDevice, format, &properties);
        std::vector<VkDrmFormatModifierPropertiesEXT> available(list.drmFormatModifierCount);
        list.pDrmFormatModifierProperties = available.data();
        consumer.GetPhysicalDeviceFormatProperties2(consumer.physicalDevice, format, &properties);
        const auto linear = std::find_if(available.begin(), available.end(), [](const auto& item) {
            constexpr VkFormatFeatureFlags required =
                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
            return item.drmFormatModifier == DRM_FORMAT_MOD_LINEAR
                && item.drmFormatModifierPlaneCount == 1
                && (item.drmFormatModifierTilingFeatures & required) == required;
        });
        if (linear == available.end())
            throw ls::vulkan_error("D3A1 consumer lacks supported single-plane linear modifier");
        const uint64_t requestedModifier = DRM_FORMAT_MOD_LINEAR;
        VkPhysicalDeviceExternalImageFormatInfo externalQuery{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
        externalQuery.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierQuery{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT};
        modifierQuery.pNext = &externalQuery;
        modifierQuery.drmFormatModifier = requestedModifier;
        modifierQuery.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkPhysicalDeviceImageFormatInfo2 imageQuery{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
        imageQuery.pNext = &modifierQuery;
        imageQuery.format = format; imageQuery.type = VK_IMAGE_TYPE_2D;
        imageQuery.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        imageQuery.usage = reverseUsage;
        VkExternalImageFormatProperties externalProperties{
            VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
        VkImageFormatProperties2 imageProperties{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
        imageProperties.pNext = &externalProperties;
        const auto capabilityResult = consumer.GetPhysicalDeviceImageFormatProperties2(
            consumer.physicalDevice, &imageQuery, &imageProperties);
        constexpr auto requiredExternalFeatures =
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
            | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
        if (capabilityResult != VK_SUCCESS
                || (externalProperties.externalMemoryProperties.externalMemoryFeatures
                    & requiredExternalFeatures) != requiredExternalFeatures
                || (externalProperties.externalMemoryProperties.compatibleHandleTypes
                    & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) == 0)
            throw ls::vulkan_error("D3A1 consumer linear DMA-BUF contract unsupported");
        VkExternalMemoryImageCreateInfo external{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkImageDrmFormatModifierListCreateInfoEXT modifier{
            VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
        modifier.pNext = &external;
        modifier.drmFormatModifierCount = 1;
        modifier.pDrmFormatModifiers = &requestedModifier;
        VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        create.pNext = &modifier; create.imageType = VK_IMAGE_TYPE_2D; create.format = format;
        create.extent = {size.width, size.height, 1}; create.mipLevels = 1; create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT;
        create.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        create.usage = reverseUsage; create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage image = VK_NULL_HANDLE;
        auto r = consumer.CreateImage(consumer.bufferDevice.device, &create, nullptr, &image);
        if (r != VK_SUCCESS) throw ls::vulkan_error(r, "D3A1 consumer layout probe creation failed");
        VkImageDrmFormatModifierPropertiesEXT actual{
            VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
        r = consumer.GetImageDrmFormatModifierPropertiesEXT(
            consumer.bufferDevice.device, image, &actual);
        if (r != VK_SUCCESS) {
            consumer.DestroyImage(consumer.bufferDevice.device, image, nullptr);
            throw ls::vulkan_error(r, "D3A1 consumer modifier query failed");
        }
        if (actual.drmFormatModifier != requestedModifier) {
            consumer.DestroyImage(consumer.bufferDevice.device, image, nullptr);
            throw ls::vulkan_error("D3A1 consumer selected unexpected modifier");
        }
        VkSubresourceLayout plane{};
        const VkImageSubresource subresource{planeAspects[0], 0, 0};
        consumer.GetImageSubresourceLayout(
            consumer.bufferDevice.device, image, &subresource, &plane);
        consumer.DestroyImage(consumer.bufferDevice.device, image, nullptr);
        if (plane.rowPitch == 0)
            throw ls::vulkan_error("D3A1 consumer returned invalid plane layout");
        return {actual.drmFormatModifier, {{.offset = plane.offset, .rowPitch = plane.rowPitch}}};
    }
}

VulkanNativeExternalImageBacking::VulkanNativeExternalImageBacking(
        VulkanNativeExternalImageBacking&& other) noexcept :
    endpoint(other.endpoint), imageHandle(std::exchange(other.imageHandle, VK_NULL_HANDLE)),
    memoryHandle(std::exchange(other.memoryHandle, VK_NULL_HANDLE)),
    allocationSize(other.allocationSize), format(other.format), extent(other.extent),
    fourcc(other.fourcc), modifier(other.modifier), planes(std::move(other.planes)),
    dedicated(other.dedicated) {}

VulkanNativeExternalImageBacking& VulkanNativeExternalImageBacking::operator=(
        VulkanNativeExternalImageBacking&& other) noexcept {
    if (this != &other) {
        reset();
        endpoint = other.endpoint;
        imageHandle = std::exchange(other.imageHandle, VK_NULL_HANDLE);
        memoryHandle = std::exchange(other.memoryHandle, VK_NULL_HANDLE);
        allocationSize = other.allocationSize; format = other.format; extent = other.extent;
        fourcc = other.fourcc; modifier = other.modifier; planes = std::move(other.planes);
        dedicated = other.dedicated;
    }
    return *this;
}

VulkanNativeExternalImageBacking::~VulkanNativeExternalImageBacking() {
    reset();
}

void VulkanNativeExternalImageBacking::reset() noexcept {
    if (imageHandle && endpoint.DestroyImage)
        endpoint.DestroyImage(endpoint.bufferDevice.device, imageHandle, nullptr);
    if (memoryHandle && endpoint.FreeMemory)
        endpoint.FreeMemory(endpoint.bufferDevice.device, memoryHandle, nullptr);
    imageHandle = VK_NULL_HANDLE;
    memoryHandle = VK_NULL_HANDLE;
}

VulkanNativeExternalImageBacking VulkanNativeExternalImageBacking::create(
        const RuntimeExchangeEndpoint& source, const RuntimeExchangeEndpoint& consumer,
        VkExtent2D size, VkFormat imageFormat) {
    if (!source.CreateImage || !source.DestroyImage || !source.GetImageMemoryRequirements2
            || !source.AllocateMemory || !source.FreeMemory || !source.BindImageMemory
            || !source.GetImageDrmFormatModifierPropertiesEXT || !source.GetImageSubresourceLayout
            || !source.GetMemoryFdKHR || !source.GetPhysicalDeviceFormatProperties2
            || !source.GetPhysicalDeviceImageFormatProperties2)
        throw std::invalid_argument("incomplete Vulkan-native image exporter endpoint");
    auto fourccOpt = drmFourccForVkFormat(imageFormat);
    if (!fourccOpt) throw std::invalid_argument("unsupported native external image format");

    VulkanNativeExternalImageBacking result;
    result.endpoint = source; result.format = imageFormat; result.extent = size;
    result.fourcc = *fourccOpt; result.modifier = DRM_FORMAT_MOD_LINEAR;
    if (imageFormat != VK_FORMAT_R8G8B8A8_UNORM)
        throw std::invalid_argument("D3A1 negotiated layout supports only R8G8B8A8_UNORM");
    const auto negotiated = discoverConsumerLayout(consumer, size, imageFormat);
    std::cerr << "[DG2X-P4C-D3A1] negotiated layout\n"
        << "  Format: " << imageFormat << "\n"
        << "  Extent: " << size.width << "x" << size.height << "\n"
        << "  Modifier: 0x" << std::hex << negotiated.modifier << std::dec << "\n"
        << "  Plane count: " << negotiated.planes.size() << "\n"
        << "  Plane 0 offset: " << negotiated.planes[0].offset << "\n"
        << "  Plane 0 rowPitch: " << negotiated.planes[0].rowPitch << "\n";
    VkDrmFormatModifierPropertiesListEXT modifierList{
        VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
    VkFormatProperties2 formatProperties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    formatProperties.pNext = &modifierList;
    source.GetPhysicalDeviceFormatProperties2(source.physicalDevice, imageFormat,
        &formatProperties);
    std::vector<VkDrmFormatModifierPropertiesEXT> available(
        modifierList.drmFormatModifierCount);
    modifierList.pDrmFormatModifierProperties = available.data();
    source.GetPhysicalDeviceFormatProperties2(source.physicalDevice, imageFormat,
        &formatProperties);
    const auto candidate = std::find_if(available.begin(), available.end(), [](const auto& item) {
        constexpr VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return item.drmFormatModifier == DRM_FORMAT_MOD_LINEAR
            && (item.drmFormatModifierTilingFeatures & required) == required;
    });
    if (candidate == available.end())
        throw ls::vulkan_error("linear DMA-BUF modifier lacks transfer support");
    const uint32_t selectedPlaneCount = static_cast<uint32_t>(negotiated.planes.size());
    VkPhysicalDeviceExternalImageFormatInfo externalQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    externalQuery.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT};
    modifierQuery.pNext = &externalQuery; modifierQuery.drmFormatModifier = result.modifier;
    modifierQuery.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkPhysicalDeviceImageFormatInfo2 imageQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    imageQuery.pNext = &modifierQuery; imageQuery.format = imageFormat;
    imageQuery.type = VK_IMAGE_TYPE_2D; imageQuery.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    imageQuery.usage = reverseUsage;
    VkExternalImageFormatProperties externalProperties{
        VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 imageProperties{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    imageProperties.pNext = &externalProperties;
    auto capabilityResult = source.GetPhysicalDeviceImageFormatProperties2(
        source.physicalDevice, &imageQuery, &imageProperties);
    const auto requiredFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT
        | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    if (capabilityResult != VK_SUCCESS
            || (externalProperties.externalMemoryProperties.externalMemoryFeatures
                & requiredFeatures) != requiredFeatures)
        throw ls::vulkan_error("linear DMA-BUF image is not importable/exportable");
    const auto explicitPlanes = normalizeExplicitImagePlaneLayouts(negotiated.planes, 1, 1);
    VkImageDrmFormatModifierExplicitCreateInfoEXT modifier{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
    modifier.drmFormatModifier = negotiated.modifier;
    modifier.drmFormatModifierPlaneCount = selectedPlaneCount;
    modifier.pPlaneLayouts = explicitPlanes.data();
    VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    modifier.pNext = &external;
    VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create.pNext = &modifier; create.imageType = VK_IMAGE_TYPE_2D; create.format = imageFormat;
    create.extent = {size.width, size.height, 1}; create.mipLevels = 1; create.arrayLayers = 1;
    create.samples = VK_SAMPLE_COUNT_1_BIT; create.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    create.usage = reverseUsage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE; create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    auto r = source.CreateImage(source.bufferDevice.device, &create, nullptr, &result.imageHandle);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "native external image creation failed");
    VkImageDrmFormatModifierPropertiesEXT materialized{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
    r = source.GetImageDrmFormatModifierPropertiesEXT(source.bufferDevice.device,
        result.imageHandle, &materialized);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "native modifier query failed");
    if (materialized.drmFormatModifier != negotiated.modifier)
        throw ls::vulkan_error("D3A1 exporter materialized unexpected DRM modifier");
    if (selectedPlaneCount == 0 || selectedPlaneCount > std::size(planeAspects))
        throw ls::vulkan_error("unsupported DRM modifier plane count");
    for (uint32_t i = 0; i < selectedPlaneCount; ++i) {
        VkImageSubresource subresource{planeAspects[i], 0, 0};
        VkSubresourceLayout plane{};
        source.GetImageSubresourceLayout(source.bufferDevice.device, result.imageHandle,
            &subresource, &plane);
        if (plane.rowPitch == 0) throw ls::vulkan_error("invalid exported DRM plane layout");
        result.planes.push_back(plane);
    }
    if (!explicitDrmImageLayoutMatches(negotiated, materialized.drmFormatModifier,
            result.planes))
        throw ls::vulkan_error("D3A1 exporter did not materialize negotiated layout");
    VkMemoryDedicatedRequirements dedicatedReq{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkImageMemoryRequirementsInfo2 query{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    query.image = result.imageHandle;
    VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    requirements.pNext = &dedicatedReq;
    source.GetImageMemoryRequirements2(source.bufferDevice.device, &query, &requirements);
    result.allocationSize = requirements.memoryRequirements.size;
    result.dedicated = dedicatedReq.requiresDedicatedAllocation == VK_TRUE
        || (externalProperties.externalMemoryProperties.externalMemoryFeatures
            & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    uint32_t type = 0;
    while (type < source.bufferDevice.memoryProperties.memoryTypeCount
            && !(requirements.memoryRequirements.memoryTypeBits & (1u << type))) ++type;
    if (type == source.bufferDevice.memoryProperties.memoryTypeCount)
        throw ls::vulkan_error("no native external image memory type");
    VkMemoryDedicatedAllocateInfo dedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.image = result.dedicated ? result.imageHandle : VK_NULL_HANDLE;
    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    exportInfo.pNext = result.dedicated ? &dedicatedInfo : nullptr;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &exportInfo; allocation.allocationSize = result.allocationSize;
    allocation.memoryTypeIndex = type;
    r = source.AllocateMemory(source.bufferDevice.device, &allocation, nullptr, &result.memoryHandle);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "native external image allocation failed");
    r = source.BindImageMemory(source.bufferDevice.device, result.imageHandle, result.memoryHandle, 0);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "native external image bind failed");
    result.modifier = materialized.drmFormatModifier;
    std::cerr << "[DG2X-P4C-D3A1] materialized B layout\n"
        << "  Modifier: 0x" << std::hex << result.modifier << std::dec << "\n"
        << "  Plane count: " << result.planes.size() << "\n"
        << "  Plane 0 offset: " << result.planes[0].offset << "\n"
        << "  Plane 0 rowPitch: " << result.planes[0].rowPitch << "\n"
        << "  Plane 0 queried size: " << result.planes[0].size << "\n"
        << "  Plane 0 arrayPitch: " << result.planes[0].arrayPitch << "\n"
        << "  Plane 0 depthPitch: " << result.planes[0].depthPitch << "\n"
        << "  Requirements size: " << result.allocationSize << "\n"
        << "  Requirements alignment: " << requirements.memoryRequirements.alignment << "\n"
        << "  Requirements memoryTypeBits: 0x" << std::hex
        << requirements.memoryRequirements.memoryTypeBits << std::dec << "\n"
        << "  Selected memory type: " << type << "\n"
        << "  Dedicated required: " << dedicatedReq.requiresDedicatedAllocation << "\n"
        << "  Dedicated preferred: " << dedicatedReq.prefersDedicatedAllocation << "\n";
    return result;
}

VulkanNativeExternalImageDescriptor VulkanNativeExternalImageBacking::exportDescriptor() {
    if (!memoryHandle) throw std::logic_error("native image backing already exported");
    VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = memoryHandle; info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    int fd = -1;
    auto r = endpoint.GetMemoryFdKHR(endpoint.bufferDevice.device, &info, &fd);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "native DMA-BUF export failed");
    return {ls::OwnedFd(fd), format, extent, fourcc, modifier, planes, allocationSize};
}
