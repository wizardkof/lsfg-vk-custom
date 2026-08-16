/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <bitset>
#include <functional>
#include <optional>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    vk::ExternalImageDescriptor makeExternalImageDescriptor(
            VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
        return {
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = { extent.width, extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        };
    }

    /// create a image
    ls::owned_ptr<VkImage> createImage(const vk::Vulkan& vk,
            const vk::ExternalImageDescriptor& descriptor,
            bool external, const vk::ImageCreateOptions& options) {
        VkImage handle{};

        const VkExternalMemoryImageCreateInfo externalInfo{
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .pNext = options.pNext,
            .handleTypes = descriptor.handleType
        };
        const VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = external ? &externalInfo : options.pNext,
            .flags = descriptor.flags,
            .imageType = descriptor.imageType,
            .format = descriptor.format,
            .extent = descriptor.extent,
            .mipLevels = descriptor.mipLevels,
            .arrayLayers = descriptor.arrayLayers,
            .samples = descriptor.samples,
            .tiling = descriptor.tiling,
            .usage = descriptor.usage,
            .sharingMode = descriptor.sharingMode,
            .queueFamilyIndexCount = static_cast<uint32_t>(options.queueFamilyIndices.size()),
            .pQueueFamilyIndices = options.queueFamilyIndices.empty()
                ? nullptr : options.queueFamilyIndices.data()
        };
        auto res = vk.df().CreateImage(vk.dev(), &imageInfo, VK_NULL_HANDLE, &handle);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkCreateImage() failed");

        return ls::owned_ptr<VkImage>(
            new VkImage(handle),
            [dev = vk.dev(), defunc = vk.df().DestroyImage](VkImage& image) {
                defunc(dev, image, VK_NULL_HANDLE);
            }
        );
    }
    /// allocate memory for a image
    struct AllocatedImageMemory {
        ls::owned_ptr<VkDeviceMemory> memory;
        vk::ExternalImageAllocation metadata;
    };

    AllocatedImageMemory allocateMemory(const vk::Vulkan& vk, VkImage image,
            ls::OwnedFd* importFd,
            std::optional<vk::ExternalImageAllocation> importedAllocation,
            bool exportMemory) {
        VkDeviceMemory handle{};

        const auto allocation = vk::resolveImageMemoryAllocation(
            importedAllocation,
            [&vk, image]() {
                VkMemoryRequirements requirements{};
                vk.df().GetImageMemoryRequirements(vk.dev(), image, &requirements);
                return requirements;
            },
            [&vk](uint32_t memoryTypeBits) {
                return vk.findMemoryTypeIndex(memoryTypeBits, false);
            }
        );

        const auto dedicatedInfo = vk::makeDedicatedImageAllocateInfo(image);
        VkImportMemoryFdInfoKHR importInfo{
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            .pNext = &dedicatedInfo,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
            .fd = importFd == nullptr ? -1 : importFd->get()
        };
        const VkExportMemoryAllocateInfo exportInfo{
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .pNext = &dedicatedInfo,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };
        const void* pNextAlloc{};
        if (importFd != nullptr)
            pNextAlloc = &importInfo;
        else if (exportMemory)
            pNextAlloc = &exportInfo;
        const VkMemoryAllocateInfo memoryInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = pNextAlloc,
            .allocationSize = allocation.allocationSize,
            .memoryTypeIndex = allocation.memoryTypeIndex
        };
        VkResult res{};
        if (importFd != nullptr) {
            res = vk::importMemoryWithOwnedFd(*importFd,
                [&vk, &importInfo, &memoryInfo, &handle](int rawFd) {
                    importInfo.fd = rawFd;
                    return vk.df().AllocateMemory(
                        vk.dev(), &memoryInfo, VK_NULL_HANDLE, &handle);
                });
        } else {
            res = vk.df().AllocateMemory(vk.dev(), &memoryInfo, VK_NULL_HANDLE, &handle);
        }
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkAllocateMemory() failed");

        res = vk.df().BindImageMemory(vk.dev(), image, handle, 0);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkBindImageMemory() failed");

        return {
            .memory = ls::owned_ptr<VkDeviceMemory>(
                new VkDeviceMemory(handle),
                [dev = vk.dev(), defunc = vk.df().FreeMemory](VkDeviceMemory& memory) {
                    defunc(dev, memory, VK_NULL_HANDLE);
                }
            ),
            .metadata = allocation
        };
    }

    ls::OwnedFd exportMemoryFd(const vk::Vulkan& vk, VkDeviceMemory memory) {
        const VkMemoryGetFdInfoKHR fdInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = memory,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };
        int fd{-1};
        const auto res = vk.df().GetMemoryFdKHR(vk.dev(), &fdInfo, &fd);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkGetMemoryFdKHR() failed");
        return ls::OwnedFd(fd);
    }
    /// create an image view
    ls::owned_ptr<VkImageView> createImageView(const vk::Vulkan& vk,
            VkImage image, VkFormat format) {
        VkImageView handle{};

        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        auto res = vk.df().CreateImageView(vk.dev(), &viewInfo, VK_NULL_HANDLE, &handle);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkCreateImageView() failed");

        return ls::owned_ptr<VkImageView>(
            new VkImageView(handle),
            [dev = vk.dev(), defunc = vk.df().DestroyImageView](VkImageView& view) {
                defunc(dev, view, VK_NULL_HANDLE);
            }
        );
    }
}

bool ExternalImageDescriptor::operator==(const ExternalImageDescriptor& other) const {
    return this->flags == other.flags
        && this->imageType == other.imageType
        && this->format == other.format
        && this->extent.width == other.extent.width
        && this->extent.height == other.extent.height
        && this->extent.depth == other.extent.depth
        && this->mipLevels == other.mipLevels
        && this->arrayLayers == other.arrayLayers
        && this->samples == other.samples
        && this->tiling == other.tiling
        && this->usage == other.usage
        && this->sharingMode == other.sharingMode
        && this->handleType == other.handleType;
}

ExternalImageDescriptor vk::makeSourceExchangeImageDescriptor(
        VkExtent2D extent, VkFormat format) {
    return makeExternalImageDescriptor(extent, format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
}

ExternalImageDescriptor vk::makeDestinationExchangeImageDescriptor(
        VkExtent2D extent, VkFormat format) {
    return makeExternalImageDescriptor(extent, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

ExternalImageAllocation vk::resolveImageMemoryAllocation(
        std::optional<ExternalImageAllocation> importedAllocation,
        const ImageMemoryRequirementsProvider& requirementsProvider,
        const ImageMemoryTypeSelector& memoryTypeSelector) {
    if (importedAllocation.has_value())
        return *importedAllocation;

    const auto requirements = requirementsProvider();
    const auto memoryTypeIndex = memoryTypeSelector(requirements.memoryTypeBits);
    if (!memoryTypeIndex.has_value())
        throw ls::vulkan_error("no suitable memory type found for image");

    return {
        .allocationSize = requirements.size,
        .memoryTypeIndex = *memoryTypeIndex
    };
}

VkResult vk::importMemoryWithOwnedFd(
        ls::OwnedFd& fd, const MemoryFdImporter& importer) {
    const auto result = importer(fd.get());
    if (result == VK_SUCCESS)
        static_cast<void>(fd.release());
    return result;
}

VkMemoryDedicatedAllocateInfo vk::makeDedicatedImageAllocateInfo(VkImage image) {
    return {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .image = image
    };
}

Image::Image(const vk::Vulkan& vk,
            VkExtent2D extent,
            VkFormat format,
            VkImageUsageFlags usage) :
        Image(vk, extent, format, usage, ImageCreateOptions{}) {
}

Image::Image(const vk::Vulkan& vk,
            VkExtent2D extent,
            VkFormat format,
            VkImageUsageFlags usage,
            const ImageCreateOptions& options) :
        image(createImage(vk, ExternalImageDescriptor{
                .flags = options.flags,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = format,
                .extent = { extent.width, extent.height, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = usage,
                .sharingMode = options.sharingMode
            }, false, options)),
        extent(extent) {
    auto allocation = allocateMemory(vk, *this->image, nullptr, std::nullopt, false);
    this->memory = std::move(allocation.memory);
    this->view = createImageView(vk, *this->image, format);
}

Image::Image(const vk::Vulkan& vk,
            const ExternalImageDescriptor& descriptor,
            ExternalImage& exportedImage) :
        image(createImage(vk, descriptor, true, ImageCreateOptions{})),
        extent{ descriptor.extent.width, descriptor.extent.height } {
    auto allocation = allocateMemory(vk, *this->image, nullptr, std::nullopt, true);
    this->memory = std::move(allocation.memory);
    this->view = createImageView(vk, *this->image, descriptor.format);
    exportedImage = {
        .fd = exportMemoryFd(vk, *this->memory),
        .descriptor = descriptor,
        .allocation = allocation.metadata
    };
}

Image::Image(const vk::Vulkan& vk, ExternalImage&& importedImage) :
        image(createImage(vk, importedImage.descriptor, true, ImageCreateOptions{})),
        extent{ importedImage.descriptor.extent.width,
            importedImage.descriptor.extent.height } {
    auto allocation = allocateMemory(vk, *this->image,
        &importedImage.fd, importedImage.allocation, false);
    this->memory = std::move(allocation.memory);
    this->view = createImageView(vk, *this->image, importedImage.descriptor.format);
}
