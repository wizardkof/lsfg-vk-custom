/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../helpers/owned_fd.hpp"
#include "../helpers/pointers.hpp"
#include "vulkan.hpp"

#include <functional>
#include <optional>
#include <span>

#include <vulkan/vulkan_core.h>

namespace vk {
    /// Vulkan descriptor shared by both sides of an OPAQUE_FD image exchange.
    struct ExternalImageDescriptor {
        VkImageCreateFlags flags{};
        VkImageType imageType{VK_IMAGE_TYPE_2D};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkExtent3D extent{};
        uint32_t mipLevels{1};
        uint32_t arrayLayers{1};
        VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
        VkImageTiling tiling{VK_IMAGE_TILING_OPTIMAL};
        VkImageUsageFlags usage{};
        VkSharingMode sharingMode{VK_SHARING_MODE_EXCLUSIVE};
        VkExternalMemoryHandleTypeFlagBits handleType{
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};

        [[nodiscard]] bool operator==(const ExternalImageDescriptor& other) const;
    };

    /// Allocation values that must be preserved when importing Vulkan OPAQUE_FD memory.
    struct ExternalImageAllocation {
        VkDeviceSize allocationSize{};
        uint32_t memoryTypeIndex{};

        bool operator==(const ExternalImageAllocation&) const = default;
    };

    /// Complete by-value description of one exported exchange image.
    struct ExternalImage {
        ls::OwnedFd fd{};
        ExternalImageDescriptor descriptor{};
        ExternalImageAllocation allocation{};
    };

    using ImageMemoryRequirementsProvider = std::function<VkMemoryRequirements()>;
    using ImageMemoryTypeSelector = std::function<std::optional<uint32_t>(uint32_t)>;
    using MemoryFdImporter = std::function<VkResult(int)>;

    /// Make the canonical source exchange descriptor.
    [[nodiscard]] ExternalImageDescriptor makeSourceExchangeImageDescriptor(
        VkExtent2D extent, VkFormat format);
    /// Make the canonical destination exchange descriptor.
    [[nodiscard]] ExternalImageDescriptor makeDestinationExchangeImageDescriptor(
        VkExtent2D extent, VkFormat format);
    /// Resolve allocation metadata, preserving imported metadata without querying locally.
    [[nodiscard]] ExternalImageAllocation resolveImageMemoryAllocation(
        std::optional<ExternalImageAllocation> importedAllocation,
        const ImageMemoryRequirementsProvider& requirementsProvider,
        const ImageMemoryTypeSelector& memoryTypeSelector);
    /// Borrow an owned FD for import and relinquish it only after successful import.
    [[nodiscard]] VkResult importMemoryWithOwnedFd(
        ls::OwnedFd& fd, const MemoryFdImporter& importer);
    /// Make the dedicated-allocation association for an image.
    [[nodiscard]] VkMemoryDedicatedAllocateInfo makeDedicatedImageAllocateInfo(VkImage image);

    /// optional creation details for non-standard images
    struct ImageCreateOptions {
        VkImageCreateFlags flags{};
        VkSharingMode sharingMode{VK_SHARING_MODE_EXCLUSIVE};
        std::span<const uint32_t> queueFamilyIndices{};
        const void* pNext{};
    };

    /// vulkan image
    class Image {
    public:
        /// create an image
        /// @param vk the vulkan instance
        /// @param extent extent of the image in pixels
        /// @param format vulkan format of the image
        /// @param usage usage flags
        /// @throws ls::vulkan_error on failure
        Image(const vk::Vulkan& vk,
            VkExtent2D extent,
            VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
            VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        /// create an image with explicit Vulkan image creation options
        /// @param vk the vulkan instance
        /// @param extent extent of the image in pixels
        /// @param format vulkan format
        /// @param usage usage flags
        /// @param options image flags/sharing/pNext details
        /// @throws ls::vulkan_error on failure
        Image(const vk::Vulkan& vk,
            VkExtent2D extent,
            VkFormat format,
            VkImageUsageFlags usage,
            const ImageCreateOptions& options);

        /// create and export a dedicated OPAQUE_FD exchange image
        Image(const vk::Vulkan& vk,
            const ExternalImageDescriptor& descriptor,
            ExternalImage& exportedImage);

        /// import and consume a dedicated OPAQUE_FD image using its original allocation metadata
        Image(const vk::Vulkan& vk, ExternalImage&& importedImage);

        /// get the image handle
        /// @return the image handle
        [[nodiscard]] const auto& handle() const { return this->image.get(); }
        /// get the image view handle
        /// @return the image view handle
        [[nodiscard]] const auto& imageview() const { return this->view.get(); }

        /// get the extent of the image
        /// @return the extent of the image
        [[nodiscard]] VkExtent2D getExtent() const { return this->extent; }
    private:
        ls::owned_ptr<VkImage> image;
        ls::owned_ptr<VkDeviceMemory> memory;
        ls::owned_ptr<VkImageView> view;

        VkExtent2D extent{};
    };
}
