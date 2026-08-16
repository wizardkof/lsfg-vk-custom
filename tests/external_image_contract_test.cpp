/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/image.hpp"

#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <vulkan/vulkan_core.h>

namespace {
    void assertCommonDescriptor(const vk::ExternalImageDescriptor& descriptor,
            VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
        assert(descriptor.flags == 0);
        assert(descriptor.imageType == VK_IMAGE_TYPE_2D);
        assert(descriptor.format == format);
        assert(descriptor.extent.width == extent.width);
        assert(descriptor.extent.height == extent.height);
        assert(descriptor.extent.depth == 1);
        assert(descriptor.mipLevels == 1);
        assert(descriptor.arrayLayers == 1);
        assert(descriptor.samples == VK_SAMPLE_COUNT_1_BIT);
        assert(descriptor.tiling == VK_IMAGE_TILING_OPTIMAL);
        assert(descriptor.usage == usage);
        assert(descriptor.sharingMode == VK_SHARING_MODE_EXCLUSIVE);
        assert(descriptor.handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);
    }

    template<typename Handle>
    Handle fakeHandle(uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>)
            return reinterpret_cast<Handle>(value);
        else
            return static_cast<Handle>(value);
    }
}

int main() {
    static_assert(!std::is_copy_constructible_v<vk::ExternalImage>);
    static_assert(!std::is_copy_assignable_v<vk::ExternalImage>);
    static_assert(std::is_move_constructible_v<vk::ExternalImage>);
    static_assert(std::is_move_assignable_v<vk::ExternalImage>);

    const VkExtent2D extent{ 3440, 1440 };
    const auto sdrSource = vk::makeSourceExchangeImageDescriptor(
        extent, VK_FORMAT_R8G8B8A8_UNORM);
    const auto sdrDestination = vk::makeDestinationExchangeImageDescriptor(
        extent, VK_FORMAT_R8G8B8A8_UNORM);
    const auto hdrSource = vk::makeSourceExchangeImageDescriptor(
        extent, VK_FORMAT_R16G16B16A16_SFLOAT);
    const auto hdrDestination = vk::makeDestinationExchangeImageDescriptor(
        extent, VK_FORMAT_R16G16B16A16_SFLOAT);

    assertCommonDescriptor(sdrSource, extent, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    assertCommonDescriptor(sdrDestination, extent, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    assertCommonDescriptor(hdrSource, extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    assertCommonDescriptor(hdrDestination, extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // The importer consumes the descriptor transported with the FD, not a reconstruction.
    vk::ExternalImage exportedSource{
        .descriptor = hdrSource,
        .allocation = { .allocationSize = 0x12345000, .memoryTypeIndex = 7 }
    };
    const auto expectedDescriptor = exportedSource.descriptor;
    const auto expectedAllocation = exportedSource.allocation;
    const vk::ExternalImage importedSource = std::move(exportedSource);
    assert(importedSource.descriptor == expectedDescriptor);
    assert(importedSource.allocation == expectedAllocation);

    // Valid imported metadata bypasses both requirements lookup and memory-type selection.
    int requirementsCalls{};
    int selectionCalls{};
    const auto importedAllocation = vk::resolveImageMemoryAllocation(
        importedSource.allocation,
        [&requirementsCalls]() {
            ++requirementsCalls;
            return VkMemoryRequirements{};
        },
        [&selectionCalls](uint32_t) -> std::optional<uint32_t> {
            ++selectionCalls;
            return 0;
        });
    assert(importedAllocation == importedSource.allocation);
    assert(requirementsCalls == 0);
    assert(selectionCalls == 0);

    // Export allocation still uses the actual Vulkan requirements and selected type.
    const auto exportedAllocation = vk::resolveImageMemoryAllocation(
        std::nullopt,
        [&requirementsCalls]() {
            ++requirementsCalls;
            return VkMemoryRequirements{
                .size = 0xABC000,
                .alignment = 0x1000,
                .memoryTypeBits = 0x24
            };
        },
        [&selectionCalls](uint32_t memoryTypeBits) -> std::optional<uint32_t> {
            ++selectionCalls;
            assert(memoryTypeBits == 0x24);
            return 5;
        });
    assert(exportedAllocation.allocationSize == 0xABC000);
    assert(exportedAllocation.memoryTypeIndex == 5);
    assert(requirementsCalls == 1);
    assert(selectionCalls == 1);

    const VkImage image = fakeHandle<VkImage>(0x1234);
    const auto dedicatedInfo = vk::makeDedicatedImageAllocateInfo(image);
    assert(dedicatedInfo.sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    assert(dedicatedInfo.image == image);
    assert(dedicatedInfo.buffer == VK_NULL_HANDLE);

    return 0;
}
