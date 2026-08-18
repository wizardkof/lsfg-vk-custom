/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "lsfg-vk-common/vulkan/external_image_transport.hpp"

#include <cassert>
#include <libdrm/drm_fourcc.h>

int main() {
    using namespace vk;
    constexpr VkExternalMemoryHandleTypeFlags handle =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    const ImageTransportCandidate candidate{
        VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        {256, 256, 1}, handle, 0, 1};
    const ImageTransportCapability good{
        VK_SUCCESS, VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
            | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT, handle, {{256, 256, 1}}};
    assert(good.usable(handle));
    assert(chooseImageTransportContract({{candidate, good, good}}).has_value());

    const ImageTransportCapability importOnly{
        VK_SUCCESS, VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT, handle, {{256, 256, 1}}};
    assert(!chooseImageTransportContract({{candidate, importOnly, good}}).has_value());
    assert(!chooseImageTransportContract({}).has_value());

    const std::vector<ImageModifierInfo> amd{{0x20, 1, 0x7}, {0x10, 2, 0x7}, {0x0, 1, 0x3}};
    const std::vector<ImageModifierInfo> nvidia{{0x30, 1, 0x7}, {0x0, 1, 0x7}};
    const auto common = intersectImageModifiers(amd, nvidia);
    assert(common.size() == 1 && common.front().modifier == 0x0);
    assert(filterImageModifiers(common, 0x3).size() == 1);
    assert(selectImageModifier({{0x20, 1, 0x7}, {0x10, 1, 0x7}})->modifier == 0x10);
    assert(intersectImageModifiers({{0x1, 1, 0x7}}, {{0x1, 2, 0x7}}).empty());
    assert(drmFourccForVkFormat(VK_FORMAT_R8G8B8A8_UNORM).value() == DRM_FORMAT_ABGR8888);
    assert(!drmFourccForVkFormat(VK_FORMAT_R32_SFLOAT).has_value());
    assert(validateImagePlaneMetadata(DRM_FORMAT_ABGR8888, 0, 1,
        DRM_FORMAT_ABGR8888, 0, 1, {{.offset = 0, .size = 16 * 256,
            .rowPitch = 16 * 256}}));
    assert(!validateImagePlaneMetadata(DRM_FORMAT_ABGR8888, 0, 1,
        DRM_FORMAT_ABGR8888, 1, 1, {{.offset = 0, .size = 16 * 256,
            .rowPitch = 16 * 256}}));
    assert(!imageNeedsDedicatedAllocation(0, {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
        nullptr, VK_FALSE, VK_FALSE}));
    assert(imageNeedsDedicatedAllocation(0, {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
        nullptr, VK_FALSE, VK_TRUE}));
    assert(imageNeedsDedicatedAllocation(VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
        {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_FALSE, VK_FALSE}));
}
