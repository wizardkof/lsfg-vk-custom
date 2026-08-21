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
    const auto normalized = normalizeExplicitImagePlaneLayouts(
        {{.offset = 17, .size = 4096, .rowPitch = 256,
            .arrayPitch = 8192, .depthPitch = 16384}}, 1, 1);
    assert(normalized.size() == 1);
    assert(normalized[0].offset == 17 && normalized[0].rowPitch == 256);
    assert(normalized[0].size == 0 && normalized[0].arrayPitch == 0
        && normalized[0].depthPitch == 0);
    const auto normalizedPlanes = normalizeExplicitImagePlaneLayouts(
        {{.offset = 1, .size = 2, .rowPitch = 3, .arrayPitch = 4, .depthPitch = 5},
         {.offset = 6, .size = 7, .rowPitch = 8, .arrayPitch = 9, .depthPitch = 10}}, 2, 3);
    assert(normalizedPlanes.size() == 2);
    assert(normalizedPlanes[0].offset == 1 && normalizedPlanes[0].rowPitch == 3
        && normalizedPlanes[0].size == 0 && normalizedPlanes[0].arrayPitch == 4
        && normalizedPlanes[0].depthPitch == 5);
    assert(normalizedPlanes[1].offset == 6 && normalizedPlanes[1].rowPitch == 8
        && normalizedPlanes[1].size == 0 && normalizedPlanes[1].arrayPitch == 9
        && normalizedPlanes[1].depthPitch == 10);
    const ExplicitDrmImageLayout negotiated{0,
        {{.offset = 0, .rowPitch = 2048}}};
    assert(explicitDrmImageLayoutMatches(negotiated, 0,
        {{.offset = 0, .size = 1024000, .rowPitch = 2048,
          .arrayPitch = 1024000, .depthPitch = 1024000}}));
    assert(!explicitDrmImageLayoutMatches(negotiated, 0,
        {{.offset = 0, .rowPitch = 2016}}));
    assert(!explicitDrmImageLayoutMatches(negotiated, 0,
        {{.offset = 256, .rowPitch = 2048}}));
    assert(!explicitDrmImageLayoutMatches(negotiated, 1,
        {{.offset = 0, .rowPitch = 2048}}));
    assert(!explicitDrmImageLayoutMatches(negotiated, 0, {}));
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
