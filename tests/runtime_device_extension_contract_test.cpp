/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_device_extension_contract.hpp"

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace {
    bool contains(const std::vector<const char*>& values, const char* expected) {
        return std::ranges::any_of(values,
            [expected](const char* value) {
                return value && std::string(value) == expected;
            });
    }
}

int main() {
    const std::vector<std::string> complete{
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME
    };
    assert(vk::supportsRuntimeCrossDeviceExtensionContract(complete));

    std::vector<const char*> enabled{VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    vk::appendAdvertisedRuntimeCrossDeviceExtensions(enabled, complete);
    assert(contains(enabled, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME));
    assert(contains(enabled, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME));
    assert(contains(enabled, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME));

    const auto enabledCount = enabled.size();
    vk::appendRuntimeCrossDeviceExtensions(enabled);
    assert(enabled.size() == enabledCount);

    const std::vector<std::string> missingForeign{
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME
    };
    assert(!vk::supportsRuntimeCrossDeviceExtensionContract(missingForeign));
    std::vector<const char*> partial;
    vk::appendAdvertisedRuntimeCrossDeviceExtensions(partial, missingForeign);
    assert(!contains(partial, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME));

    return 0;
}
