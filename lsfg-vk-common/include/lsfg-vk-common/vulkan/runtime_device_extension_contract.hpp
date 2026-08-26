/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {

    inline constexpr std::array<const char*, 3>
        RUNTIME_CROSS_DEVICE_DEVICE_EXTENSIONS{
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
            VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME
        };

    [[nodiscard]] inline bool runtimeDeviceExtensionAdvertised(
            const std::vector<std::string>& advertised, const char* extension) {
        return std::ranges::find(advertised, std::string(extension)) != advertised.end();
    }

    [[nodiscard]] inline bool supportsRuntimeCrossDeviceExtensionContract(
            const std::vector<std::string>& advertised) {
        return std::ranges::all_of(RUNTIME_CROSS_DEVICE_DEVICE_EXTENSIONS,
            [&advertised](const char* extension) {
                return runtimeDeviceExtensionAdvertised(advertised, extension);
            });
    }

    inline void appendRuntimeDeviceExtension(
            std::vector<const char*>& enabled, const char* extension) {
        const auto alreadyEnabled = std::ranges::any_of(enabled,
            [extension](const char* candidate) {
                return candidate && std::string(candidate) == extension;
            });
        if (!alreadyEnabled)
            enabled.push_back(extension);
    }

    inline void appendRuntimeCrossDeviceExtensions(
            std::vector<const char*>& enabled) {
        for (const auto* extension : RUNTIME_CROSS_DEVICE_DEVICE_EXTENSIONS)
            appendRuntimeDeviceExtension(enabled, extension);
    }

    inline void appendAdvertisedRuntimeCrossDeviceExtensions(
            std::vector<const char*>& enabled,
            const std::vector<std::string>& advertised) {
        for (const auto* extension : RUNTIME_CROSS_DEVICE_DEVICE_EXTENSIONS) {
            if (runtimeDeviceExtensionAdvertised(advertised, extension))
                appendRuntimeDeviceExtension(enabled, extension);
        }
    }

}
