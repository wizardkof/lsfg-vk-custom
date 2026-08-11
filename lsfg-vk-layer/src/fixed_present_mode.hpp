/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <vulkan/vulkan_core.h>

#include <vector>

namespace lsfgvk::layer {

    /// Choose the least-blocking safe WSI mode for asynchronous Fixed output.
    /// creationAllowed is empty when VkSwapchainPresentModesCreateInfoKHR is
    /// absent; otherwise candidates must also be listed there.
    [[nodiscard]] VkPresentModeKHR selectFixedPresentMode(
        const std::vector<VkPresentModeKHR>& supported,
        const std::vector<VkPresentModeKHR>& creationAllowed,
        VkPresentModeKHR originalMode);

    [[nodiscard]] const char* fixedPresentModeName(VkPresentModeKHR mode);

}
