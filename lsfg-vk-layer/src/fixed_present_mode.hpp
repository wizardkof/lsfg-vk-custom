/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <vulkan/vulkan_core.h>

#include <optional>
#include <vector>

namespace lsfgvk::layer {

    /// Choose the least-blocking safe WSI mode for asynchronous Fixed output.
    /// creationAllowed is empty when VkSwapchainPresentModesCreateInfoKHR is
    /// absent; otherwise candidates must also be listed there.
    [[nodiscard]] VkPresentModeKHR selectFixedPresentMode(
        const std::vector<VkPresentModeKHR>& supported,
        const std::vector<VkPresentModeKHR>& creationAllowed,
        VkPresentModeKHR originalMode);

    /// Select a Fixed mode from an application-provided dynamic present-mode
    /// declaration that can also support Adaptive FIFO.
    [[nodiscard]] std::optional<VkPresentModeKHR> selectApplicationDualPresentMode(
        const std::vector<VkPresentModeKHR>& creationAllowed,
        VkPresentModeKHR originalMode);

    [[nodiscard]] const char* fixedPresentModeName(VkPresentModeKHR mode);

}
