/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "fixed_present_mode.hpp"

#include <algorithm>
#include <array>

using namespace lsfgvk::layer;

namespace {
    [[nodiscard]] bool contains(
            const std::vector<VkPresentModeKHR>& modes,
            VkPresentModeKHR mode) {
        return std::find(modes.begin(), modes.end(), mode) != modes.end();
    }
}

VkPresentModeKHR lsfgvk::layer::selectFixedPresentMode(
        const std::vector<VkPresentModeKHR>& supported,
        const std::vector<VkPresentModeKHR>& creationAllowed,
        VkPresentModeKHR originalMode) {
    const auto usable = [&](VkPresentModeKHR mode) {
        return contains(supported, mode)
            && (creationAllowed.empty() || contains(creationAllowed, mode));
    };

    // MAILBOX is preferred because it avoids FIFO backlog while remaining
    // tear-free. IMMEDIATE removes vblank waiting too, but may visibly tear.
    constexpr std::array preferred{
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR,
        VK_PRESENT_MODE_FIFO_KHR,
    };
    for (const auto mode : preferred) {
        if (usable(mode))
            return mode;
    }

    // A restrictive swapchain-maintenance pNext may exclude FIFO. Keep the
    // application's original mode rather than creating an invalid swapchain.
    if (usable(originalMode))
        return originalMode;

    // FIFO is required by VK_KHR_surface, so this should only be reachable for
    // malformed/inconsistent input. Keep a deterministic conservative value.
    return VK_PRESENT_MODE_FIFO_KHR;
}

const char* lsfgvk::layer::fixedPresentModeName(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            return "IMMEDIATE";
        case VK_PRESENT_MODE_MAILBOX_KHR:
            return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_KHR:
            return "FIFO";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            return "FIFO_RELAXED";
#ifdef VK_PRESENT_MODE_FIFO_LATEST_READY_KHR
        case VK_PRESENT_MODE_FIFO_LATEST_READY_KHR:
            return "FIFO_LATEST_READY";
#endif
        default:
            return "OTHER";
    }
}
