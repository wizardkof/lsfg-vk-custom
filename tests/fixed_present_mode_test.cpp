/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "../lsfg-vk-layer/src/fixed_present_mode.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using lsfgvk::layer::fixedPresentModeName;
using lsfgvk::layer::selectFixedPresentMode;

int main() {
    {
        const std::vector<VkPresentModeKHR> supported{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_IMMEDIATE_KHR,
            VK_PRESENT_MODE_MAILBOX_KHR,
        };
        assert(selectFixedPresentMode(supported, {}, VK_PRESENT_MODE_FIFO_KHR)
            == VK_PRESENT_MODE_MAILBOX_KHR);
    }

    {
        const std::vector<VkPresentModeKHR> supported{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_IMMEDIATE_KHR,
        };
        assert(selectFixedPresentMode(supported, {}, VK_PRESENT_MODE_FIFO_KHR)
            == VK_PRESENT_MODE_IMMEDIATE_KHR);
    }

    {
        const std::vector<VkPresentModeKHR> supported{VK_PRESENT_MODE_FIFO_KHR};
        assert(selectFixedPresentMode(supported, {}, VK_PRESENT_MODE_FIFO_KHR)
            == VK_PRESENT_MODE_FIFO_KHR);
    }

    {
        // Maintenance1 creation list restricts the mode we are allowed to use.
        const std::vector<VkPresentModeKHR> supported{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_IMMEDIATE_KHR,
            VK_PRESENT_MODE_MAILBOX_KHR,
        };
        const std::vector<VkPresentModeKHR> allowed{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_IMMEDIATE_KHR,
        };
        assert(selectFixedPresentMode(supported, allowed, VK_PRESENT_MODE_FIFO_KHR)
            == VK_PRESENT_MODE_IMMEDIATE_KHR);
    }

    {
        // If FIFO is excluded too, preserve the valid original mode.
        const std::vector<VkPresentModeKHR> supported{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_FIFO_RELAXED_KHR,
        };
        const std::vector<VkPresentModeKHR> allowed{VK_PRESENT_MODE_FIFO_RELAXED_KHR};
        assert(selectFixedPresentMode(
            supported, allowed, VK_PRESENT_MODE_FIFO_RELAXED_KHR)
            == VK_PRESENT_MODE_FIFO_RELAXED_KHR);
    }

    assert(std::string(fixedPresentModeName(VK_PRESENT_MODE_MAILBOX_KHR)) == "MAILBOX");
    assert(std::string(fixedPresentModeName(VK_PRESENT_MODE_IMMEDIATE_KHR)) == "IMMEDIATE");
    assert(std::string(fixedPresentModeName(VK_PRESENT_MODE_FIFO_KHR)) == "FIFO");

    std::cout << "All FixedPresentMode tests passed.\n";
    return 0;
}
