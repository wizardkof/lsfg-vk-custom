/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <vulkan/vulkan_core.h>

namespace vk {

using QueueSubmitObserver = std::function<void(
    VkQueue, uint32_t, const VkSubmitInfo*, VkFence)>;
using QueueSubmitResultObserver = std::function<void(
    VkQueue, uint32_t, const VkSubmitInfo*, VkFence, VkResult)>;

/// Observe and execute an already-finalized Vulkan queue submission without
/// rebuilding or modifying any submit structure.
VkResult executeQueueSubmit(PFN_vkQueueSubmit dispatch, VkQueue queue,
    uint32_t submitCount, const VkSubmitInfo* submits, VkFence fence,
    const QueueSubmitObserver& observer = {},
    const QueueSubmitResultObserver& resultObserver = {});

}
