/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {
enum class PresentResultClass : unsigned char { PreEnqueueFailure, EnqueuedRejection,
    DeviceLost, Indeterminate, Normal };

[[nodiscard]] constexpr PresentResultClass classifyPresentResult(VkResult result) noexcept {
    switch (result) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return PresentResultClass::PreEnqueueFailure;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
    case VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT:
        return PresentResultClass::EnqueuedRejection;
    case VK_ERROR_DEVICE_LOST:
        return PresentResultClass::DeviceLost;
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
        return PresentResultClass::Normal;
    default:
        return PresentResultClass::Indeterminate;
    }
}

/// Emulates only the application-present-fence portion of a synthetic
/// Class-B abort after application waits have already been consumed. The
/// caller supplies the exact queue serialization boundary used by production.
template<class QueueSubmit>
VkResult compensateEnqueuedPresentAbort(VkQueue queue, VkFence applicationFence,
        QueueSubmit&& queueSubmit) {
    if (applicationFence == VK_NULL_HANDLE) return VK_SUCCESS;
    const VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
    return queueSubmit(queue, submitInfo, applicationFence);
}
}
