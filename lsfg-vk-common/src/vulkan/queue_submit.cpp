/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/queue_submit.hpp"

#include <stdexcept>

namespace vk {

VkResult executeQueueSubmit(PFN_vkQueueSubmit dispatch, VkQueue queue,
        uint32_t submitCount, const VkSubmitInfo* submits, VkFence fence,
        const QueueSubmitObserver& observer,
        const QueueSubmitResultObserver& resultObserver) {
    if (!dispatch || queue == VK_NULL_HANDLE || (submitCount != 0 && !submits))
        throw std::invalid_argument("invalid queue submit execution request");
    if (observer)
        observer(queue, submitCount, submits, fence);
    const auto result = dispatch(queue, submitCount, submits, fence);
    if (resultObserver)
        resultObserver(queue, submitCount, submits, fence, result);
    return result;
}

}
