#include "d3b1_present_path.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"

#include <stdexcept>

namespace lsfgvk::layer {
namespace {
    vk::Barrier barrier(VkImage image, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
            VkImageLayout oldLayout, VkImageLayout newLayout,
            uint32_t srcFamily = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstFamily = VK_QUEUE_FAMILY_IGNORED) {
        return vk::makeImageBarrier(image, srcAccess, dstAccess, oldLayout, newLayout,
            srcFamily, dstFamily, vk::exchangeImageSubresourceRange());
    }
}

VkResult executeD3B1PresentPath(D3B1PresentPath& path) {
    if (!path.state || *path.state != D3B1PresentationState::A_HANDOFF_SUBMITTED)
        throw std::logic_error("D3B1 presentation state is not eligible");
    bool submitted = false;
    bool retired = false;
    const auto retire = [&] {
        if (!retired) { path.retire(); retired = true; }
    };
    try {
        if (path.source.image == VK_NULL_HANDLE
                || path.source.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                || path.source.destinationQueueFamily == VK_QUEUE_FAMILY_IGNORED
                || path.hiddenAcquire == VK_NULL_HANDLE
                || path.returnedForPresent == VK_NULL_HANDLE
                || path.finalPresentSemaphore == VK_NULL_HANDLE
                || path.renderFence == VK_NULL_HANDLE
                || path.commandPoolFamily == VK_QUEUE_FAMILY_IGNORED
                || path.commandPoolFamily != path.submitQueueFamily
                || (path.submitQueueFlags & VK_QUEUE_GRAPHICS_BIT) == 0
                || !path.sourceBlitSupported || !path.destinationBlitSupported)
            throw std::runtime_error("D3B1 returned-source blit preflight failed");

        const auto hidden = path.acquireHidden();
        if (hidden.image == VK_NULL_HANDLE)
            throw std::runtime_error("D3B1 hidden WSI acquire failed");
        std::vector<vk::Barrier> pre;
        if (path.source.ownershipAcquireRequired) {
            pre.push_back(barrier(path.source.image, 0, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                path.source.sourceQueueFamily, path.source.destinationQueueFamily));
        }
        pre.push_back(barrier(hidden.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        const std::vector<vk::Barrier> post{barrier(hidden.image,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)};
        path.recordBlit(pre, path.source.image, hidden.image,
            path.source.extent, hidden.extent, post);
        const std::vector<VkSemaphore> waits{path.hiddenAcquire, path.returnedForPresent};
        const std::vector<VkPipelineStageFlags> stages(
            waits.size(), VK_PIPELINE_STAGE_TRANSFER_BIT);
        const auto submitPresent = path.submitAndPresent(waits, stages,
            path.finalPresentSemaphore, path.renderFence, hidden.index);
        submitted = submitPresent.offloadSubmitted;
        if (submitted) *path.state = D3B1PresentationState::OFFLOAD_SUBMITTED;
        *path.state = D3B1PresentationState::PRESENT_SUBMITTED;
        if (submitPresent.presentResult != VK_SUCCESS)
            throw ls::vulkan_error(submitPresent.presentResult,
                "D3B1 hidden WSI present failed");
        *path.state = D3B1PresentationState::PRESENT_SUCCESS;
        if (!path.waitRenderFence())
            throw ls::vulkan_error(VK_TIMEOUT, "D3B1 offload completion failed");
        submitted = false;
        *path.state = D3B1PresentationState::OFFLOAD_COMPLETED;
        if (!path.completeDiagnostics())
            throw std::runtime_error("D3B1 terminal diagnostics did not pass");
        retire();
        *path.state = D3B1PresentationState::PASS;
        path.emitMarker();
        return VK_SUCCESS;
    } catch (...) {
        *path.state = D3B1PresentationState::FAILED;
        if (submitted) {
            try { static_cast<void>(path.waitRenderFence()); }
            catch (...) {}
        }
        retire();
        throw;
    }
}

VkResult submitD3B1PresentNonblocking(
        D3B1PresentPath& path, D3B1PendingPresent& pending) {
    if (!path.state || *path.state != D3B1PresentationState::A_HANDOFF_SUBMITTED
            || pending.submitAccepted || pending.completed || !path.tryRetireRenderFence)
        throw std::logic_error("D3B1 nonblocking presentation is not submit-ready");
    try {
        if (path.source.image == VK_NULL_HANDLE
                || path.source.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                || path.source.destinationQueueFamily == VK_QUEUE_FAMILY_IGNORED
                || path.hiddenAcquire == VK_NULL_HANDLE
                || path.returnedForPresent == VK_NULL_HANDLE
                || path.finalPresentSemaphore == VK_NULL_HANDLE
                || path.renderFence == VK_NULL_HANDLE
                || path.commandPoolFamily == VK_QUEUE_FAMILY_IGNORED
                || path.commandPoolFamily != path.submitQueueFamily
                || (path.submitQueueFlags & VK_QUEUE_GRAPHICS_BIT) == 0
                || !path.sourceBlitSupported || !path.destinationBlitSupported)
            throw std::runtime_error("D3B1 returned-source blit preflight failed");
        const auto hidden = path.acquireHidden();
        if (hidden.image == VK_NULL_HANDLE)
            throw std::runtime_error("D3B1 hidden WSI acquire failed");
        std::vector<vk::Barrier> pre;
        if (path.source.ownershipAcquireRequired)
            pre.push_back(barrier(path.source.image, 0, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                path.source.sourceQueueFamily, path.source.destinationQueueFamily));
        pre.push_back(barrier(hidden.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        const std::vector<vk::Barrier> post{barrier(hidden.image,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)};
        path.recordBlit(pre, path.source.image, hidden.image,
            path.source.extent, hidden.extent, post);
        const std::vector<VkSemaphore> waits{path.hiddenAcquire, path.returnedForPresent};
        const std::vector<VkPipelineStageFlags> stages(
            waits.size(), VK_PIPELINE_STAGE_TRANSFER_BIT);
        const auto submitPresent = path.submitAndPresent(waits, stages,
            path.finalPresentSemaphore, path.renderFence, hidden.index);
        pending.submitAccepted = submitPresent.offloadSubmitted;
        if (pending.submitAccepted) *path.state = D3B1PresentationState::OFFLOAD_SUBMITTED;
        *path.state = D3B1PresentationState::PRESENT_SUBMITTED;
        pending.presentResult = submitPresent.presentResult;
        if (pending.presentResult != VK_SUCCESS)
            throw ls::vulkan_error(pending.presentResult,
                "D3B1 hidden WSI present failed");
        *path.state = D3B1PresentationState::PRESENT_SUCCESS;
        return pending.presentResult;
    } catch (...) {
        *path.state = D3B1PresentationState::FAILED;
        if (!pending.submitAccepted && path.retire) path.retire();
        throw;
    }
}

D3B1RetirementResult tryRetireD3B1Present(
        D3B1PresentPath& path, D3B1PendingPresent& pending) {
    if (!path.state || !pending.submitAccepted || pending.completed
            || (*path.state != D3B1PresentationState::PRESENT_SUCCESS
                && *path.state != D3B1PresentationState::FAILED)
            || !path.tryRetireRenderFence)
        return D3B1RetirementResult::FAILED;
    VkResult status{};
    try { status = path.tryRetireRenderFence(); }
    catch (...) { *path.state = D3B1PresentationState::FAILED; return D3B1RetirementResult::FAILED; }
    if (status == VK_NOT_READY || status == VK_TIMEOUT)
        return D3B1RetirementResult::NOT_READY;
    if (status == VK_ERROR_DEVICE_LOST) {
        *path.state = D3B1PresentationState::FAILED;
        return D3B1RetirementResult::DEVICE_LOST;
    }
    if (status != VK_SUCCESS) {
        *path.state = D3B1PresentationState::FAILED;
        return D3B1RetirementResult::FAILED;
    }
    try {
        *path.state = D3B1PresentationState::OFFLOAD_COMPLETED;
        if (pending.presentResult != VK_SUCCESS || !path.completeDiagnostics()) {
            *path.state = D3B1PresentationState::FAILED;
            return D3B1RetirementResult::FAILED;
        }
        if (path.retire) path.retire();
        pending.completed = true;
        *path.state = D3B1PresentationState::PASS;
        if (path.emitMarker) path.emitMarker();
        return D3B1RetirementResult::RETIRED;
    } catch (...) {
        *path.state = D3B1PresentationState::FAILED;
        return D3B1RetirementResult::FAILED;
    }
}
}
