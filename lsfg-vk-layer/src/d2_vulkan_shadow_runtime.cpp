/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d2_vulkan_shadow_runtime.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <unistd.h>

namespace lsfgvk::layer {

namespace {
VkSemaphore createSemaphore(const vk::Vulkan& vulkan, bool exportSyncFd) {
    const VkExportSemaphoreCreateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    const VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = exportSyncFd ? &exportInfo : nullptr};
    VkSemaphore semaphore{};
    const auto result = vulkan.df().CreateSemaphore(
        vulkan.dev(), &info, nullptr, &semaphore);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "D2 vkCreateSemaphore failed");
    return semaphore;
}

VkFence createFence(const vk::Vulkan& vulkan, bool exportSyncFd) {
    const VkExportFenceCreateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
    const VkFenceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = exportSyncFd ? &exportInfo : nullptr};
    VkFence fence{};
    const auto result = vulkan.df().CreateFence(vulkan.dev(), &info, nullptr, &fence);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "D2 vkCreateFence failed");
    return fence;
}
}

struct D2VulkanShadowRuntime::Transaction {
    const vk::Vulkan* vulkan{};
    VkSemaphore acquireSemaphore{};
    VkFence acquireFence{};
    VkSemaphore completionSemaphore{};
    VkFence completionFence{};
    VkSemaphore retirementSemaphore{};
    VkFence retirementFence{};
    std::unique_ptr<vk::CommandBuffer> commandBuffer;
    bool submitted{};
    bool retirementSubmitted{};

    ~Transaction() {
        if (!vulkan) return;
        commandBuffer.reset();
        const auto& df = vulkan->df();
        if (completionFence) df.DestroyFence(vulkan->dev(), completionFence, nullptr);
        if (acquireFence) df.DestroyFence(vulkan->dev(), acquireFence, nullptr);
        if (retirementFence) df.DestroyFence(vulkan->dev(), retirementFence, nullptr);
        if (completionSemaphore)
            df.DestroySemaphore(vulkan->dev(), completionSemaphore, nullptr);
        if (retirementSemaphore)
            df.DestroySemaphore(vulkan->dev(), retirementSemaphore, nullptr);
        if (acquireSemaphore)
            df.DestroySemaphore(vulkan->dev(), acquireSemaphore, nullptr);
    }
};

D2VulkanShadowRuntime::D2VulkanShadowRuntime(const vk::Vulkan& vulkan,
        VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
        std::span<const VkImage> realImages, VkFormat format, VkExtent2D extent)
    : vulkan(&vulkan), queue(queue), queueMutex(std::move(queueMutex)),
      realImages(realImages.begin(), realImages.end()),
      shadowInitialized(realImages.size()),
      preparedHiddenByImage(realImages.size()) {
    shadows.reserve(realImages.size());
    for (size_t i = 0; i < realImages.size(); ++i)
        shadows.push_back(std::make_unique<vk::Image>(vulkan, extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            vk::ImageCreateOptions{.arrayLayers = 1}));
}

D2VulkanShadowRuntime::~D2VulkanShadowRuntime() = default;

size_t D2VulkanShadowRuntime::shadowCount() const noexcept { return shadows.size(); }

VkImage D2VulkanShadowRuntime::shadowImage(uint32_t index) const noexcept {
    return index < shadows.size() ? shadows[index]->handle() : VK_NULL_HANDLE;
}

std::shared_ptr<D2VulkanShadowRuntime::Transaction>
D2VulkanShadowRuntime::prepareTransaction(
        bool exportSemaphore, bool exportFence) const {
    auto transaction = std::make_shared<Transaction>();
    transaction->vulkan = vulkan;
    transaction->acquireSemaphore = createSemaphore(*vulkan, false);
    transaction->acquireFence = createFence(*vulkan, false);
    if (exportSemaphore)
        transaction->completionSemaphore = createSemaphore(*vulkan, true);
    if (exportFence)
        transaction->completionFence = createFence(*vulkan, false);
    transaction->retirementSemaphore = createSemaphore(*vulkan, false);
    transaction->retirementFence = createFence(*vulkan, false);
    transaction->commandBuffer = std::make_unique<vk::CommandBuffer>(*vulkan);
    return transaction;
}

VkResult D2VulkanShadowRuntime::recordCopy(Transaction& transaction,
        VkImage source, VkImage destination, bool save) const {
    try {
        transaction.commandBuffer->begin(*vulkan);
        bool initialized{};
        if (save)
            for (size_t i = 0; i < shadows.size(); ++i)
                if (shadows[i]->handle() == destination) {
                    initialized = shadowInitialized[i];
                    break;
                }
        const VkImageMemoryBarrier pre[2]{
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
             .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
             .dstAccessMask = save ? VK_ACCESS_TRANSFER_READ_BIT
                                   : VK_ACCESS_TRANSFER_WRITE_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
             .newLayout = save ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                               : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = save ? source : destination,
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
             .srcAccessMask = save
                 ? (initialized ? VK_ACCESS_TRANSFER_READ_BIT : 0u)
                 : VK_ACCESS_TRANSFER_READ_BIT,
             .dstAccessMask = save ? VK_ACCESS_TRANSFER_WRITE_BIT
                                   : VK_ACCESS_TRANSFER_READ_BIT,
             .oldLayout = save
                 ? (initialized ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                : VK_IMAGE_LAYOUT_UNDEFINED)
                               : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
             .newLayout = save ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                               : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = save ? destination : source,
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}};
        vulkan->df().CmdPipelineBarrier(transaction.commandBuffer->handle(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 2, pre);
        const VkImageCopy region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .extent = {shadows.front()->getExtent().width,
                       shadows.front()->getExtent().height, 1}};
        vulkan->df().CmdCopyImage(transaction.commandBuffer->handle(), source,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier post{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = save ? VK_ACCESS_TRANSFER_READ_BIT
                                  : VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = save ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                              : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = save ? source : destination,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vulkan->df().CmdPipelineBarrier(transaction.commandBuffer->handle(),
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &post);
        if (save) {
            VkImageMemoryBarrier shadowPost{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = destination,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            vulkan->df().CmdPipelineBarrier(transaction.commandBuffer->handle(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &shadowPost);
        }
        transaction.commandBuffer->end(*vulkan);
        return VK_SUCCESS;
    } catch (const ls::vulkan_error& error) {
        return error.error();
    } catch (...) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

VkResult D2VulkanShadowRuntime::submit(Transaction& transaction,
        VkSemaphore waitSemaphore, bool copy) const {
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSemaphore signals[2]{transaction.completionSemaphore,
        transaction.retirementSemaphore};
    const uint32_t signalCount = transaction.completionSemaphore ? 2u : 1u;
    const VkCommandBuffer commandBuffer = transaction.commandBuffer->handle();
    const VkSubmitInfo first{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = waitSemaphore ? 1u : 0u,
        .pWaitSemaphores = waitSemaphore ? &waitSemaphore : nullptr,
        .pWaitDstStageMask = waitSemaphore ? &waitStage : nullptr,
        .commandBufferCount = copy ? 1u : 0u,
        .pCommandBuffers = copy ? &commandBuffer : nullptr,
        .signalSemaphoreCount = signalCount,
        .pSignalSemaphores = transaction.completionSemaphore ? signals
            : &transaction.retirementSemaphore};
    const VkPipelineStageFlags retireStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    const VkSubmitInfo retire{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &transaction.retirementSemaphore,
        .pWaitDstStageMask = &retireStage};
    const std::scoped_lock lock(*queueMutex);
    auto result = vulkan->df().QueueSubmit(
        queue, 1, &first, transaction.completionFence);
    if (result != VK_SUCCESS) return result;
    transaction.submitted = true;
    result = vulkan->df().QueueSubmit(queue, 1, &retire, transaction.retirementFence);
    if (result == VK_SUCCESS) transaction.retirementSubmitted = true;
    return result;
}

void D2VulkanShadowRuntime::reapCompletedTransactions() noexcept {
    const auto getFenceStatus = vulkan->df().GetFenceStatus;
    if (!getFenceStatus) return;

    auto prepared = [&](const Transaction* transaction) noexcept {
        for (const auto* value : preparedHiddenByImage)
            if (value == transaction) return true;
        return false;
    };

    // This is opportunistic observation, not a completion wait: each fence is
    // queried at most once per API entry and VK_NOT_READY immediately leaves
    // that transaction retained.  The already-submitted retirement fence is
    // the exact authority for destroying the command/sync objects.
    for (auto it = retainedTransactions.begin(); it != retainedTransactions.end();) {
        auto* transaction = it->get();
        if (!transaction->retirementSubmitted || prepared(transaction)
                || it->use_count() != 1
                || getFenceStatus(vulkan->dev(), transaction->retirementFence)
                    != VK_SUCCESS) {
            ++it;
            continue;
        }
        it = retainedTransactions.erase(it);
    }
}

VkResult D2VulkanShadowRuntime::handoff(Transaction& transaction,
        VkSemaphore applicationSemaphore, VkFence applicationFence) const {
    if (applicationSemaphore) {
        int fd{-1};
        const VkSemaphoreGetFdInfoKHR exportInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .semaphore = transaction.completionSemaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        auto result = vulkan->df().GetSemaphoreFdKHR(
            vulkan->dev(), &exportInfo, &fd);
        if (result != VK_SUCCESS) return result;
        const VkImportSemaphoreFdInfoKHR importInfo{
            .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            .semaphore = applicationSemaphore,
            .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            .fd = fd};
        result = vulkan->df().ImportSemaphoreFdKHR(vulkan->dev(), &importInfo);
        if (result != VK_SUCCESS && fd >= 0) ::close(fd);
        if (result != VK_SUCCESS) return result;
    }
    (void)applicationFence;
    return VK_SUCCESS;
}

VkResult D2VulkanShadowRuntime::publicAcquire(D2RealWsiState& state,
        VkSemaphore applicationSemaphore, VkFence applicationFence,
        uint32_t* imageIndex, const DownstreamAcquire& downstream,
        const DownstreamRelease& release,
        AcquireFenceProxyRegistry* proxies) noexcept {
    reapCompletedTransactions();
    if (applicationFence != VK_NULL_HANDLE && !proxies)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<Transaction> transaction;
    try {
        transaction = prepareTransaction(
            applicationSemaphore != VK_NULL_HANDLE,
            true);
    } catch (const ls::vulkan_error& error) { return error.error(); }
      catch (...) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    try { retainedTransactions.push_back(std::move(transaction)); }
    catch (...) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    auto& retained = *retainedTransactions.back();

    std::optional<AcquireFenceProxyRegistry::Prepared> preparedProxy;
    if (applicationFence) {
        preparedProxy = proxies->prepare(applicationFence,
            retained.completionFence, retainedTransactions.back());
        if (!preparedProxy) {
            retainedTransactions.pop_back();
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    uint32_t index{};
    const auto downstreamResult = downstream(
        retained.acquireSemaphore, retained.acquireFence, &index);
    if (downstreamResult != VK_SUCCESS && downstreamResult != VK_SUBOPTIMAL_KHR) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        retainedTransactions.pop_back();
        return downstreamResult;
    }
    if (index >= realImages.size()
            || !state.recordPublicAcquireSuccess(index)) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        state.failPublicAcquire(index);
        return VK_ERROR_DEVICE_LOST;
    }
    if (!imageIndex) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        if (vulkan->df().WaitForFences(vulkan->dev(), 1,
                    &retained.acquireFence, VK_TRUE, UINT64_MAX) == VK_SUCCESS
                && release && release(index) == VK_SUCCESS
                && state.recoverPublicAcquire(index, false)) {
            retainedTransactions.pop_back();
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        state.failPublicAcquire(index);
        return VK_ERROR_DEVICE_LOST;
    }
    const auto disposition = state.restoreDisposition(index);
    if (disposition == D2RestoreDisposition::InvalidShadowAuthority) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        if (vulkan->df().WaitForFences(vulkan->dev(), 1,
                    &retained.acquireFence, VK_TRUE, UINT64_MAX) == VK_SUCCESS
                && release && release(index) == VK_SUCCESS
                && state.recoverPublicAcquire(index, false)) {
            retainedTransactions.pop_back();
            return VK_ERROR_UNKNOWN;
        }
        state.failPublicAcquire(index);
        return VK_ERROR_DEVICE_LOST;
    }
    const bool restore = disposition == D2RestoreDisposition::Required;
    if (restore) {
        const auto record = recordCopy(
            retained, shadows[index]->handle(), realImages[index], false);
        if (record != VK_SUCCESS) {
            if (preparedProxy) proxies->abort(*preparedProxy);
            if (vulkan->df().WaitForFences(vulkan->dev(), 1,
                        &retained.acquireFence, VK_TRUE, UINT64_MAX) == VK_SUCCESS
                    && release && release(index) == VK_SUCCESS
                    && state.recoverPublicAcquire(index, false)) {
                retainedTransactions.pop_back();
                return record;
            }
            state.failPublicAcquire(index);
            return VK_ERROR_DEVICE_LOST;
        }
    }
    const auto submitResult = submit(retained, retained.acquireSemaphore, restore);
    if (submitResult != VK_SUCCESS) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        const bool accepted = retained.submitted;
        const bool cleanFailure = !accepted
            && (submitResult == VK_ERROR_OUT_OF_HOST_MEMORY
                || submitResult == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        const VkFence authority = accepted
            ? retained.completionFence : retained.acquireFence;
        const bool completed = (accepted || cleanFailure)
            && vulkan->df().WaitForFences
            && vulkan->df().WaitForFences(vulkan->dev(), 1,
                &authority, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        if (completed && release
                && release(index) == VK_SUCCESS
                && state.recoverPublicAcquire(index, accepted && restore)) {
            if (!accepted) retainedTransactions.pop_back();
            return submitResult;
        }
        state.failPublicAcquire(index);
        return VK_ERROR_DEVICE_LOST;
    }
    const auto handoffResult = handoff(
        retained, applicationSemaphore, applicationFence);
    if (handoffResult != VK_SUCCESS) {
        if (preparedProxy) proxies->abort(*preparedProxy);
        if (!release || vulkan->df().WaitForFences(vulkan->dev(), 1,
                    &retained.completionFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS
                || release(index) != VK_SUCCESS
                || !state.recoverPublicAcquire(index, restore)) {
            state.failPublicAcquire(index);
            return VK_ERROR_DEVICE_LOST;
        }
        return handoffResult;
    }
    // All remaining operations are prevalidated, non-allocating commits.
    if (restore) static_cast<void>(state.commitRestore(index));
    static_cast<void>(state.commitPublicAcquire(index));
    if (preparedProxy) proxies->commit(*preparedProxy);
    if (preparedProxy && preparedProxy->valid()) {
        state.failPublicAcquire(index);
        return VK_ERROR_DEVICE_LOST;
    }
    *imageIndex = index;
    return downstreamResult;
}

VkResult D2VulkanShadowRuntime::releaseUntouchedHidden(
        D2RealWsiState& state, uint32_t imageIndex, Transaction& transaction,
        VkSemaphore acquireSemaphore, const DownstreamRelease& downstream) noexcept {
    if (imageIndex >= realImages.size() || !state.prepareHiddenRelease(imageIndex)) {
        state.markTerminalForLsfgWork();
        return VK_ERROR_DEVICE_LOST;
    }

    // The queue operation below exists only to consume/retire the successful
    // Acquire signal.  It does not reference the WSI image.  Consequently a
    // failure to enqueue this drain must not orphan an otherwise untouched
    // physical acquisition: the image can still be returned through
    // Maintenance1 while the Transaction keeps the semaphore object alive to
    // the device-retirement boundary.
    const auto drainResult = submit(transaction, acquireSemaphore, false);
    if (drainResult == VK_ERROR_DEVICE_LOST) {
        (void)state.markHiddenAcquireTerminal(imageIndex);
        return VK_ERROR_DEVICE_LOST;
    }
    if (drainResult != VK_SUCCESS) {
        const auto waitResult = vulkan->df().WaitForFences(vulkan->dev(), 1,
            &transaction.acquireFence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS) {
            (void)state.markHiddenAcquireTerminal(imageIndex);
            return VK_ERROR_DEVICE_LOST;
        }
    }
    const auto releaseResult = downstream(imageIndex);
    state.finishHiddenRelease(imageIndex, releaseResult);

    if (releaseResult != VK_SUCCESS)
        (void)state.markHiddenAcquireTerminal(imageIndex);
    if (releaseResult != VK_SUCCESS) return releaseResult;
    return drainResult;
}

VkResult D2VulkanShadowRuntime::consumeAndReleaseIneligible(
        D2RealWsiState& state, uint32_t imageIndex,
        VkSemaphore internalAcquireSemaphore,
        const DownstreamRelease& downstream) noexcept {
    reapCompletedTransactions();
    std::shared_ptr<Transaction> transaction;
    try { transaction = prepareTransaction(false, false); }
    catch (...) { state.markTerminalForLsfgWork(); return VK_ERROR_DEVICE_LOST; }
    if (imageIndex >= realImages.size()) {
        try { retainedTransactions.push_back(std::move(transaction)); }
        catch (...) {}
        state.markTerminalForLsfgWork();
        return VK_ERROR_DEVICE_LOST;
    }
    try { retainedTransactions.push_back(std::move(transaction)); }
    catch (...) { state.markTerminalForLsfgWork(); return VK_ERROR_DEVICE_LOST; }
    auto* retained = retainedTransactions.back().get();
    const auto result = releaseUntouchedHidden(state, imageIndex,
        *retained, internalAcquireSemaphore, downstream);
    if (!retained->submitted && result != VK_ERROR_DEVICE_LOST)
        retainedTransactions.pop_back();
    return result;
}

VkResult D2VulkanShadowRuntime::submitShadowSave(D2RealWsiState& state,
        uint32_t imageIndex, uint64_t generation,
        VkSemaphore internalAcquireSemaphore) noexcept {
    reapCompletedTransactions();
    if (imageIndex >= realImages.size()) return VK_ERROR_DEVICE_LOST;
    try { retainedTransactions.push_back(prepareTransaction(false, false)); }
    catch (...) { state.markTerminalForLsfgWork(); return VK_ERROR_DEVICE_LOST; }
    auto& transaction = *retainedTransactions.back();
    auto result = recordCopy(transaction, realImages[imageIndex],
        shadows[imageIndex]->handle(), true);
    if (result == VK_SUCCESS)
        result = submit(transaction, internalAcquireSemaphore, true);
    if (result != VK_SUCCESS || !state.commitShadowSave(imageIndex, generation)) {
        state.markTerminalForLsfgWork();
        return VK_ERROR_DEVICE_LOST;
    }
    shadowInitialized[imageIndex] = true;
    return VK_SUCCESS;
}

D2HiddenAcquireResult D2VulkanShadowRuntime::attemptHiddenAcquire(
        D2RealWsiState& state, VkSwapchainKHR swapchain,
        const HiddenDownstreamAcquire& downstreamAcquire,
        const DownstreamRelease& downstreamRelease) noexcept {
    reapCompletedTransactions();
    if (!state.hiddenAcquireOpportunityAllowed())
        return {D2HiddenAcquireOutcome::NoImage, VK_NOT_READY, 0, false};
    try { retainedTransactions.push_back(prepareTransaction(false, false)); }
    catch (...) {
        state.markTerminalForLsfgWork();
        return {D2HiddenAcquireOutcome::NoImage,
            VK_ERROR_OUT_OF_HOST_MEMORY, 0, false};
    }
    auto* transaction = retainedTransactions.back().get();
    auto result = attemptD2HiddenAcquire(state, swapchain,
        transaction->acquireSemaphore, transaction->acquireFence,
        downstreamAcquire);
    if (result.outcome == D2HiddenAcquireOutcome::NoImage) {
        retainedTransactions.pop_back();
        return result;
    }
    if (result.outcome == D2HiddenAcquireOutcome::OwnedTerminal
            || result.physicalIndex >= preparedHiddenByImage.size())
        return result;
    preparedHiddenByImage[result.physicalIndex] = transaction;
    if (result.outcome == D2HiddenAcquireOutcome::OwnedIneligible) {
        const auto releaseResult = releaseUntouchedHidden(state, result.physicalIndex,
            *transaction, transaction->acquireSemaphore, downstreamRelease);
        preparedHiddenByImage[result.physicalIndex] = nullptr;
        if (!transaction->submitted && releaseResult != VK_ERROR_DEVICE_LOST)
            retainedTransactions.pop_back();
    }
    return result;
}

VkResult D2VulkanShadowRuntime::submitPreparedShadowSave(
        D2RealWsiState& state, uint32_t imageIndex,
        uint64_t generation, const DownstreamRelease& downstreamRelease) noexcept {
    if (imageIndex >= preparedHiddenByImage.size()
            || !preparedHiddenByImage[imageIndex])
        return VK_ERROR_DEVICE_LOST;
    auto& transaction = *preparedHiddenByImage[imageIndex];
    auto result = recordCopy(transaction, realImages[imageIndex],
        shadows[imageIndex]->handle(), true);
    const bool copyRecorded = result == VK_SUCCESS;
    if (copyRecorded)
        result = submit(transaction, transaction.acquireSemaphore, true);

    if (result != VK_SUCCESS) {
        const bool cleanSubmitFailure = copyRecorded && !transaction.submitted
            && (result == VK_ERROR_OUT_OF_HOST_MEMORY
                || result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        if (!copyRecorded || cleanSubmitFailure) {
            // No GPU command referencing Ri was accepted.  The image remains
            // ACQUIRED_UNTOUCHED and can follow the exact same Maintenance1
            // recovery as an ineligible hidden acquire.
            (void)releaseUntouchedHidden(state, imageIndex, transaction,
                transaction.acquireSemaphore, downstreamRelease);
            state.markTerminalForLsfgWork();
        } else if (transaction.submitted) {
            // The first submit (which reads Ri into ShadowRi) was accepted,
            // but a later retirement submit failed.  Clean Release is no
            // longer legal until that GPU use has completed, so retain the
            // physical acquisition and its Transaction to the terminal
            // device boundary.
            (void)state.markHiddenGpuUseAccepted(imageIndex);
            state.markTerminalForLsfgWork();
        } else {
            // For DEVICE_LOST/UNKNOWN-style first-submit failures Vulkan does
            // not provide the clean OOM failure-to-enqueue guarantee.  Keep
            // the acquired image as terminal authority rather than guessing
            // that an untouched Release is legal.
            (void)state.markHiddenAcquireTerminal(imageIndex);
        }
        preparedHiddenByImage[imageIndex] = nullptr;
        return VK_ERROR_DEVICE_LOST;
    }

    if (!state.commitShadowSave(imageIndex, generation)) {
        state.markTerminalForLsfgWork();
        preparedHiddenByImage[imageIndex] = nullptr;
        return VK_ERROR_DEVICE_LOST;
    }
    shadowInitialized[imageIndex] = true;
    preparedHiddenByImage[imageIndex] = nullptr;
    return VK_SUCCESS;
}

}
