/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/buffer.hpp"
#include "lsfg-vk-common/vulkan/descriptor_set.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/shader.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cstdint>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    /// create a command buffer
    ls::owned_ptr<VkCommandBuffer> createCommandBuffer(const vk::Vulkan& vk,
            VkCommandPool pool) {
        VkCommandBuffer handle{};

        const VkCommandBufferAllocateInfo commandBufferInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        auto res = vk.df().AllocateCommandBuffers(vk.dev(), &commandBufferInfo, &handle);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkAllocateCommandBuffers() failed");

        auto setLoaderData = vk.loaderdatafunc();
        if (setLoaderData) {
            res = (*setLoaderData)(vk.dev(), handle);
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res, "vkSetDeviceLoaderData() failed");
        }

        return ls::owned_ptr<VkCommandBuffer>(
            new VkCommandBuffer(handle),
            [dev = vk.dev(), pool, defunc = vk.df().FreeCommandBuffers](
                VkCommandBuffer& commandBufferModule
            ) {
                defunc(dev, pool, 1, &commandBufferModule);
            }
        );
    }
}

CommandBuffer::CommandBuffer(const vk::Vulkan& vk)
        : commandBuffer(createCommandBuffer(vk, vk.cmdpool())) {}

CommandBuffer::CommandBuffer(const vk::Vulkan& vk, VkCommandPool pool)
        : commandBuffer(createCommandBuffer(vk, pool)) {}

void CommandBuffer::begin(const vk::Vulkan& vk) const {
    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    auto res = vk.df().BeginCommandBuffer(*this->commandBuffer, &beginInfo);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkBeginCommandBuffer() failed");
}

void CommandBuffer::insertBarriers(const vk::Vulkan& vk,
        const std::vector<vk::Barrier>& barriers,
        VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask) const {
    vk.df().CmdPipelineBarrier(*this->commandBuffer,
        srcStageMask, dstStageMask,
        0,
        0, VK_NULL_HANDLE,
        0, VK_NULL_HANDLE,
        static_cast<uint32_t>(barriers.size()), barriers.data()
    );
}

void CommandBuffer::dispatch(const vk::Vulkan& vk,
        const vk::Shader& shader,
        const vk::DescriptorSet& set,
        const std::vector<vk::Barrier>& barriers,
        uint32_t x, uint32_t y, uint32_t z) const {
    vk.df().CmdPipelineBarrier(*this->commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, VK_NULL_HANDLE,
        0, VK_NULL_HANDLE,
        static_cast<uint32_t>(barriers.size()), barriers.data()
    );
    vk.df().CmdBindPipeline(*this->commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.pipeline()
    );
    vk.df().CmdBindDescriptorSets(*this->commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.pipelinelayout(),
        0, 1, &set.handle(),
        0, VK_NULL_HANDLE
    );
    vk.df().CmdDispatch(*this->commandBuffer, x, y, z);
}

void CommandBuffer::blitImage(const vk::Vulkan& vk,
        const std::vector<vk::Barrier>& preBarriers,
        std::pair<VkImage, VkImage> images, VkExtent2D extent,
        const std::vector<vk::Barrier>& postBarriers,
        uint32_t arrayLayers) const {
    blitImage(vk, preBarriers, images, {extent, extent}, postBarriers, arrayLayers);
}

VkImageBlit vk::makeImageBlitRegion(
        std::pair<VkExtent2D, VkExtent2D> extents,
        uint32_t arrayLayers) noexcept {
    return VkImageBlit{
        .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = arrayLayers
        },
        .srcOffsets = {
            { 0, 0, 0 },
            { static_cast<int32_t>(extents.first.width),
              static_cast<int32_t>(extents.first.height), 1 }
        },
        .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = arrayLayers
        },
        .dstOffsets = {
            { 0, 0, 0 },
            { static_cast<int32_t>(extents.second.width),
              static_cast<int32_t>(extents.second.height), 1 }
        }
    };
}

void CommandBuffer::blitImage(const vk::Vulkan& vk,
        const std::vector<vk::Barrier>& preBarriers,
        std::pair<VkImage, VkImage> images,
        std::pair<VkExtent2D, VkExtent2D> extents,
        const std::vector<vk::Barrier>& postBarriers,
        uint32_t arrayLayers) const {
    vk.df().CmdPipelineBarrier(*this->commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, VK_NULL_HANDLE,
        0, VK_NULL_HANDLE,
        static_cast<uint32_t>(preBarriers.size()), preBarriers.data()
    );

    const auto region = vk::makeImageBlitRegion(extents, arrayLayers);
    vk.df().CmdBlitImage(*this->commandBuffer,
        images.first, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        images.second, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region,
        VK_FILTER_NEAREST
    );

    vk.df().CmdPipelineBarrier(*this->commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, VK_NULL_HANDLE,
        0, VK_NULL_HANDLE,
        static_cast<uint32_t>(postBarriers.size()), postBarriers.data()
    );
}

void CommandBuffer::copyBufferToImage(const vk::Vulkan& vk,
        const vk::Buffer& buffer, const vk::Image& image) const {
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_NONE,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.handle(),
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vk.df().CmdPipelineBarrier(*this->commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, VK_NULL_HANDLE,
        0, VK_NULL_HANDLE,
        1, &barrier
    );

    const VkBufferImageCopy region{
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1
        },
        .imageExtent = {
            .width = image.getExtent().width,
            .height = image.getExtent().height,
            .depth = 1
        }
    };
    vk.df().CmdCopyBufferToImage(*this->commandBuffer,
        buffer.handle(), image.handle(),
        VK_IMAGE_LAYOUT_GENERAL, 1, &region
    );
}

void CommandBuffer::end(const vk::Vulkan& vk) const {
    auto res = vk.df().EndCommandBuffer(*this->commandBuffer);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkEndCommandBuffer() failed");
}

void CommandBuffer::submit(const vk::Vulkan& vk,
        std::vector<VkSemaphore> waitSemaphores,
        VkSemaphore waitTimelineSemaphore, uint64_t waitValue,
        std::vector<VkSemaphore> signalSemaphores,
        VkSemaphore signalTimelineSemaphore, uint64_t signalValue,
        VkFence fence, VkPipelineStageFlags waitStage) const {
    // create arrays of semaphores and values
    if (waitTimelineSemaphore)
        waitSemaphores.push_back(waitTimelineSemaphore);

    std::vector<uint64_t> waitValues(waitSemaphores.size(), 0);
    waitValues.back() = waitValue;

    if (signalTimelineSemaphore)
        signalSemaphores.push_back(signalTimelineSemaphore);

    std::vector<uint64_t> signalValues(signalSemaphores.size(), 0);
    signalValues.back() = signalValue;

    // create submit info
    const VkTimelineSemaphoreSubmitInfo timelineInfo{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size()),
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size()),
        .pSignalSemaphoreValues = signalValues.data()
    };
    std::vector<VkPipelineStageFlags> stages(waitSemaphores.size(), waitStage);
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelineInfo,
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &*this->commandBuffer,
        .signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data()
    };
    auto res = vk.df().QueueSubmit(vk.queue(), 1, &submitInfo, fence);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkQueueSubmit() failed");
}


void CommandBuffer::submit(const vk::Vulkan& vk, VkQueue queue,
        std::vector<VkSemaphore> waitSemaphores,
        VkSemaphore waitTimelineSemaphore, uint64_t waitValue,
        std::vector<VkSemaphore> signalSemaphores,
        VkSemaphore signalTimelineSemaphore, uint64_t signalValue,
        VkFence fence, VkPipelineStageFlags waitStage,
        const SubmitObserver& observer,
        const SubmitResultObserver& resultObserver) const {
    if (waitTimelineSemaphore)
        waitSemaphores.push_back(waitTimelineSemaphore);

    std::vector<uint64_t> waitValues(waitSemaphores.size(), 0);
    if (waitTimelineSemaphore)
        waitValues.back() = waitValue;

    if (signalTimelineSemaphore)
        signalSemaphores.push_back(signalTimelineSemaphore);

    std::vector<uint64_t> signalValues(signalSemaphores.size(), 0);
    if (signalTimelineSemaphore)
        signalValues.back() = signalValue;

    const VkTimelineSemaphoreSubmitInfo timelineInfo{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size()),
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size()),
        .pSignalSemaphoreValues = signalValues.data()
    };
    std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), waitStage);
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelineInfo,
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &*this->commandBuffer,
        .signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data()
    };
    const QueueSubmitObserver queueObserver = observer
        ? QueueSubmitObserver{[&](VkQueue observedQueue, uint32_t count,
              const VkSubmitInfo* submits, VkFence observedFence) {
              if (count != 1 || !submits)
                  throw std::logic_error("CommandBuffer submit observation shape changed");
              observer(observedQueue, submits[0], observedFence);
          }}
        : QueueSubmitObserver{};
    const QueueSubmitResultObserver queueResultObserver = resultObserver
        ? QueueSubmitResultObserver{[&](VkQueue observedQueue, uint32_t count,
              const VkSubmitInfo* submits, VkFence observedFence, VkResult result) {
              if (count != 1 || !submits)
                  throw std::logic_error("CommandBuffer result observation shape changed");
              resultObserver(observedQueue, submits[0], observedFence, result);
          }}
        : QueueSubmitResultObserver{};
    auto res = executeQueueSubmit(vk.df().QueueSubmit, queue, 1, &submitInfo, fence,
        queueObserver, queueResultObserver);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkQueueSubmit() failed");
}

void CommandBuffer::submit(const vk::Vulkan& vk, VkQueue queue,
        const CommandBufferSubmit& submission, VkFence fence,
        const SubmitObserver& observer,
        const SubmitResultObserver& resultObserver) const {
    PreparedCommandBufferSubmit prepared;
    prepareCommandBufferSubmit(*this->commandBuffer, submission, prepared);
    if (observer) observer(queue, prepared.submitInfo, fence);
    const auto result = vk.df().QueueSubmit(queue, 1, &prepared.submitInfo, fence);
    if (resultObserver) resultObserver(queue, prepared.submitInfo, fence, result);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "vkQueueSubmit() failed");
}

void vk::prepareCommandBufferSubmit(VkCommandBuffer commandBuffer,
        const CommandBufferSubmit& submission,
        PreparedCommandBufferSubmit& prepared) {
    const size_t waitCount = submission.binaryWaits.size()
        + submission.timelineWaits.size();
    const size_t signalCount = submission.binarySignals.size()
        + submission.timelineSignals.size();
    if (waitCount > MAX_COMMAND_BUFFER_SUBMIT_SEMAPHORES
            || signalCount > MAX_COMMAND_BUFFER_SUBMIT_SEMAPHORES)
        throw std::logic_error("CommandBuffer submit semaphore capacity exceeded");
    if (!submission.binaryWaitStages.empty()
            && submission.binaryWaitStages.size() != submission.binaryWaits.size())
        throw std::logic_error("binary wait stage count does not match wait count");

    size_t cursor{};
    for (size_t i = 0; i < submission.binaryWaits.size(); ++i, ++cursor) {
        prepared.waits[cursor] = submission.binaryWaits[i];
        prepared.waitStages[cursor] = submission.binaryWaitStages.empty()
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : submission.binaryWaitStages[i];
    }
    for (const auto& wait : submission.timelineWaits) {
        prepared.waits[cursor] = wait.semaphore;
        prepared.waitValues[cursor] = wait.value;
        prepared.waitStages[cursor] = wait.stage;
        ++cursor;
    }

    cursor = 0;
    for (const auto semaphore : submission.binarySignals)
        prepared.signals[cursor++] = semaphore;
    for (const auto& signal : submission.timelineSignals) {
        prepared.signals[cursor] = signal.semaphore;
        prepared.signalValues[cursor] = signal.value;
        ++cursor;
    }

    prepared.timelineInfo = VkTimelineSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = static_cast<uint32_t>(waitCount),
        .pWaitSemaphoreValues = prepared.waitValues.data(),
        .signalSemaphoreValueCount = static_cast<uint32_t>(signalCount),
        .pSignalSemaphoreValues = prepared.signalValues.data()
    };
    prepared.submitInfo = VkSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &prepared.timelineInfo,
        .waitSemaphoreCount = static_cast<uint32_t>(waitCount),
        .pWaitSemaphores = prepared.waits.data(),
        .pWaitDstStageMask = prepared.waitStages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &prepared.commandBuffer,
        .signalSemaphoreCount = static_cast<uint32_t>(signalCount),
        .pSignalSemaphores = prepared.signals.data()
    };
    prepared.commandBuffer = commandBuffer;
}

void CommandBuffer::submit(const vk::Vulkan& vk,
        const CommandBufferSubmit& submission, VkFence fence) const {
    this->submit(vk, vk.queue(), submission, fence);
}


void CommandBuffer::submit(const vk::Vulkan& vk) const {
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &*this->commandBuffer
    };
    const vk::Fence fence{vk};
    auto res = vk.df().QueueSubmit(vk.queue(), 1, &submitInfo, fence.handle());
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkQueueSubmit() failed");

    if (!fence.wait(vk))
        throw ls::vulkan_error(VK_TIMEOUT, "Fence::wait() timed out");
}
