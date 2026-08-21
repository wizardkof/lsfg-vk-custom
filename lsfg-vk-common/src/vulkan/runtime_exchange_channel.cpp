/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/external_image_transport.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/fnv1a.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <libdrm/drm_fourcc.h>

#include <vulkan/vulkan_core.h>

using namespace vk;

bool vk::validRuntimePayloadExchangeOrder(
        const std::vector<std::string>& events) noexcept {
    static const std::vector<std::string> expected{
        "A_WRITE", "A_SIGNAL_EXPORT", "B_IMPORT_WAIT", "B_OBSERVE_A",
        "B_WRITE", "B_SIGNAL_EXPORT", "A_IMPORT_WAIT", "A_OBSERVE_B",
        "FINAL_SUBMIT", "HOST_WAIT"
    };
    return events == expected;
}

bool vk::verifyRuntimePayloadPattern(
        const std::vector<uint32_t>& words, uint32_t expected) noexcept {
    return std::all_of(words.begin(), words.end(),
        [expected](uint32_t word) { return word == expected; });
}

namespace {
    constexpr uint32_t PATTERN_A = 0xA5A5A5A5;
    constexpr uint32_t PATTERN_B = 0x5A5A5A5A;

    class FinalFence {
    public:
        FinalFence(const RuntimeExchangeEndpoint& endpoint) : endpoint(&endpoint) {
            const VkFenceCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
            };
            const auto result = endpoint.CreateFence(
                endpoint.bufferDevice.device, &info, nullptr, &this->handle);
            if (result != VK_SUCCESS)
                throw ls::vulkan_error(result, "vkCreateFence() failed for runtime exchange channel");
        }

        FinalFence(const FinalFence&) = delete;
        FinalFence& operator=(const FinalFence&) = delete;
        ~FinalFence() {
            if (this->endpoint && this->handle != VK_NULL_HANDLE)
                this->endpoint->DestroyFence(
                    this->endpoint->bufferDevice.device, this->handle, nullptr);
        }

        [[nodiscard]] VkFence get() const noexcept { return this->handle; }

    private:
        const RuntimeExchangeEndpoint* endpoint{};
        VkFence handle{};
    };

    void validateEndpoint(const RuntimeExchangeEndpoint& endpoint, const char* label) {
        const bool valid = endpoint.bufferDevice.device != VK_NULL_HANDLE
            && endpoint.semaphoreDevice.device == endpoint.bufferDevice.device
            && endpoint.queue != VK_NULL_HANDLE
            && endpoint.QueueSubmit
            && endpoint.CreateFence
            && endpoint.DestroyFence
            && endpoint.WaitForFences
            && endpoint.DeviceWaitIdle;
        if (!valid)
            throw std::invalid_argument(std::string("invalid runtime exchange endpoint: ") + label);
    }

    void submitSignal(const RuntimeExchangeEndpoint& endpoint, VkSemaphore signal) {
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() signal failed for runtime exchange channel");
    }

    void submitWaitSignal(const RuntimeExchangeEndpoint& endpoint,
            VkSemaphore wait, VkSemaphore signal) {
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait,
            .pWaitDstStageMask = &waitStage,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() wait/signal failed for runtime exchange channel");
    }

    void submitFinalWait(const RuntimeExchangeEndpoint& endpoint,
            VkSemaphore wait, VkFence fence) {
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait,
            .pWaitDstStageMask = &waitStage
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, fence);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() final wait failed for runtime exchange channel");
    }

    struct LocalBuffer {
        const RuntimeExchangeEndpoint* endpoint{};
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkMemoryPropertyFlags properties{};
        ~LocalBuffer() {
            if (endpoint && buffer)
                endpoint->bufferDevice.funcs.DestroyBuffer(endpoint->bufferDevice.device,
                    buffer, nullptr);
            if (endpoint && memory)
                endpoint->bufferDevice.funcs.FreeMemory(endpoint->bufferDevice.device,
                    memory, nullptr);
        }
    };

    LocalBuffer makeStaging(const RuntimeExchangeEndpoint& endpoint, VkDeviceSize size) {
        const VkBufferCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        LocalBuffer result{&endpoint};
        auto r = endpoint.bufferDevice.funcs.CreateBuffer(
            endpoint.bufferDevice.device, &info, nullptr, &result.buffer);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkCreateBuffer() failed for payload staging");
        VkMemoryRequirements requirements{};
        endpoint.bufferDevice.funcs.GetBufferMemoryRequirements(
            endpoint.bufferDevice.device, result.buffer, &requirements);
        const auto index = selectExternalBufferMemoryType(
            requirements.memoryTypeBits, endpoint.bufferDevice.memoryProperties,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!index)
            throw ls::vulkan_error("no host-visible payload staging memory type");
        result.properties = endpoint.bufferDevice.memoryProperties.memoryTypes[*index].propertyFlags;
        const VkMemoryAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *index
        };
        r = endpoint.bufferDevice.funcs.AllocateMemory(
            endpoint.bufferDevice.device, &allocate, nullptr, &result.memory);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkAllocateMemory() failed for payload staging");
        r = endpoint.bufferDevice.funcs.BindBufferMemory(
            endpoint.bufferDevice.device, result.buffer, result.memory, 0);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkBindBufferMemory() failed for payload staging");
        return result;
    }

    struct CommandResources {
        const RuntimeExchangeEndpoint* endpoint{};
        VkCommandPool pool{};
        VkCommandBuffer command{};
        ~CommandResources() {
            if (!endpoint)
                return;
            if (command)
                endpoint->FreeCommandBuffers(endpoint->bufferDevice.device, pool, 1, &command);
            if (pool)
                endpoint->DestroyCommandPool(endpoint->bufferDevice.device, pool, nullptr);
        }
    };

    CommandResources beginCommands(const RuntimeExchangeEndpoint& endpoint) {
        CommandResources result{&endpoint};
        const VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = endpoint.queueFamilyIndex
        };
        auto r = endpoint.CreateCommandPool(endpoint.bufferDevice.device,
            &poolInfo, nullptr, &result.pool);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkCreateCommandPool() failed for payload exchange");
        const VkCommandBufferAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = result.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        r = endpoint.AllocateCommandBuffers(endpoint.bufferDevice.device, &allocate, &result.command);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkAllocateCommandBuffers() failed for payload exchange");
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        r = endpoint.BeginCommandBuffer(result.command, &begin);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkBeginCommandBuffer() failed for payload exchange");
        return result;
    }

    void ownershipBarrier(const RuntimeExchangeEndpoint& endpoint, VkCommandBuffer command,
            VkBuffer buffer, uint32_t sourceFamily, uint32_t destinationFamily,
            VkAccessFlags sourceAccess, VkAccessFlags destinationAccess,
            VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage,
            VkDeviceSize size) {
        const VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = sourceAccess,
            .dstAccessMask = destinationAccess,
            .srcQueueFamilyIndex = sourceFamily,
            .dstQueueFamilyIndex = destinationFamily,
            .buffer = buffer,
            .offset = 0,
            .size = size
        };
        endpoint.CmdPipelineBarrier(command, sourceStage, destinationStage, 0,
            0, nullptr, 1, &barrier, 0, nullptr);
    }

    void submitPayload(const RuntimeExchangeEndpoint& endpoint, VkCommandBuffer command,
            VkSemaphore wait, VkSemaphore signal, VkFence fence = VK_NULL_HANDLE) {
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = wait ? 1U : 0U,
            .pWaitSemaphores = wait ? &wait : nullptr,
            .pWaitDstStageMask = wait ? &waitStage : nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &command,
            .signalSemaphoreCount = signal ? 1U : 0U,
            .pSignalSemaphores = signal ? &signal : nullptr
        };
        const auto r = endpoint.QueueSubmit(endpoint.queue, 1, &submit, fence);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkQueueSubmit() failed for payload exchange");
    }

    bool verifyStaging(const RuntimeExchangeEndpoint& endpoint, const LocalBuffer& staging,
            VkDeviceSize size, uint32_t expected) {
        void* mapped{};
        auto r = endpoint.MapMemory(endpoint.bufferDevice.device, staging.memory, 0,
            VK_WHOLE_SIZE, 0, &mapped);
        if (r != VK_SUCCESS)
            throw ls::vulkan_error(r, "vkMapMemory() failed for payload verification");
        if ((staging.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            const VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = staging.memory,
                .size = VK_WHOLE_SIZE
            };
            r = endpoint.InvalidateMappedMemoryRanges(endpoint.bufferDevice.device, 1, &range);
            if (r != VK_SUCCESS) {
                endpoint.UnmapMemory(endpoint.bufferDevice.device, staging.memory);
                throw ls::vulkan_error(r, "vkInvalidateMappedMemoryRanges() failed for payload verification");
            }
        }
        const auto* words = static_cast<const uint32_t*>(mapped);
        const auto count = static_cast<size_t>(size / sizeof(uint32_t));
        bool matches = true;
        for (size_t i = 0; i < count; ++i) {
            if (words[i] != expected) { matches = false; break; }
        }
        endpoint.UnmapMemory(endpoint.bufferDevice.device, staging.memory);
        return matches;
    }
}

RuntimeImageEndpoint::RuntimeImageEndpoint(RuntimeImageEndpoint&& other) noexcept
    : endpoint(std::move(other.endpoint)), imageHandle(std::exchange(other.imageHandle, VK_NULL_HANDLE)),
      memory(std::move(other.memory)), memoryRequirements(other.memoryRequirements),
      importDiagnostics(other.importDiagnostics), commandPool(std::exchange(other.commandPool, VK_NULL_HANDLE)),
      commandBuffers(std::move(other.commandBuffers)), stagingBuffer(std::exchange(other.stagingBuffer, VK_NULL_HANDLE)),
      stagingMemory(std::exchange(other.stagingMemory, VK_NULL_HANDLE)), stagingMapped(std::exchange(other.stagingMapped, nullptr)),
      stagingHostCoherent(other.stagingHostCoherent), stagingSize(other.stagingSize),
      finalFence(std::exchange(other.finalFence, VK_NULL_HANDLE)), extent(other.extent) {}

RuntimeImageEndpoint& RuntimeImageEndpoint::operator=(RuntimeImageEndpoint&& other) noexcept {
    if (this != &other) {
        this->~RuntimeImageEndpoint();
        endpoint = std::move(other.endpoint);
        imageHandle = std::exchange(other.imageHandle, VK_NULL_HANDLE);
        memory = std::move(other.memory);
        memoryRequirements = other.memoryRequirements;
        importDiagnostics = other.importDiagnostics;
        commandPool = std::exchange(other.commandPool, VK_NULL_HANDLE);
        commandBuffers = std::move(other.commandBuffers);
        stagingBuffer = std::exchange(other.stagingBuffer, VK_NULL_HANDLE);
        stagingMemory = std::exchange(other.stagingMemory, VK_NULL_HANDLE);
        stagingMapped = std::exchange(other.stagingMapped, nullptr);
        stagingHostCoherent = other.stagingHostCoherent;
        stagingSize = other.stagingSize;
        finalFence = std::exchange(other.finalFence, VK_NULL_HANDLE);
        extent = other.extent;
    }
    return *this;
}

RuntimeImageEndpoint::~RuntimeImageEndpoint() {
    if (stagingMapped && endpoint.UnmapMemory)
        endpoint.UnmapMemory(endpoint.bufferDevice.device, stagingMemory);
    if (finalFence && endpoint.DestroyFence)
        endpoint.DestroyFence(endpoint.bufferDevice.device, finalFence, nullptr);
    if (!commandBuffers.empty() && endpoint.FreeCommandBuffers)
        endpoint.FreeCommandBuffers(endpoint.bufferDevice.device, commandPool,
            static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    if (commandPool && endpoint.DestroyCommandPool)
        endpoint.DestroyCommandPool(endpoint.bufferDevice.device, commandPool, nullptr);
    if (stagingBuffer && endpoint.bufferDevice.funcs.DestroyBuffer)
        endpoint.bufferDevice.funcs.DestroyBuffer(endpoint.bufferDevice.device, stagingBuffer, nullptr);
    if (stagingMemory && endpoint.bufferDevice.funcs.FreeMemory)
        endpoint.bufferDevice.funcs.FreeMemory(endpoint.bufferDevice.device, stagingMemory, nullptr);
    if (imageHandle && endpoint.DestroyImage)
        endpoint.DestroyImage(endpoint.bufferDevice.device, imageHandle, nullptr);
}

RuntimeImageEndpoint RuntimeImageEndpoint::createExecutionResources(
        RuntimeImageEndpoint&& source, uint32_t commandBufferCount) {
    auto& endpoint = source.endpoint;
    if (!endpoint.CreateCommandPool || !endpoint.DestroyCommandPool
            || !endpoint.AllocateCommandBuffers || !endpoint.CreateBuffer
            || !endpoint.DestroyBuffer || !endpoint.GetBufferMemoryRequirements
            || !endpoint.AllocateMemory || !endpoint.FreeMemory
            || !endpoint.BindBufferMemory || !endpoint.MapMemory || !endpoint.UnmapMemory
            || !endpoint.CreateFence || !endpoint.DestroyFence)
        throw std::runtime_error("runtime image execution dispatch incomplete");
    source.stagingSize = static_cast<VkDeviceSize>(source.extent.width)
        * source.extent.height * 4;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = source.stagingSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto result = endpoint.CreateBuffer(endpoint.bufferDevice.device, &bufferInfo, nullptr,
        &source.stagingBuffer);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 staging buffer failed");
    VkMemoryRequirements requirements{};
    endpoint.GetBufferMemoryRequirements(endpoint.bufferDevice.device, source.stagingBuffer,
        &requirements);
    std::optional<uint32_t> selected;
    for (uint32_t i = 0; i < endpoint.bufferDevice.memoryProperties.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        const auto flags = endpoint.bufferDevice.memoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            selected = i; break;
        }
        if (!selected && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) selected = i;
    }
    if (!selected) throw std::runtime_error("B3.1 staging memory type unavailable");
    source.stagingHostCoherent = (endpoint.bufferDevice.memoryProperties
        .memoryTypes[*selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = *selected;
    result = endpoint.AllocateMemory(endpoint.bufferDevice.device, &allocate, nullptr,
        &source.stagingMemory);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 staging memory failed");
    result = endpoint.BindBufferMemory(endpoint.bufferDevice.device, source.stagingBuffer,
        source.stagingMemory, 0);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 staging bind failed");
    result = endpoint.MapMemory(endpoint.bufferDevice.device, source.stagingMemory, 0,
        source.stagingSize, 0, &source.stagingMapped);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 staging map failed");
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = endpoint.queueFamilyIndex;
    result = endpoint.CreateCommandPool(endpoint.bufferDevice.device, &pool, nullptr,
        &source.commandPool);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 command pool failed");
    VkCommandBufferAllocateInfo commands{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commands.commandPool = source.commandPool;
    commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commands.commandBufferCount = commandBufferCount;
    source.commandBuffers.resize(commandBufferCount);
    result = endpoint.AllocateCommandBuffers(endpoint.bufferDevice.device, &commands,
        source.commandBuffers.data());
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 command buffers failed");
    if (commandBufferCount > 1) {
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        result = endpoint.CreateFence(endpoint.bufferDevice.device, &fence, nullptr,
            &source.finalFence);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.1 final fence failed");
    }
    return std::move(source);
}

SyncFdPayload RuntimeImageEndpoint::executeInitialChained(RuntimeImageEndpoint& endpoint) {
    if (endpoint.commandBuffers.empty() || !endpoint.endpoint.CmdPipelineBarrier
            || !endpoint.endpoint.CmdClearColorImage || !endpoint.endpoint.QueueSubmit)
        throw std::runtime_error("B3.2 A dispatch/resources incomplete");
    const VkCommandBuffer command = endpoint.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = endpoint.endpoint.BeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 vkBeginCommandBuffer");
    const VkImageMemoryBarrier acquire{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_FOREIGN_EXT, endpoint.endpoint.queueFamilyIndex,
        endpoint.imageHandle, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    endpoint.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);
    const VkClearColorValue pattern{{0.6470588F, 0.6470588F, 0.6470588F, 0.6470588F}};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    endpoint.endpoint.CmdClearColorImage(command, endpoint.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &pattern, 1, &range);
    const VkImageMemoryBarrier release{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        endpoint.endpoint.queueFamilyIndex, VK_QUEUE_FAMILY_FOREIGN_EXT,
        endpoint.imageHandle, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    endpoint.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
    result = endpoint.endpoint.EndCommandBuffer(command);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 vkEndCommandBuffer");

    auto signal = createExportableSyncFdSemaphore(endpoint.endpoint.semaphoreDevice);
    const VkSemaphore signalHandle = signal.handle();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signalHandle;
    if (endpoint.finalFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        result = endpoint.endpoint.CreateFence(endpoint.endpoint.bufferDevice.device,
            &fenceInfo, nullptr, &endpoint.finalFence);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 fence creation");
    }
    result = endpoint.endpoint.QueueSubmit(endpoint.endpoint.queue, 1, &submit,
        VK_NULL_HANDLE);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 vkQueueSubmit A");
    return exportSyncFd(endpoint.endpoint.semaphoreDevice, signal.handle());
}

void RuntimeImageEndpoint::executeInitialDiagnostic(RuntimeImageEndpoint& endpoint) {
    auto exported = executeInitialChained(endpoint);
    std::cerr << "[DG2X-P4B-B3.2] SYNC_FD A->B export: "
        << (exported.sentinel() ? "SENTINEL_-1" : "FD") << "\n";
    VkFence diagnosticFence{};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    auto result = endpoint.endpoint.CreateFence(endpoint.endpoint.bufferDevice.device,
        &fenceInfo, nullptr, &diagnosticFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 diagnostic fence creation");
    // The standalone wrapper submits independently only for its diagnostic mode.
    // The chained primitive above intentionally returns without a host wait.
    result = endpoint.endpoint.QueueSubmit(endpoint.endpoint.queue, 0, nullptr, diagnosticFence);
    if (result != VK_SUCCESS) {
        endpoint.endpoint.DestroyFence(endpoint.endpoint.bufferDevice.device, diagnosticFence, nullptr);
        throw ls::vulkan_error(result, "B3.2 diagnostic fence submit");
    }
    result = endpoint.endpoint.WaitForFences(endpoint.endpoint.bufferDevice.device, 1,
        &diagnosticFence, VK_TRUE, UINT64_MAX);
    endpoint.endpoint.DestroyFence(endpoint.endpoint.bufferDevice.device, diagnosticFence, nullptr);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.2 diagnostic fence wait");
}

void RuntimeImageEndpoint::executeAtoBDiagnostic(RuntimeImageEndpoint& imageA,
        RuntimeImageEndpoint& imageB) {
    auto payloadAB = executeInitialChained(imageA);
    auto importB = createSyncFdImportSemaphore(imageB.endpoint.semaphoreDevice);
    importSyncFdTemporary(imageB.endpoint.semaphoreDevice, importB.handle(), payloadAB);

    if (imageB.commandBuffers.empty() || !imageB.endpoint.CmdCopyImageToBuffer)
        throw std::runtime_error("B3.3 B image execution dispatch incomplete");
    const VkCommandBuffer command = imageB.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = imageB.endpoint.BeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 vkBeginCommandBuffer B");
    const VkImageMemoryBarrier acquire{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_FOREIGN_EXT, imageB.endpoint.queueFamilyIndex,
        imageB.imageHandle, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    imageB.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);
    const VkBufferImageCopy copy{0, 0, 0,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {256, 256, 1}};
    imageB.endpoint.CmdCopyImageToBuffer(command, imageB.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageB.stagingBuffer, 1, &copy);
    const VkImageMemoryBarrier toClear{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, imageB.endpoint.queueFamilyIndex,
        imageB.endpoint.queueFamilyIndex, imageB.imageHandle,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    imageB.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
    const VkClearColorValue patternB{{0.3529412F, 0.3529412F, 0.3529412F, 0.3529412F}};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    imageB.endpoint.CmdClearColorImage(command, imageB.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &patternB, 1, &range);
    const VkImageMemoryBarrier release{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        imageB.endpoint.queueFamilyIndex, VK_QUEUE_FAMILY_FOREIGN_EXT,
        imageB.imageHandle, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    imageB.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
    result = imageB.endpoint.EndCommandBuffer(command);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 vkEndCommandBuffer B");
    auto signalB = createExportableSyncFdSemaphore(imageB.endpoint.semaphoreDevice);
    const VkSemaphore waitHandle = importB.handle();
    const VkSemaphore signalHandle = signalB.handle();
    constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &waitHandle;
    submit.pWaitDstStageMask = &waitStage; submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command; submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signalHandle;
    VkFence diagnosticFence{};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    result = imageB.endpoint.CreateFence(imageB.endpoint.bufferDevice.device,
        &fenceInfo, nullptr, &diagnosticFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 diagnostic fence creation");
    result = imageB.endpoint.QueueSubmit(imageB.endpoint.queue, 1, &submit, diagnosticFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 vkQueueSubmit B");
    auto payloadBA = exportSyncFd(imageB.endpoint.semaphoreDevice, signalB.handle());
    std::cerr << "[DG2X-P4B-B3.3] SYNC_FD B->A export: "
        << (payloadBA.sentinel() ? "SENTINEL_-1" : "FD") << "\n";
    result = imageB.endpoint.WaitForFences(imageB.endpoint.bufferDevice.device, 1,
        &diagnosticFence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 diagnostic fence wait");
    imageB.endpoint.DestroyFence(imageB.endpoint.bufferDevice.device, diagnosticFence, nullptr);
    if (!imageB.stagingHostCoherent) {
        VkMappedMemoryRange rangeMemory{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        rangeMemory.memory = imageB.stagingMemory; rangeMemory.size = VK_WHOLE_SIZE;
        result = imageB.endpoint.InvalidateMappedMemoryRanges(
            imageB.endpoint.bufferDevice.device, 1, &rangeMemory);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.3 staging invalidate");
    }
    const auto* bytes = static_cast<const uint8_t*>(imageB.stagingMapped);
    constexpr uint8_t expectedByte = 0xA5;
    for (VkDeviceSize i = 0; i < imageB.stagingSize; ++i)
        if (bytes[i] != expectedByte)
            throw std::runtime_error("B3.3 Pattern A mismatch at staging offset "
                + std::to_string(i) + " expected=0xA5 actual=0x"
                + [&] { char out[3]{}; std::snprintf(out, sizeof(out), "%02X", bytes[i]);
                    return std::string(out); }());
}

void RuntimeImageEndpoint::executeCompleteRoundTrip(RuntimeImageEndpoint& imageA,
        RuntimeImageEndpoint& imageB) {
    auto payloadAB = executeInitialChained(imageA);
    auto importB = createSyncFdImportSemaphore(imageB.endpoint.semaphoreDevice);
    importSyncFdTemporary(imageB.endpoint.semaphoreDevice, importB.handle(), payloadAB);
    const VkCommandBuffer bcmd = imageB.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = imageB.endpoint.BeginCommandBuffer(bcmd, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkBeginCommandBuffer B");
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageMemoryBarrier acquireB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_FOREIGN_EXT, imageB.endpoint.queueFamilyIndex, imageB.imageHandle,
        range};
    imageB.endpoint.CmdPipelineBarrier(bcmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquireB);
    const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0}, {256, 256, 1}};
    imageB.endpoint.CmdCopyImageToBuffer(bcmd, imageB.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageB.stagingBuffer, 1, &copy);
    const VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        imageB.endpoint.queueFamilyIndex, imageB.endpoint.queueFamilyIndex, imageB.imageHandle,
        range};
    imageB.endpoint.CmdPipelineBarrier(bcmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);
    const VkClearColorValue patternB{{0.3529412F, 0.3529412F, 0.3529412F, 0.3529412F}};
    imageB.endpoint.CmdClearColorImage(bcmd, imageB.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &patternB, 1, &range);
    const VkImageMemoryBarrier releaseB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageB.endpoint.queueFamilyIndex,
        VK_QUEUE_FAMILY_FOREIGN_EXT, imageB.imageHandle, range};
    imageB.endpoint.CmdPipelineBarrier(bcmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &releaseB);
    result = imageB.endpoint.EndCommandBuffer(bcmd);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkEndCommandBuffer B");
    auto signalB = createExportableSyncFdSemaphore(imageB.endpoint.semaphoreDevice);
    const VkSemaphore waitB = importB.handle(), signalHandle = signalB.handle();
    constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submitB{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submitB.waitSemaphoreCount = 1;
    submitB.pWaitSemaphores = &waitB; submitB.pWaitDstStageMask = &stage;
    submitB.commandBufferCount = 1; submitB.pCommandBuffers = &bcmd;
    submitB.signalSemaphoreCount = 1; submitB.pSignalSemaphores = &signalHandle;
    result = imageB.endpoint.QueueSubmit(imageB.endpoint.queue, 1, &submitB, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkQueueSubmit B");
    auto payloadBA = exportSyncFd(imageB.endpoint.semaphoreDevice, signalB.handle());
    auto importA = createSyncFdImportSemaphore(imageA.endpoint.semaphoreDevice);
    importSyncFdTemporary(imageA.endpoint.semaphoreDevice, importA.handle(), payloadBA);
    const VkCommandBuffer acmd = imageA.commandBuffers.at(1);
    result = imageA.endpoint.BeginCommandBuffer(acmd, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkBeginCommandBuffer A final");
    const VkImageMemoryBarrier acquireA{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT,
        imageA.endpoint.queueFamilyIndex, imageA.imageHandle, range};
    imageA.endpoint.CmdPipelineBarrier(acmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquireA);
    imageA.endpoint.CmdCopyImageToBuffer(acmd, imageA.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageA.stagingBuffer, 1, &copy);
    result = imageA.endpoint.EndCommandBuffer(acmd);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkEndCommandBuffer A final");
    if (!imageA.finalFence || !imageA.endpoint.ResetFences)
        throw std::runtime_error("B3.4 final fence/reset dispatch unavailable");
    result = imageA.endpoint.ResetFences(imageA.endpoint.bufferDevice.device, 1,
        &imageA.finalFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkResetFences");
    const VkSemaphore finalWait = importA.handle();
    VkSubmitInfo finalSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; finalSubmit.waitSemaphoreCount = 1;
    finalSubmit.pWaitSemaphores = &finalWait; finalSubmit.pWaitDstStageMask = &stage;
    finalSubmit.commandBufferCount = 1; finalSubmit.pCommandBuffers = &acmd;
    result = imageA.endpoint.QueueSubmit(imageA.endpoint.queue, 1, &finalSubmit,
        imageA.finalFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkQueueSubmit A final");
    result = imageA.endpoint.WaitForFences(imageA.endpoint.bufferDevice.device, 1,
        &imageA.finalFence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "B3.4 vkWaitForFences");
    auto check = [](RuntimeImageEndpoint& image, uint8_t expected) {
        if (!image.stagingHostCoherent) {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = image.stagingMemory; r.size = VK_WHOLE_SIZE;
            auto v = image.endpoint.InvalidateMappedMemoryRanges(image.endpoint.bufferDevice.device, 1, &r);
            if (v != VK_SUCCESS) throw ls::vulkan_error(v, "B3.4 staging invalidate");
        }
        const auto* p = static_cast<const uint8_t*>(image.stagingMapped);
        for (VkDeviceSize i = 0; i < image.stagingSize; ++i)
            if (p[i] != expected) throw std::runtime_error("B3.4 staging mismatch offset " + std::to_string(i));
    };
    check(imageB, 0xA5); check(imageA, 0x5A);
}

void RuntimeImageEndpoint::executeRealFrameTransport(RuntimeImageEndpoint& imageA,
        RuntimeImageEndpoint& imageB, VkImage sourceImage, VkExtent2D sourceExtent,
        VkSemaphore bridgeWait) {
    if (sourceExtent.width != imageA.extent.width || sourceExtent.height != imageA.extent.height
            || imageA.extent.width != imageB.extent.width || imageA.extent.height != imageB.extent.height)
        throw std::runtime_error("P4C-C source/transport extent mismatch");
    if (!imageA.endpoint.CmdBlitImage || !imageB.endpoint.CmdCopyImageToBuffer)
        throw std::runtime_error("P4C-C image transfer dispatch incomplete");
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkCommandBuffer acmd = imageA.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = imageA.endpoint.BeginCommandBuffer(acmd, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkBeginCommandBuffer A");
    const VkImageMemoryBarrier sourceToRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        0, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, sourceImage, range};
    const VkImageMemoryBarrier transportAcquire{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT,
        imageA.endpoint.queueFamilyIndex, imageA.imageHandle, range};
    imageA.endpoint.CmdPipelineBarrier(acmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2,
        std::array{sourceToRead, transportAcquire}.data());
    const VkImageBlit blit{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {static_cast<int32_t>(sourceExtent.width),
            static_cast<int32_t>(sourceExtent.height), 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(imageA.extent.width),
            static_cast<int32_t>(imageA.extent.height), 1}}
    };
    imageA.endpoint.CmdBlitImage(acmd, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        imageA.imageHandle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
        VK_FILTER_NEAREST);
    const VkImageMemoryBarrier sourceRestore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        sourceImage, range};
    const VkImageMemoryBarrier transportRelease{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageA.endpoint.queueFamilyIndex,
        VK_QUEUE_FAMILY_FOREIGN_EXT, imageA.imageHandle, range};
    imageA.endpoint.CmdPipelineBarrier(acmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2,
        std::array{sourceRestore, transportRelease}.data());
    result = imageA.endpoint.EndCommandBuffer(acmd);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkEndCommandBuffer A");
    auto signalA = createExportableSyncFdSemaphore(imageA.endpoint.semaphoreDevice);
    const VkSemaphore signal = signalA.handle();
    VkSubmitInfo submitA{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitA.waitSemaphoreCount = bridgeWait ? 1U : 0U;
    submitA.pWaitSemaphores = bridgeWait ? &bridgeWait : nullptr;
    constexpr VkPipelineStageFlags transferStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    submitA.pWaitDstStageMask = bridgeWait ? &transferStage : nullptr;
    submitA.commandBufferCount = 1; submitA.pCommandBuffers = &acmd;
    submitA.signalSemaphoreCount = 1; submitA.pSignalSemaphores = &signal;
    result = imageA.endpoint.QueueSubmit(imageA.endpoint.queue, 1, &submitA, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkQueueSubmit A");
    auto payload = exportSyncFd(imageA.endpoint.semaphoreDevice, signalA.handle());
    auto importB = createSyncFdImportSemaphore(imageB.endpoint.semaphoreDevice);
    importSyncFdTemporary(imageB.endpoint.semaphoreDevice, importB.handle(), payload);

    const VkCommandBuffer bcmd = imageB.commandBuffers.front();
    result = imageB.endpoint.BeginCommandBuffer(bcmd, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkBeginCommandBuffer B");
    const VkImageMemoryBarrier acquireB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT,
        imageB.endpoint.queueFamilyIndex, imageB.imageHandle, range};
    imageB.endpoint.CmdPipelineBarrier(bcmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquireB);
    const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0}, {imageB.extent.width, imageB.extent.height, 1}};
    imageB.endpoint.CmdCopyImageToBuffer(bcmd, imageB.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageB.stagingBuffer, 1, &copy);
    const VkImageMemoryBarrier releaseB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageB.endpoint.queueFamilyIndex,
        VK_QUEUE_FAMILY_FOREIGN_EXT, imageB.imageHandle, range};
    imageB.endpoint.CmdPipelineBarrier(bcmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &releaseB);
    result = imageB.endpoint.EndCommandBuffer(bcmd);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkEndCommandBuffer B");
    const VkSemaphore waitB = importB.handle();
    VkSubmitInfo submitB{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submitB.waitSemaphoreCount = 1;
    submitB.pWaitSemaphores = &waitB; submitB.pWaitDstStageMask = &transferStage;
    submitB.commandBufferCount = 1; submitB.pCommandBuffers = &bcmd;
    VkFence fence{}; VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    result = imageB.endpoint.CreateFence(imageB.endpoint.bufferDevice.device,
        &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C B fence creation");
    result = imageB.endpoint.QueueSubmit(imageB.endpoint.queue, 1, &submitB, fence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C vkQueueSubmit B");
    result = imageB.endpoint.WaitForFences(imageB.endpoint.bufferDevice.device, 1,
        &fence, VK_TRUE, UINT64_MAX);
    imageB.endpoint.DestroyFence(imageB.endpoint.bufferDevice.device, fence, nullptr);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C B fence wait");
    if (!imageB.stagingHostCoherent) {
        VkMappedMemoryRange invalidate{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        invalidate.memory = imageB.stagingMemory; invalidate.size = VK_WHOLE_SIZE;
        result = imageB.endpoint.InvalidateMappedMemoryRanges(
            imageB.endpoint.bufferDevice.device, 1, &invalidate);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-C B readback invalidate");
    }
    const auto* bytes = static_cast<const uint8_t*>(imageB.stagingMapped);
    uint64_t checksum = ::lsfgvk::common::fnv1a64(bytes, imageB.stagingSize); VkDeviceSize nonZero{};
    for (VkDeviceSize i = 0; i < imageB.stagingSize; ++i) {
        nonZero += bytes[i] != 0;
    }
    if (nonZero == 0) throw std::runtime_error("P4C-C B readback is entirely zero");
    std::cerr << "[DG2X-P4C-C] Real frame A->B transport\n"
        << "  Runtime mode: CAPTURE_ONLY\n"
        << "  Source VkImage: " << sourceImage << "\n"
        << "  Source extent: " << sourceExtent.width << 'x' << sourceExtent.height << "\n"
        << "  Transport format: VK_FORMAT_B8G8R8A8_UNORM\n"
        << "  Transport extent: " << imageA.extent.width << 'x' << imageA.extent.height << "\n"
        << "  bridgePresentWaits: PASS\n"
        << "  Source -> transport GPU blit: PASS\n"
        << "  Source state restore: PASS\n"
        << "  Transport A release FOREIGN: PASS\n"
        << "  DMA_BUF export/import: PASS\n"
        << "  SYNC_FD A->B: PASS\n"
        << "  Transport B acquire FOREIGN: PASS\n"
        << "  Transport B -> staging copy: PASS\n"
        << "  B capture submit: PASS\n"
        << "  B fence completion: PASS\n"
        << "  B captured byte count: " << imageB.stagingSize << "\n"
        << "  B non-zero bytes: " << nonZero << "\n"
        << "  B checksum: 0x" << std::hex << checksum << std::dec << "\n"
        << "  CPU frame bridge: NONE\n"
        << "  LSFG backend execution: NONE\n"
        << "  Generated frame: NONE\n"
        << "  Presentation from B: NONE\n"
        << "  Frame transport connected: NO\n"
        << "DG2X_P4C_C_REAL_FRAME_A_TO_B_PASS\n"
        << "cross-device real frame reached generation GPU, but LSFG frame processing is not connected yet\n";
}

SyncFdPayload RuntimeImageEndpoint::submitRealFrameTransportA(
        RuntimeImageEndpoint& imageA, VkImage sourceImage, VkExtent2D sourceExtent,
        VkSemaphore bridgeWait) {
    if (sourceExtent.width != imageA.extent.width || sourceExtent.height != imageA.extent.height)
        throw std::runtime_error("P4C-D1 source/transport extent mismatch");
    if (!imageA.endpoint.CmdBlitImage)
        throw std::runtime_error("P4C-D1 A image transfer dispatch incomplete");
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkCommandBuffer command = imageA.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = imageA.endpoint.BeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-D1 vkBeginCommandBuffer A");
    const std::array acquire{
        VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, sourceImage, range},
        VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT,
            imageA.endpoint.queueFamilyIndex, imageA.imageHandle, range}};
    imageA.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, acquire.size(), acquire.data());
    const VkImageBlit blit{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {static_cast<int32_t>(sourceExtent.width),
            static_cast<int32_t>(sourceExtent.height), 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(imageA.extent.width),
            static_cast<int32_t>(imageA.extent.height), 1}}};
    imageA.endpoint.CmdBlitImage(command, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        imageA.imageHandle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    const std::array release{
        VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, sourceImage, range},
        VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageA.endpoint.queueFamilyIndex,
            VK_QUEUE_FAMILY_FOREIGN_EXT, imageA.imageHandle, range}};
    imageA.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, release.size(), release.data());
    result = imageA.endpoint.EndCommandBuffer(command);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-D1 vkEndCommandBuffer A");
    auto signal = createExportableSyncFdSemaphore(imageA.endpoint.semaphoreDevice);
    const VkSemaphore signalHandle = signal.handle();
    constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = bridgeWait ? 1U : 0U;
    submit.pWaitSemaphores = bridgeWait ? &bridgeWait : nullptr;
    submit.pWaitDstStageMask = bridgeWait ? &stage : nullptr;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &signalHandle;
    result = imageA.endpoint.QueueSubmit(imageA.endpoint.queue, 1, &submit, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-D1 vkQueueSubmit A");
    return exportSyncFd(imageA.endpoint.semaphoreDevice, signal.handle());
}

std::vector<uint8_t> RuntimeImageEndpoint::readForeignImage(
        RuntimeImageEndpoint& imageA, SyncFdPayload payload) {
    if (imageA.commandBuffers.empty() || !imageA.endpoint.CmdCopyImageToBuffer
            || !imageA.endpoint.QueueSubmit || !imageA.endpoint.WaitForFences)
        throw std::runtime_error("D3A1 A readback dispatch/resources incomplete");
    bool aSubmitted = false;
    bool aCompleted = false;
    try {
    auto imported = createSyncFdImportSemaphore(imageA.endpoint.semaphoreDevice);
    importSyncFdTemporary(imageA.endpoint.semaphoreDevice, imported.handle(), payload);
    const VkCommandBuffer command = imageA.commandBuffers.front();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto result = imageA.endpoint.BeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 vkBeginCommandBuffer A");
    const VkImageMemoryBarrier acquire{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT,
        imageA.endpoint.queueFamilyIndex, imageA.imageHandle,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    imageA.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);
    const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0}, {imageA.extent.width, imageA.extent.height, 1}};
    imageA.endpoint.CmdCopyImageToBuffer(command, imageA.imageHandle,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageA.stagingBuffer, 1, &copy);
    const VkBufferMemoryBarrier hostRead{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, imageA.stagingBuffer, 0, imageA.stagingSize};
    imageA.endpoint.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostRead, 0, nullptr);
    result = imageA.endpoint.EndCommandBuffer(command);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 vkEndCommandBuffer A");
    if (!imageA.finalFence) {
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        result = imageA.endpoint.CreateFence(imageA.endpoint.bufferDevice.device,
            &info, nullptr, &imageA.finalFence);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 final A fence creation");
    }
    const VkSemaphore wait = imported.handle();
    constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &wait;
    submit.pWaitDstStageMask = &stage; submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    result = imageA.endpoint.QueueSubmit(imageA.endpoint.queue, 1, &submit, imageA.finalFence);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 vkQueueSubmit A");
    aSubmitted = true;
    result = imageA.endpoint.WaitForFences(imageA.endpoint.bufferDevice.device, 1,
        &imageA.finalFence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 final A fence wait");
    aCompleted = true;
    if (!imageA.stagingHostCoherent) {
        VkMappedMemoryRange invalidate{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        invalidate.memory = imageA.stagingMemory; invalidate.size = VK_WHOLE_SIZE;
        result = imageA.endpoint.InvalidateMappedMemoryRanges(
            imageA.endpoint.bufferDevice.device, 1, &invalidate);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A1 A readback invalidate");
    }
    const auto* first = static_cast<const uint8_t*>(imageA.stagingMapped);
    return {first, first + imageA.stagingSize};
    } catch (...) {
        if (aSubmitted && !aCompleted && imageA.endpoint.DeviceWaitIdle)
            static_cast<void>(imageA.endpoint.DeviceWaitIdle(imageA.endpoint.bufferDevice.device));
        throw;
    }
}

RuntimeImageEndpoint vk::createRuntimeImageEndpoint(const RuntimeExchangeEndpoint& source,
        ls::OwnedFd fd, const RuntimeImageBackingInfo& backing) {
    if (!source.CreateImage || !source.DestroyImage || !source.GetImageMemoryRequirements2
            || !source.BindImageMemory || !source.GetMemoryFdPropertiesKHR)
        throw std::invalid_argument("image-capable runtime endpoint is incomplete");
    const auto expectedFourcc = drmFourccForVkFormat(backing.format);
    if (!expectedFourcc || backing.fourcc != *expectedFourcc || backing.planeCount == 0)
        throw std::invalid_argument("runtime image backing format/plane mismatch");
    RuntimeImageEndpoint result;
    result.endpoint = source;
    const std::vector<VkSubresourceLayout> queriedPlaneLayouts = backing.planes.empty()
        ? std::vector<VkSubresourceLayout>{backing.plane} : backing.planes;
    if (queriedPlaneLayouts.size() != backing.planeCount)
        throw std::invalid_argument("runtime image backing plane layout count mismatch");
    const auto planeLayouts = normalizeExplicitImagePlaneLayouts(
        queriedPlaneLayouts, 1, 1);
    VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkImageDrmFormatModifierExplicitCreateInfoEXT modifier{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
    modifier.drmFormatModifier = backing.modifier;
    modifier.drmFormatModifierPlaneCount = backing.planeCount;
    modifier.pPlaneLayouts = planeLayouts.data();
    modifier.pNext = &external;
    result.extent = backing.extent;
    const VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &modifier, 0,
        VK_IMAGE_TYPE_2D, backing.format,
        {backing.extent.width, backing.extent.height, 1}, 1, 1,
        VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        backing.usage,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
    auto r = source.CreateImage(source.bufferDevice.device, &create, nullptr, &result.imageHandle);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "vkCreateImage runtime image failed");
    VkImageMemoryRequirementsInfo2 query{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    query.image = result.imageHandle;
    VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    result.memoryRequirements.pNext = &dedicated;
    VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    requirements.pNext = &dedicated;
    source.GetImageMemoryRequirements2(source.bufferDevice.device, &query, &requirements);
    result.memoryRequirements = requirements;
    if (backing.backingSize < requirements.memoryRequirements.size)
        throw std::invalid_argument("external image backing is smaller than Vulkan requirements");
    ExternalMemoryImportPlan plan{
        source.bufferDevice.device, source.AllocateMemory, source.FreeMemory,
        source.GetMemoryFdPropertiesKHR, source.bufferDevice.memoryProperties,
        backing.backingSize, requirements.memoryRequirements.memoryTypeBits,
        result.imageHandle, false, dedicated.requiresDedicatedAllocation == VK_TRUE, false};
    try {
        result.memory = ImportedExternalMemory::import(plan, std::move(fd), &result.importDiagnostics);
    } catch (const ls::vulkan_error& error) {
        throw ls::vulkan_error(error.error(), "runtime image external import failed");
    }
    std::cerr << "[DG2X-P4C-D3A1] AMD import memory\n"
        << "  Requirements size: " << requirements.memoryRequirements.size << "\n"
        << "  Requirements alignment: " << requirements.memoryRequirements.alignment << "\n"
        << "  Requirements memoryTypeBits: 0x" << std::hex
        << requirements.memoryRequirements.memoryTypeBits << "\n"
        << "  FD memoryTypeBits: 0x" << result.importDiagnostics.fdMemoryTypeBits << "\n"
        << "  Intersection: 0x" << result.importDiagnostics.intersectionMemoryTypeBits
        << std::dec << "\n"
        << "  Selected memory type: " << result.importDiagnostics.memoryTypeIndex << "\n"
        << "  Dedicated required: " << dedicated.requiresDedicatedAllocation << "\n"
        << "  Dedicated preferred: " << dedicated.prefersDedicatedAllocation << "\n"
        << "  Dedicated allocation: " << result.importDiagnostics.dedicated << "\n";
    r = source.BindImageMemory(source.bufferDevice.device, result.imageHandle,
        result.memory.memory(), 0);
    if (r != VK_SUCCESS) throw ls::vulkan_error(r, "vkBindImageMemory runtime image failed");
    return result;
}

RuntimeExchangeChannel::RuntimeExchangeChannel(
        RuntimeExchangeEndpoint render,
        RuntimeExchangeEndpoint generation,
        ImportedExternalBuffer renderBuffer,
        ImportedExternalBuffer generationBuffer,
        SyncFdSemaphore renderExport,
        SyncFdSemaphore renderImport,
        SyncFdSemaphore generationExport,
        SyncFdSemaphore generationImport) noexcept :
    renderEndpoint(std::move(render)),
    generationEndpoint(std::move(generation)),
    renderImportedBuffer(std::move(renderBuffer)),
    generationImportedBuffer(std::move(generationBuffer)),
    renderExportSemaphore(std::move(renderExport)),
    renderImportSemaphore(std::move(renderImport)),
    generationExportSemaphore(std::move(generationExport)),
    generationImportSemaphore(std::move(generationImport)) {
}

RuntimeExchangeChannel vk::createRuntimeExchangeChannel(
        const RuntimeDevicePair& pair,
        RuntimeExchangeEndpoint render,
        RuntimeExchangeEndpoint generation,
        ls::OwnedFd renderFd,
        ls::OwnedFd generationFd,
        const RuntimeExchangeChannelInfo& info) {
    if (!pair.crossDevice())
        throw std::invalid_argument(
            "runtime cross-device exchange channel requires CROSS_PHYSICAL_DEVICE mode");
    validateEndpoint(render, "render");
    validateEndpoint(generation, "generation");
    if (info.logicalSize == 0 || info.backingSize < info.logicalSize || info.usage == 0)
        throw std::invalid_argument("invalid runtime exchange channel buffer contract");

    const ExternalBufferImportInfo importInfo{
        .logicalSize = info.logicalSize,
        .backingSize = info.backingSize,
        .usage = info.usage,
        .requiredMemoryProperties = 0,
        .preferredMemoryProperties = 0
    };

    auto renderBuffer = importDmaBufBuffer(
        render.bufferDevice, std::move(renderFd), importInfo);
    auto generationBuffer = importDmaBufBuffer(
        generation.bufferDevice, std::move(generationFd), importInfo);

    auto renderExport = createExportableSyncFdSemaphore(render.semaphoreDevice);
    auto renderImport = createSyncFdImportSemaphore(render.semaphoreDevice);
    auto generationExport = createExportableSyncFdSemaphore(generation.semaphoreDevice);
    auto generationImport = createSyncFdImportSemaphore(generation.semaphoreDevice);

    return RuntimeExchangeChannel(
        std::move(render), std::move(generation),
        std::move(renderBuffer), std::move(generationBuffer),
        std::move(renderExport), std::move(renderImport),
        std::move(generationExport), std::move(generationImport));
}

RuntimeExchangeSyncDiagnostics RuntimeExchangeChannel::validateSyncRoundTrip() {
    // Keep the final fence alive across failure cleanup so it is never destroyed
    // while a successful final submit could still reference it.
    FinalFence finalFence(this->renderEndpoint);
    bool renderSubmitted = false;
    bool generationSubmitted = false;
    try {
        // A -> B. No host wait is allowed between these submissions.
        submitSignal(this->renderEndpoint, this->renderExportSemaphore.handle());
        renderSubmitted = true;
        auto renderPayload = exportSyncFd(
            this->renderEndpoint.semaphoreDevice, this->renderExportSemaphore.handle());
        const bool renderSentinel = renderPayload.sentinel();
        importSyncFdTemporary(this->generationEndpoint.semaphoreDevice,
            this->generationImportSemaphore.handle(), renderPayload);

        submitWaitSignal(this->generationEndpoint,
            this->generationImportSemaphore.handle(),
            this->generationExportSemaphore.handle());
        generationSubmitted = true;
        auto generationPayload = exportSyncFd(
            this->generationEndpoint.semaphoreDevice,
            this->generationExportSemaphore.handle());
        const bool generationSentinel = generationPayload.sentinel();
        importSyncFdTemporary(this->renderEndpoint.semaphoreDevice,
            this->renderImportSemaphore.handle(), generationPayload);

        submitFinalWait(this->renderEndpoint,
            this->renderImportSemaphore.handle(), finalFence.get());

        // First and only host wait in the successful round trip: all A -> B -> A
        // submits already exist. DeviceWaitIdle is reserved for exceptional cleanup.
        const VkFence fence = finalFence.get();
        const auto waitResult = this->renderEndpoint.WaitForFences(
            this->renderEndpoint.bufferDevice.device,
            1, &fence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS)
            throw ls::vulkan_error(waitResult,
                "vkWaitForFences() failed for runtime exchange channel");

        return {
            .renderToGenerationSentinel = renderSentinel,
            .generationToRenderSentinel = generationSentinel,
            .hostWaitBeforeFinalSubmit = false
        };
    } catch (...) {
        // Do not destroy semaphores/buffers while submitted work can still reference
        // them. These waits are failure cleanup only and are never the success bridge.
        if (generationSubmitted)
            static_cast<void>(this->generationEndpoint.DeviceWaitIdle(
                this->generationEndpoint.bufferDevice.device));
        if (renderSubmitted)
            static_cast<void>(this->renderEndpoint.DeviceWaitIdle(
                this->renderEndpoint.bufferDevice.device));
        throw;
    }
}

RuntimeExchangePayloadDiagnostics RuntimeExchangeChannel::validatePayloadRoundTrip(
        VkDeviceSize payloadSize) {
    if (payloadSize == 0 || (payloadSize % sizeof(uint32_t)) != 0
            || payloadSize > renderImportedBuffer.diagnostics().requirements.size)
        throw std::invalid_argument("invalid runtime payload size");

    FinalFence finalFence(this->renderEndpoint);
    auto generationStaging = makeStaging(this->generationEndpoint, payloadSize);
    auto renderStaging = makeStaging(this->renderEndpoint, payloadSize);
    bool renderSubmitted = false;
    bool generationSubmitted = false;
    try {
        auto renderCommands = beginCommands(this->renderEndpoint);
        ownershipBarrier(this->renderEndpoint, renderCommands.command,
            renderImportedBuffer.buffer(), VK_QUEUE_FAMILY_FOREIGN_EXT,
            renderEndpoint.queueFamilyIndex, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, payloadSize);
        this->renderEndpoint.CmdFillBuffer(renderCommands.command,
            renderImportedBuffer.buffer(), 0, payloadSize, PATTERN_A);
        ownershipBarrier(this->renderEndpoint, renderCommands.command,
            renderImportedBuffer.buffer(), renderEndpoint.queueFamilyIndex,
            VK_QUEUE_FAMILY_FOREIGN_EXT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, payloadSize);
        if (this->renderEndpoint.EndCommandBuffer(renderCommands.command) != VK_SUCCESS)
            throw ls::vulkan_error("vkEndCommandBuffer() failed for render payload write");
        submitPayload(this->renderEndpoint, renderCommands.command,
            VK_NULL_HANDLE, renderExportSemaphore.handle());
        renderSubmitted = true;
        auto renderPayload = exportSyncFd(this->renderEndpoint.semaphoreDevice,
            renderExportSemaphore.handle());
        importSyncFdTemporary(this->generationEndpoint.semaphoreDevice,
            generationImportSemaphore.handle(), renderPayload);

        auto generationCommands = beginCommands(this->generationEndpoint);
        ownershipBarrier(this->generationEndpoint, generationCommands.command,
            generationImportedBuffer.buffer(), VK_QUEUE_FAMILY_FOREIGN_EXT,
            generationEndpoint.queueFamilyIndex, 0,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, payloadSize);
        const VkBufferCopy copyToGeneration{0, 0, payloadSize};
        this->generationEndpoint.CmdCopyBuffer(generationCommands.command,
            generationImportedBuffer.buffer(), generationStaging.buffer, 1, &copyToGeneration);
        const VkMemoryBarrier copyBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
        };
        this->generationEndpoint.CmdPipelineBarrier(generationCommands.command,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            1, &copyBarrier, 0, nullptr, 0, nullptr);
        this->generationEndpoint.CmdFillBuffer(generationCommands.command,
            generationImportedBuffer.buffer(), 0, payloadSize, PATTERN_B);
        ownershipBarrier(this->generationEndpoint, generationCommands.command,
            generationImportedBuffer.buffer(), generationEndpoint.queueFamilyIndex,
            VK_QUEUE_FAMILY_FOREIGN_EXT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, payloadSize);
        if (this->generationEndpoint.EndCommandBuffer(generationCommands.command) != VK_SUCCESS)
            throw ls::vulkan_error("vkEndCommandBuffer() failed for generation payload exchange");
        submitPayload(this->generationEndpoint, generationCommands.command,
            generationImportSemaphore.handle(), generationExportSemaphore.handle());
        generationSubmitted = true;
        auto generationPayload = exportSyncFd(this->generationEndpoint.semaphoreDevice,
            generationExportSemaphore.handle());
        importSyncFdTemporary(this->renderEndpoint.semaphoreDevice,
            renderImportSemaphore.handle(), generationPayload);

        auto finalCommands = beginCommands(this->renderEndpoint);
        ownershipBarrier(this->renderEndpoint, finalCommands.command,
            renderImportedBuffer.buffer(), VK_QUEUE_FAMILY_FOREIGN_EXT,
            renderEndpoint.queueFamilyIndex, 0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, payloadSize);
        const VkBufferCopy copyToRender{0, 0, payloadSize};
        this->renderEndpoint.CmdCopyBuffer(finalCommands.command,
            renderImportedBuffer.buffer(), renderStaging.buffer, 1, &copyToRender);
        if (this->renderEndpoint.EndCommandBuffer(finalCommands.command) != VK_SUCCESS)
            throw ls::vulkan_error("vkEndCommandBuffer() failed for final payload verification");
        submitPayload(this->renderEndpoint, finalCommands.command,
            renderImportSemaphore.handle(), VK_NULL_HANDLE, finalFence.get());

        const VkFence fence = finalFence.get();
        const auto waitResult = this->renderEndpoint.WaitForFences(
            this->renderEndpoint.bufferDevice.device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS)
            throw ls::vulkan_error(waitResult, "vkWaitForFences() failed for payload exchange");
        const bool observedA = verifyStaging(this->generationEndpoint, generationStaging,
            payloadSize, PATTERN_A);
        const bool observedB = verifyStaging(this->renderEndpoint, renderStaging,
            payloadSize, PATTERN_B);
        if (!observedA)
            throw ls::vulkan_error("generation payload verification failed for Pattern A");
        if (!observedB)
            throw ls::vulkan_error("render payload verification failed for Pattern B");
        return {payloadSize, true, observedA, true, observedB, false};
    } catch (...) {
        if (generationSubmitted)
            static_cast<void>(this->generationEndpoint.DeviceWaitIdle(
                this->generationEndpoint.bufferDevice.device));
        if (renderSubmitted)
            static_cast<void>(this->renderEndpoint.DeviceWaitIdle(
                this->renderEndpoint.bufferDevice.device));
        throw;
    }
}
