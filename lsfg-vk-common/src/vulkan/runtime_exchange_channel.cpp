/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

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
