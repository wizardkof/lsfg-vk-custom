/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "external_buffer_transport.hpp"
#include "external_semaphore_sync.hpp"
#include "runtime_device_pair.hpp"
#include "external_memory_import.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {

    class Vulkan;

    /// Minimal logical-device endpoint needed by the P3D cross-device control channel.
    /// Both the application-managed render device and backend-managed generation device
    /// can be adapted from an existing vk::Vulkan without transferring ownership.
    struct RuntimeExchangeEndpoint {
        ExternalBufferDevice bufferDevice;
        ExternalSemaphoreDevice semaphoreDevice;
        VkQueue queue{};
        uint32_t queueFamilyIndex{};
        PFN_vkQueueSubmit QueueSubmit{};
        PFN_vkCreateFence CreateFence{};
        PFN_vkDestroyFence DestroyFence{};
        PFN_vkWaitForFences WaitForFences{};
        PFN_vkResetFences ResetFences{};
        PFN_vkDeviceWaitIdle DeviceWaitIdle{};
        PFN_vkCreateCommandPool CreateCommandPool{};
        PFN_vkDestroyCommandPool DestroyCommandPool{};
        PFN_vkAllocateCommandBuffers AllocateCommandBuffers{};
        PFN_vkFreeCommandBuffers FreeCommandBuffers{};
        PFN_vkBeginCommandBuffer BeginCommandBuffer{};
        PFN_vkEndCommandBuffer EndCommandBuffer{};
        PFN_vkCmdPipelineBarrier CmdPipelineBarrier{};
        PFN_vkCmdFillBuffer CmdFillBuffer{};
        PFN_vkCmdCopyBuffer CmdCopyBuffer{};
        PFN_vkMapMemory MapMemory{};
        PFN_vkUnmapMemory UnmapMemory{};
        PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges{};

        // Borrowed physical-device/image dispatch. The endpoint never destroys
        // the instance, physical device, logical device, or queue.
        VkPhysicalDevice physicalDevice{};
        PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2{};
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2{};
        PFN_vkCreateImage CreateImage{};
        PFN_vkDestroyImage DestroyImage{};
        PFN_vkGetImageMemoryRequirements2 GetImageMemoryRequirements2{};
        PFN_vkBindImageMemory BindImageMemory{};
        PFN_vkCmdClearColorImage CmdClearColorImage{};
        PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer{};
        PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR{};
        PFN_vkAllocateMemory AllocateMemory{};
        PFN_vkFreeMemory FreeMemory{};
        PFN_vkCreateBuffer CreateBuffer{};
        PFN_vkDestroyBuffer DestroyBuffer{};
        PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{};
        PFN_vkBindBufferMemory BindBufferMemory{};
    };

    struct RuntimeExchangeChannelInfo {
        VkDeviceSize logicalSize{};
        VkDeviceSize backingSize{};
        VkBufferUsageFlags usage{};
    };

    struct RuntimeImageBackingInfo {
        uint32_t fourcc{};
        uint64_t modifier{};
        uint32_t planeCount{};
        VkSubresourceLayout plane{};
    };


    class RuntimeImageEndpoint {
    public:
        RuntimeImageEndpoint() noexcept = default;
        RuntimeImageEndpoint(const RuntimeImageEndpoint&) = delete;
        RuntimeImageEndpoint& operator=(const RuntimeImageEndpoint&) = delete;
        RuntimeImageEndpoint(RuntimeImageEndpoint&&) noexcept;
        RuntimeImageEndpoint& operator=(RuntimeImageEndpoint&&) noexcept;
        ~RuntimeImageEndpoint();
        [[nodiscard]] VkImage image() const noexcept { return imageHandle; }
        [[nodiscard]] const ExternalMemoryImportDiagnostics& diagnostics() const noexcept { return importDiagnostics; }
        [[nodiscard]] const VkMemoryRequirements2& requirements() const noexcept { return memoryRequirements; }
        [[nodiscard]] bool executionResourcesCreated() const noexcept { return commandPool != VK_NULL_HANDLE; }
        [[nodiscard]] static RuntimeImageEndpoint createExecutionResources(
            RuntimeImageEndpoint&& endpoint, uint32_t commandBufferCount);
        static void executeInitialDiagnostic(RuntimeImageEndpoint& endpoint);
        [[nodiscard]] static SyncFdPayload executeInitialChained(RuntimeImageEndpoint& endpoint);
        static void executeAtoBDiagnostic(RuntimeImageEndpoint& imageA,
            RuntimeImageEndpoint& imageB);
        static void executeCompleteRoundTrip(RuntimeImageEndpoint& imageA,
            RuntimeImageEndpoint& imageB);
    private:
        friend RuntimeImageEndpoint createRuntimeImageEndpoint(
            const RuntimeExchangeEndpoint&, ls::OwnedFd, const RuntimeImageBackingInfo&);
        RuntimeExchangeEndpoint endpoint{};
        VkImage imageHandle{};
        ImportedExternalMemory memory{};
        VkMemoryRequirements2 memoryRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        ExternalMemoryImportDiagnostics importDiagnostics{};
        VkCommandPool commandPool{};
        std::vector<VkCommandBuffer> commandBuffers;
        VkBuffer stagingBuffer{};
        VkDeviceMemory stagingMemory{};
        void* stagingMapped{};
        bool stagingHostCoherent{};
        VkDeviceSize stagingSize{};
        VkFence finalFence{};
    };

    [[nodiscard]] RuntimeImageEndpoint createRuntimeImageEndpoint(
        const RuntimeExchangeEndpoint&, ls::OwnedFd, const RuntimeImageBackingInfo&);


    struct RuntimeExchangeSyncDiagnostics {
        bool renderToGenerationSentinel{};
        bool generationToRenderSentinel{};
        bool hostWaitBeforeFinalSubmit{};
    };

    struct RuntimeExchangePayloadDiagnostics {
        VkDeviceSize payloadSize{};
        bool renderWrite{};
        bool generationObservedRender{};
        bool generationWrite{};
        bool renderObservedGeneration{};
        bool hostWaitBeforeFinalSubmit{};
    };

    [[nodiscard]] bool validRuntimePayloadExchangeOrder(
        const std::vector<std::string>& events) noexcept;
    [[nodiscard]] bool verifyRuntimePayloadPattern(
        const std::vector<uint32_t>& words, uint32_t expected) noexcept;

    /// Adapt an already-created logical Vulkan device to the narrow P3D endpoint contract.
    [[nodiscard]] RuntimeExchangeEndpoint makeRuntimeExchangeEndpoint(const Vulkan& vk);

    /// Runtime control channel composed from the P3A DMA-BUF import primitive and the
    /// P3B SYNC_FD semaphore primitive. P3D intentionally validates only buffer import,
    /// synchronization, ownership, and lifetime; frame-image transport is connected later.
    class RuntimeExchangeChannel {
    public:
        RuntimeExchangeChannel() = delete;
        RuntimeExchangeChannel(const RuntimeExchangeChannel&) = delete;
        RuntimeExchangeChannel& operator=(const RuntimeExchangeChannel&) = delete;
        RuntimeExchangeChannel(RuntimeExchangeChannel&&) noexcept = default;
        RuntimeExchangeChannel& operator=(RuntimeExchangeChannel&&) noexcept = default;
        ~RuntimeExchangeChannel() = default;

        [[nodiscard]] VkBuffer renderBuffer() const noexcept {
            return this->renderImportedBuffer.buffer();
        }
        [[nodiscard]] VkBuffer generationBuffer() const noexcept {
            return this->generationImportedBuffer.buffer();
        }
        [[nodiscard]] const ExternalBufferImportDiagnostics& renderBufferDiagnostics() const noexcept {
            return this->renderImportedBuffer.diagnostics();
        }
        [[nodiscard]] const ExternalBufferImportDiagnostics& generationBufferDiagnostics() const noexcept {
            return this->generationImportedBuffer.diagnostics();
        }

        /// Submit an empty binary-semaphore round trip A -> B -> A using SYNC_FD.
        /// The only host fence wait occurs after the final A submit.
        [[nodiscard]] RuntimeExchangeSyncDiagnostics validateSyncRoundTrip();

        /// Execute a real GPU-side A -> B -> A payload exchange over the already
        /// imported DMA-BUF buffers. CPU readback is performed only after the
        /// final A fence and is never used as the cross-device transport.
        [[nodiscard]] RuntimeExchangePayloadDiagnostics validatePayloadRoundTrip(
            VkDeviceSize payloadSize);

    private:
        friend RuntimeExchangeChannel createRuntimeExchangeChannel(
            const RuntimeDevicePair&,
            RuntimeExchangeEndpoint,
            RuntimeExchangeEndpoint,
            ls::OwnedFd,
            ls::OwnedFd,
            const RuntimeExchangeChannelInfo&);

        RuntimeExchangeChannel(
            RuntimeExchangeEndpoint render,
            RuntimeExchangeEndpoint generation,
            ImportedExternalBuffer renderBuffer,
            ImportedExternalBuffer generationBuffer,
            SyncFdSemaphore renderExport,
            SyncFdSemaphore renderImport,
            SyncFdSemaphore generationExport,
            SyncFdSemaphore generationImport) noexcept;

        RuntimeExchangeEndpoint renderEndpoint;
        RuntimeExchangeEndpoint generationEndpoint;
        ImportedExternalBuffer renderImportedBuffer;
        ImportedExternalBuffer generationImportedBuffer;
        SyncFdSemaphore renderExportSemaphore;
        SyncFdSemaphore renderImportSemaphore;
        SyncFdSemaphore generationExportSemaphore;
        SyncFdSemaphore generationImportSemaphore;
    };

    /// Create the P3D channel over two duplicated descriptors referring to the same DMA-BUF.
    /// Cross-physical-device mode is required by design; same-GPU continues through the
    /// existing production path and does not construct this control channel.
    [[nodiscard]] RuntimeExchangeChannel createRuntimeExchangeChannel(
        const RuntimeDevicePair& pair,
        RuntimeExchangeEndpoint render,
        RuntimeExchangeEndpoint generation,
        ls::OwnedFd renderFd,
        ls::OwnedFd generationFd,
        const RuntimeExchangeChannelInfo& info);

}
