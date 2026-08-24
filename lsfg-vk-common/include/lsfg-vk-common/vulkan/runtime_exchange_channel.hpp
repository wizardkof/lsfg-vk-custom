/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "external_buffer_transport.hpp"
#include "external_semaphore_sync.hpp"
#include "runtime_device_pair.hpp"
#include "external_memory_import.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {

    class Vulkan;
    class RuntimeImageEndpoint;

    struct RuntimeForeignImageHandoffInfo {
        uint32_t destinationQueueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
        VkSemaphore signalSemaphore{VK_NULL_HANDLE}; // borrowed; never destroyed here
    };

    class RuntimeForeignImageReadbackPending;

    class RuntimeForeignImageView {
    public:
        [[nodiscard]] bool valid() const noexcept { return imageHandle != VK_NULL_HANDLE && !lifetime.expired(); }
        [[nodiscard]] VkImage image() const noexcept { return valid() ? imageHandle : VK_NULL_HANDLE; }
        [[nodiscard]] VkFormat format() const noexcept { return formatValue; }
        [[nodiscard]] VkExtent2D extent() const noexcept { return extentValue; }
        [[nodiscard]] VkImageLayout layout() const noexcept { return layoutValue; }
        [[nodiscard]] uint32_t queueFamily() const noexcept {
            return handoffPending ? VK_QUEUE_FAMILY_IGNORED : queueFamilyValue;
        }
        [[nodiscard]] uint32_t sourceQueueFamily() const noexcept { return sourceQueueFamilyValue; }
        [[nodiscard]] uint32_t destinationQueueFamily() const noexcept { return destinationQueueFamilyValue; }
        [[nodiscard]] bool handoffPendingAcquire() const noexcept { return handoffPending; }
        [[nodiscard]] uint64_t modifier() const noexcept { return modifierValue; }
    private:
        friend class RuntimeForeignImageReadbackPending;
        VkImage imageHandle{};
        VkFormat formatValue{VK_FORMAT_UNDEFINED};
        VkExtent2D extentValue{};
        VkImageLayout layoutValue{VK_IMAGE_LAYOUT_UNDEFINED};
        uint32_t queueFamilyValue{VK_QUEUE_FAMILY_IGNORED};
        uint32_t sourceQueueFamilyValue{VK_QUEUE_FAMILY_IGNORED};
        uint32_t destinationQueueFamilyValue{VK_QUEUE_FAMILY_IGNORED};
        uint64_t modifierValue{};
        bool handoffPending{};
        std::weak_ptr<const uint8_t> lifetime;
    };

    class RuntimeForeignImageReadbackPending {
    public:
        RuntimeForeignImageReadbackPending() noexcept = default;
        RuntimeForeignImageReadbackPending(const RuntimeForeignImageReadbackPending&) = delete;
        RuntimeForeignImageReadbackPending& operator=(const RuntimeForeignImageReadbackPending&) = delete;
        RuntimeForeignImageReadbackPending(RuntimeForeignImageReadbackPending&&) noexcept;
        RuntimeForeignImageReadbackPending& operator=(RuntimeForeignImageReadbackPending&&) noexcept;
        ~RuntimeForeignImageReadbackPending();
        [[nodiscard]] bool valid() const noexcept;
        // completed() remains the diagnostic readback-consumption state.
        // Fence retirement is independent so terminal ownership need not wait.
        [[nodiscard]] bool completed() const noexcept { return completionConsumed; }
        [[nodiscard]] bool retirementObserved() const noexcept { return fenceRetired; }
        [[nodiscard]] bool terminalReady() const noexcept;
        [[nodiscard]] RuntimeForeignImageView imageView() const noexcept;
        [[nodiscard]] VkFence retirementFence() const noexcept { return fence; }
        [[nodiscard]] VkSemaphore importedWaitSemaphore() const noexcept {
            return imported.handle();
        }
        [[nodiscard]] VkSemaphore returnedSignalSemaphore() const noexcept {
            return borrowedSignalSemaphore;
        }
        [[nodiscard]] bool submissionAccepted() const noexcept { return submitted; }
        [[nodiscard]] bool handoffRequested() const noexcept { return handoffRequestedValue; }
        [[nodiscard]] bool ownershipReleaseRequired() const noexcept { return releaseRequired; }
    private:
        friend class RuntimeImageEndpoint;
        std::unique_ptr<RuntimeImageEndpoint> owner;
        SyncFdSemaphore imported;
        VkFence fence{};
        std::weak_ptr<const uint8_t> lifetime;
        bool submitted{};
        bool completionConsumed{};
        bool fenceRetired{};
        bool failed{};
        bool handoffRequestedValue{};
        bool releaseRequired{};
        uint32_t sourceQueueFamilyValue{VK_QUEUE_FAMILY_IGNORED};
        uint32_t destinationQueueFamilyValue{VK_QUEUE_FAMILY_IGNORED};
        VkSemaphore borrowedSignalSemaphore{VK_NULL_HANDLE};
        void reset() noexcept;
        [[nodiscard]] RuntimeImageEndpoint releaseOwner() noexcept;
    };

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
        PFN_vkGetFenceStatus GetFenceStatus{};
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
        PhysicalDeviceIdentity identity{};
        PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2{};
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2{};
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2KHR{};
        PFN_vkCreateImage CreateImage{};
        PFN_vkDestroyImage DestroyImage{};
        PFN_vkGetImageMemoryRequirements2 GetImageMemoryRequirements2{};
        PFN_vkBindImageMemory BindImageMemory{};
        PFN_vkCmdClearColorImage CmdClearColorImage{};
        PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer{};
        PFN_vkCmdBlitImage CmdBlitImage{};
        PFN_vkCmdCopyImage CmdCopyImage{};
        PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout{};
        PFN_vkGetImageDrmFormatModifierPropertiesEXT GetImageDrmFormatModifierPropertiesEXT{};
        PFN_vkGetMemoryFdKHR GetMemoryFdKHR{};
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
        VkExtent2D extent{256, 256};
        VkFormat format{VK_FORMAT_B8G8R8A8_UNORM};
        VkImageUsageFlags usage{VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VkDeviceSize backingSize{};
        uint32_t fourcc{};
        uint64_t modifier{};
        uint32_t planeCount{};
        VkSubresourceLayout plane{};
        std::vector<VkSubresourceLayout> planes;
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
        static void executeRealFrameTransport(RuntimeImageEndpoint& imageA,
            RuntimeImageEndpoint& imageB, VkImage sourceImage,
            VkExtent2D sourceExtent, VkSemaphore bridgeWait);
        [[nodiscard]] static SyncFdPayload submitRealFrameTransportA(
            RuntimeImageEndpoint& imageA, VkImage sourceImage,
            VkExtent2D sourceExtent, VkSemaphore bridgeWait);
        [[nodiscard]] static std::vector<uint8_t> readForeignImage(
            RuntimeImageEndpoint& imageA, SyncFdPayload payload);
        [[nodiscard]] static RuntimeForeignImageReadbackPending submitForeignImageReadback(
            RuntimeImageEndpoint&& imageA, SyncFdPayload payload,
            std::optional<RuntimeForeignImageHandoffInfo> handoff = std::nullopt);
        [[nodiscard]] static std::vector<uint8_t> completeForeignImageReadback(
            RuntimeForeignImageReadbackPending& pending);
        // Single nonblocking fence observation for lifecycle retirement.
        [[nodiscard]] static bool tryRetireForeignImageReadback(
            RuntimeForeignImageReadbackPending& pending);
    private:
        friend struct RuntimeImageEndpointTestAccess;
        friend class RuntimeForeignImageReadbackPending;
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
        VkExtent2D extent{256, 256};
        VkFormat format{VK_FORMAT_UNDEFINED};
        uint64_t modifier{};
        std::shared_ptr<const uint8_t> lifetime{std::make_shared<const uint8_t>(0)};
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
