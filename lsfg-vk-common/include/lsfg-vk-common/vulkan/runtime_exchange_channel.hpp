/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "external_buffer_transport.hpp"
#include "external_semaphore_sync.hpp"
#include "runtime_device_pair.hpp"

#include <cstdint>

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
        PFN_vkDeviceWaitIdle DeviceWaitIdle{};
    };

    struct RuntimeExchangeChannelInfo {
        VkDeviceSize logicalSize{};
        VkDeviceSize backingSize{};
        VkBufferUsageFlags usage{};
    };

    struct RuntimeExchangeSyncDiagnostics {
        bool renderToGenerationSentinel{};
        bool generationToRenderSentinel{};
        bool hostWaitBeforeFinalSubmit{};
    };

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
