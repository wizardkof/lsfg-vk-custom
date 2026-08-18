/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../helpers/owned_fd.hpp"

#include <stdexcept>
#include <utility>

#include <vulkan/vulkan_core.h>

namespace vk {

    // Minimal Vulkan dispatch needed for binary semaphore SYNC_FD export/import.
    // The abstraction deliberately owns only semaphore/FD mechanics; queue
    // submission ordering remains the caller's responsibility.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    struct ExternalSemaphoreDeviceFuncs {
        PFN_vkCreateSemaphore CreateSemaphore{};
        PFN_vkDestroySemaphore DestroySemaphore{};
        PFN_vkGetSemaphoreFdKHR GetSemaphoreFdKHR{};
        PFN_vkImportSemaphoreFdKHR ImportSemaphoreFdKHR{};
    };

    struct ExternalSemaphoreDevice {
        VkDevice device{};
        ExternalSemaphoreDeviceFuncs funcs{};
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    enum class ExternalSemaphoreFailure {
        SemaphoreCreate,
        SyncFdExport,
        InvalidSyncFdPayload,
        SyncFdImport
    };

    class ExternalSemaphoreError final : public std::runtime_error {
    public:
        ExternalSemaphoreError(ExternalSemaphoreFailure failure,
            VkResult result, const char* message);

        [[nodiscard]] ExternalSemaphoreFailure failure() const noexcept {
            return this->failureCode;
        }
        [[nodiscard]] VkResult result() const noexcept { return this->vkResult; }

    private:
        ExternalSemaphoreFailure failureCode;
        VkResult vkResult;
    };

    class SyncFdPayload {
    public:
        SyncFdPayload() noexcept = default;
        SyncFdPayload(const SyncFdPayload&) = delete;
        SyncFdPayload& operator=(const SyncFdPayload&) = delete;
        SyncFdPayload(SyncFdPayload&&) noexcept = default;
        SyncFdPayload& operator=(SyncFdPayload&&) noexcept = default;

        [[nodiscard]] bool valid() const noexcept { return this->validPayload; }
        [[nodiscard]] bool sentinel() const noexcept {
            return this->validPayload && this->sentinelPayload;
        }
        [[nodiscard]] int nativeFd() const noexcept {
            return this->sentinel() ? -1 : this->fd.get();
        }

    private:
        friend SyncFdPayload exportSyncFd(
            const ExternalSemaphoreDevice&, VkSemaphore);
        friend void importSyncFdTemporary(
            const ExternalSemaphoreDevice&, VkSemaphore, SyncFdPayload&);

        SyncFdPayload(bool sentinel, ls::OwnedFd fd) noexcept :
            validPayload(true), sentinelPayload(sentinel), fd(std::move(fd)) {}

        void invalidateAfterImport() noexcept;

        bool validPayload{};
        bool sentinelPayload{};
        ls::OwnedFd fd;
    };

    class SyncFdSemaphore {
    public:
        SyncFdSemaphore() noexcept = default;
        SyncFdSemaphore(const SyncFdSemaphore&) = delete;
        SyncFdSemaphore& operator=(const SyncFdSemaphore&) = delete;
        SyncFdSemaphore(SyncFdSemaphore&& other) noexcept;
        SyncFdSemaphore& operator=(SyncFdSemaphore&& other) noexcept;
        ~SyncFdSemaphore();

        [[nodiscard]] VkSemaphore handle() const noexcept {
            return this->semaphoreHandle;
        }
        explicit operator bool() const noexcept {
            return this->semaphoreHandle != VK_NULL_HANDLE;
        }

    private:
        friend SyncFdSemaphore createExportableSyncFdSemaphore(
            const ExternalSemaphoreDevice&);
        friend SyncFdSemaphore createSyncFdImportSemaphore(
            const ExternalSemaphoreDevice&);

        SyncFdSemaphore(VkDevice device,
            PFN_vkDestroySemaphore destroySemaphore,
            VkSemaphore semaphore) noexcept;

        void reset() noexcept;

        VkDevice deviceHandle{};
        PFN_vkDestroySemaphore destroySemaphore{};
        VkSemaphore semaphoreHandle{};
    };

    /// Create a binary semaphore whose payload can be exported as SYNC_FD.
    [[nodiscard]] SyncFdSemaphore createExportableSyncFdSemaphore(
        const ExternalSemaphoreDevice& device);

    /// Create a local binary semaphore suitable for a temporary SYNC_FD import.
    [[nodiscard]] SyncFdSemaphore createSyncFdImportSemaphore(
        const ExternalSemaphoreDevice& device);

    /// Export the current binary semaphore payload as SYNC_FD. A valid result
    /// can carry either a real owned fd (>= 0) or the already-signaled -1 sentinel.
    [[nodiscard]] SyncFdPayload exportSyncFd(
        const ExternalSemaphoreDevice& device, VkSemaphore semaphore);

    /// Temporarily import a SYNC_FD payload into a local binary semaphore.
    /// Successful import consumes a real fd into Vulkan; -1 is a valid sentinel
    /// and has no userspace descriptor to close.
    void importSyncFdTemporary(
        const ExternalSemaphoreDevice& device,
        VkSemaphore semaphore,
        SyncFdPayload& payload);

}
