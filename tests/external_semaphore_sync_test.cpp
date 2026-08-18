/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/external_semaphore_sync.hpp"

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <type_traits>
#include <vector>

#include <unistd.h>
#include <vulkan/vulkan_core.h>

namespace {
    std::vector<bool> createdExportable;
    std::vector<VkSemaphore> destroyed;
    uintptr_t nextSemaphoreValue{1};
    int exportedNativeFd{-1};
    VkResult getResult{VK_SUCCESS};
    VkResult importResult{VK_SUCCESS};
    int lastImportedFd{-2};

    template<typename T>
    T fakeHandle(uintptr_t value) {
        if constexpr (std::is_pointer_v<T>)
            return reinterpret_cast<T>(value);
        else
            return static_cast<T>(value);
    }

    VkResult createSemaphore(VkDevice,
            const VkSemaphoreCreateInfo* info,
            const VkAllocationCallbacks*,
            VkSemaphore* semaphore) {
        assert(info);
        assert(info->sType == VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        assert(info->flags == 0);

        const bool exportable = info->pNext != nullptr;
        createdExportable.push_back(exportable);
        if (exportable) {
            const auto* exportInfo = static_cast<const VkExportSemaphoreCreateInfo*>(
                info->pNext);
            assert(exportInfo->sType == VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
            assert(exportInfo->handleTypes ==
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        }

        *semaphore = fakeHandle<VkSemaphore>(nextSemaphoreValue++);
        return VK_SUCCESS;
    }

    void destroySemaphore(VkDevice, VkSemaphore semaphore,
            const VkAllocationCallbacks*) {
        destroyed.push_back(semaphore);
    }

    VkResult getSemaphoreFd(VkDevice,
            const VkSemaphoreGetFdInfoKHR* info,
            int* fd) {
        assert(info);
        assert(info->sType == VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
        assert(info->semaphore != VK_NULL_HANDLE);
        assert(info->handleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        if (getResult != VK_SUCCESS)
            return getResult;
        *fd = exportedNativeFd;
        return VK_SUCCESS;
    }

    VkResult importSemaphoreFd(VkDevice,
            const VkImportSemaphoreFdInfoKHR* info) {
        assert(info);
        assert(info->sType == VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR);
        assert(info->semaphore != VK_NULL_HANDLE);
        assert(info->flags == VK_SEMAPHORE_IMPORT_TEMPORARY_BIT);
        assert(info->handleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        lastImportedFd = info->fd;

        if (importResult != VK_SUCCESS)
            return importResult;

        // Simulate Vulkan consuming a real fd on successful import.
        if (info->fd >= 0)
            assert(::close(info->fd) == 0);
        return VK_SUCCESS;
    }

    vk::ExternalSemaphoreDevice makeDevice() {
        return {
            fakeHandle<VkDevice>(100),
            {
                createSemaphore,
                destroySemaphore,
                getSemaphoreFd,
                importSemaphoreFd
            }
        };
    }

    void resetState() {
        createdExportable.clear();
        destroyed.clear();
        nextSemaphoreValue = 1;
        exportedNativeFd = -1;
        getResult = VK_SUCCESS;
        importResult = VK_SUCCESS;
        lastImportedFd = -2;
    }
}

int main() {
    resetState();
    {
        auto exportSemaphore = vk::createExportableSyncFdSemaphore(makeDevice());
        auto importSemaphore = vk::createSyncFdImportSemaphore(makeDevice());
        assert(exportSemaphore);
        assert(importSemaphore);
        assert((createdExportable == std::vector<bool>{true, false}));
    }
    assert(destroyed.size() == 2);

    resetState();
    int successPipe[2]{};
    assert(::pipe(successPipe) == 0);
    ::close(successPipe[1]);
    const int successfulFd = successPipe[0];
    exportedNativeFd = successfulFd;
    {
        auto exportSemaphore = vk::createExportableSyncFdSemaphore(makeDevice());
        auto importSemaphore = vk::createSyncFdImportSemaphore(makeDevice());
        auto payload = vk::exportSyncFd(makeDevice(), exportSemaphore.handle());
        assert(payload.valid());
        assert(!payload.sentinel());
        assert(payload.nativeFd() == successfulFd);

        vk::importSyncFdTemporary(
            makeDevice(), importSemaphore.handle(), payload);
        assert(!payload.valid());
        assert(lastImportedFd == successfulFd);

        errno = 0;
        assert(::fcntl(successfulFd, F_GETFD) == -1);
        assert(errno == EBADF);
    }

    resetState();
    exportedNativeFd = -1;
    {
        auto exportSemaphore = vk::createExportableSyncFdSemaphore(makeDevice());
        auto importSemaphore = vk::createSyncFdImportSemaphore(makeDevice());
        auto payload = vk::exportSyncFd(makeDevice(), exportSemaphore.handle());
        assert(payload.valid());
        assert(payload.sentinel());
        assert(payload.nativeFd() == -1);

        vk::importSyncFdTemporary(
            makeDevice(), importSemaphore.handle(), payload);
        assert(!payload.valid());
        assert(lastImportedFd == -1);
    }

    resetState();
    int failurePipe[2]{};
    assert(::pipe(failurePipe) == 0);
    ::close(failurePipe[1]);
    const int failedFd = failurePipe[0];
    exportedNativeFd = failedFd;
    importResult = static_cast<VkResult>(-7);
    {
        auto exportSemaphore = vk::createExportableSyncFdSemaphore(makeDevice());
        auto importSemaphore = vk::createSyncFdImportSemaphore(makeDevice());
        auto payload = vk::exportSyncFd(makeDevice(), exportSemaphore.handle());

        bool failed{};
        try {
            vk::importSyncFdTemporary(
                makeDevice(), importSemaphore.handle(), payload);
        } catch (const vk::ExternalSemaphoreError& error) {
            failed = true;
            assert(error.failure() == vk::ExternalSemaphoreFailure::SyncFdImport);
            assert(error.result() == static_cast<VkResult>(-7));
        }
        assert(failed);
        assert(payload.valid());
        assert(payload.nativeFd() == failedFd);
        assert(::fcntl(failedFd, F_GETFD) != -1);
    }
    errno = 0;
    assert(::fcntl(failedFd, F_GETFD) == -1);
    assert(errno == EBADF);

    resetState();
    {
        auto importSemaphore = vk::createSyncFdImportSemaphore(makeDevice());
        vk::SyncFdPayload payload;
        bool failed{};
        try {
            vk::importSyncFdTemporary(
                makeDevice(), importSemaphore.handle(), payload);
        } catch (const vk::ExternalSemaphoreError& error) {
            failed = true;
            assert(error.failure() ==
                vk::ExternalSemaphoreFailure::InvalidSyncFdPayload);
            assert(error.result() == VK_SUCCESS);
        }
        assert(failed);
        assert(!payload.valid());
    }

    resetState();
    exportedNativeFd = -2;
    {
        auto exportSemaphore = vk::createExportableSyncFdSemaphore(makeDevice());
        bool failed{};
        try {
            static_cast<void>(vk::exportSyncFd(
                makeDevice(), exportSemaphore.handle()));
        } catch (const vk::ExternalSemaphoreError& error) {
            failed = true;
            assert(error.failure() ==
                vk::ExternalSemaphoreFailure::InvalidSyncFdPayload);
            assert(error.result() == VK_SUCCESS);
        }
        assert(failed);
    }
}
