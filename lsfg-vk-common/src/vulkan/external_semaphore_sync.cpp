/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/external_semaphore_sync.hpp"

#include <utility>

#include <vulkan/vulkan_core.h>

using namespace vk;

ExternalSemaphoreError::ExternalSemaphoreError(
        ExternalSemaphoreFailure failure, VkResult result, const char* message) :
    std::runtime_error(message), failureCode(failure), vkResult(result) {}

void SyncFdPayload::invalidateAfterImport() noexcept {
    if (!this->sentinelPayload)
        static_cast<void>(this->fd.release());
    this->validPayload = false;
    this->sentinelPayload = false;
}

SyncFdSemaphore::SyncFdSemaphore(
        VkDevice device,
        PFN_vkDestroySemaphore destroySemaphore,
        VkSemaphore semaphore) noexcept :
    deviceHandle(device),
    destroySemaphore(destroySemaphore),
    semaphoreHandle(semaphore) {}

SyncFdSemaphore::SyncFdSemaphore(SyncFdSemaphore&& other) noexcept :
    deviceHandle(std::exchange(other.deviceHandle, VK_NULL_HANDLE)),
    destroySemaphore(std::exchange(other.destroySemaphore, nullptr)),
    semaphoreHandle(std::exchange(other.semaphoreHandle, VK_NULL_HANDLE)) {}

SyncFdSemaphore& SyncFdSemaphore::operator=(SyncFdSemaphore&& other) noexcept {
    if (this == &other)
        return *this;

    this->reset();
    this->deviceHandle = std::exchange(other.deviceHandle, VK_NULL_HANDLE);
    this->destroySemaphore = std::exchange(other.destroySemaphore, nullptr);
    this->semaphoreHandle = std::exchange(other.semaphoreHandle, VK_NULL_HANDLE);
    return *this;
}

SyncFdSemaphore::~SyncFdSemaphore() {
    this->reset();
}

void SyncFdSemaphore::reset() noexcept {
    if (this->semaphoreHandle != VK_NULL_HANDLE && this->destroySemaphore)
        this->destroySemaphore(this->deviceHandle, this->semaphoreHandle, nullptr);

    this->semaphoreHandle = VK_NULL_HANDLE;
    this->deviceHandle = VK_NULL_HANDLE;
    this->destroySemaphore = nullptr;
}

namespace {
    VkSemaphore createRawSyncFdSemaphore(
            const ExternalSemaphoreDevice& device, bool exportable) {
        VkExportSemaphoreCreateInfo exportInfo{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO
        };
        exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

        VkSemaphoreCreateInfo createInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        createInfo.pNext = exportable ? &exportInfo : nullptr;

        VkSemaphore semaphore{};
        const VkResult result = device.funcs.CreateSemaphore(
            device.device, &createInfo, nullptr, &semaphore);
        if (result != VK_SUCCESS)
            throw ExternalSemaphoreError(
                ExternalSemaphoreFailure::SemaphoreCreate,
                result, "SYNC_FD semaphore creation failed");
        return semaphore;
    }
}

SyncFdSemaphore vk::createExportableSyncFdSemaphore(
        const ExternalSemaphoreDevice& device) {
    return SyncFdSemaphore{
        device.device,
        device.funcs.DestroySemaphore,
        createRawSyncFdSemaphore(device, true)
    };
}

SyncFdSemaphore vk::createSyncFdImportSemaphore(
        const ExternalSemaphoreDevice& device) {
    return SyncFdSemaphore{
        device.device,
        device.funcs.DestroySemaphore,
        createRawSyncFdSemaphore(device, false)
    };
}

SyncFdPayload vk::exportSyncFd(
        const ExternalSemaphoreDevice& device, VkSemaphore semaphore) {
    VkSemaphoreGetFdInfoKHR info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR
    };
    info.semaphore = semaphore;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    int nativeFd = -2;
    const VkResult result = device.funcs.GetSemaphoreFdKHR(
        device.device, &info, &nativeFd);
    if (result != VK_SUCCESS)
        throw ExternalSemaphoreError(
            ExternalSemaphoreFailure::SyncFdExport,
            result, "SYNC_FD export failed");
    if (nativeFd < -1)
        throw ExternalSemaphoreError(
            ExternalSemaphoreFailure::InvalidSyncFdPayload,
            VK_SUCCESS, "SYNC_FD export returned an invalid descriptor");

    if (nativeFd == -1)
        return SyncFdPayload{true, ls::OwnedFd{}};
    return SyncFdPayload{false, ls::OwnedFd{nativeFd}};
}

void vk::importSyncFdTemporary(
        const ExternalSemaphoreDevice& device,
        VkSemaphore semaphore,
        SyncFdPayload& payload) {
    if (!payload.valid() || payload.nativeFd() < -1)
        throw ExternalSemaphoreError(
            ExternalSemaphoreFailure::InvalidSyncFdPayload,
            VK_SUCCESS, "invalid SYNC_FD payload state");

    VkImportSemaphoreFdInfoKHR info{
        VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR
    };
    info.semaphore = semaphore;
    info.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    info.fd = payload.nativeFd();

    const VkResult result = device.funcs.ImportSemaphoreFdKHR(
        device.device, &info);
    if (result != VK_SUCCESS)
        throw ExternalSemaphoreError(
            ExternalSemaphoreFailure::SyncFdImport,
            result, "temporary SYNC_FD import failed");

    payload.invalidateAfterImport();
}
