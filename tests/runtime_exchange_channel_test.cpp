/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <unistd.h>
#include <vulkan/vulkan_core.h>

namespace {
    template<typename T>
    T fakeHandle(uintptr_t value) {
        if constexpr (std::is_pointer_v<T>)
            return reinterpret_cast<T>(value);
        else
            return static_cast<T>(value);
    }

    const VkDevice DEVICE_A = fakeHandle<VkDevice>(0xA001);
    const VkDevice DEVICE_B = fakeHandle<VkDevice>(0xB001);
    const VkQueue QUEUE_A = fakeHandle<VkQueue>(0xA101);
    const VkQueue QUEUE_B = fakeHandle<VkQueue>(0xB101);

    struct Trace {
        std::vector<std::string> events;
        uint32_t bufferCreates{};
        uint32_t semaphoreCreates{};
        uint32_t fdExports{};
        uint32_t tempImports{};
        uint32_t hostWaits{};
        uint32_t deviceIdleWaits{};
    } trace;

    VkResult createBuffer(VkDevice, const VkBufferCreateInfo* info,
            const VkAllocationCallbacks*, VkBuffer* buffer) {
        assert(info);
        assert(info->size == 64 * 1024);
        assert((info->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0);
        assert((info->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0);
        const auto* external = static_cast<const VkExternalMemoryBufferCreateInfo*>(info->pNext);
        assert(external);
        assert(external->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
        assert(external->handleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        *buffer = fakeHandle<VkBuffer>(0xC000 + ++trace.bufferCreates);
        return VK_SUCCESS;
    }

    void destroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}

    void getBufferMemoryRequirements(VkDevice device, VkBuffer, VkMemoryRequirements* req) {
        *req = {
            .size = 64 * 1024,
            .alignment = 16,
            .memoryTypeBits = device == DEVICE_A ? 0x8U : 0x1U
        };
    }

    VkResult getMemoryFdProperties(VkDevice device,
            VkExternalMemoryHandleTypeFlagBits type,
            int fd, VkMemoryFdPropertiesKHR* properties) {
        assert(type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        assert(fd >= 0);
        properties->memoryTypeBits = device == DEVICE_A ? 0x8U : 0x1U;
        return VK_SUCCESS;
    }

    VkResult allocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
            const VkAllocationCallbacks*, VkDeviceMemory* memory) {
        assert(info);
        const auto* import = static_cast<const VkImportMemoryFdInfoKHR*>(info->pNext);
        assert(import);
        assert(import->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
        assert(import->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        assert(import->fd >= 0);
        assert((device == DEVICE_A && info->memoryTypeIndex == 3)
            || (device == DEVICE_B && info->memoryTypeIndex == 0));
        // Successful Vulkan import consumes the userspace FD.
        ::close(import->fd);
        *memory = fakeHandle<VkDeviceMemory>(device == DEVICE_A ? 0xDA01 : 0xDB01);
        return VK_SUCCESS;
    }

    void freeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
    VkResult bindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
        return VK_SUCCESS;
    }

    VkResult createSemaphore(VkDevice, const VkSemaphoreCreateInfo* info,
            const VkAllocationCallbacks*, VkSemaphore* semaphore) {
        assert(info);
        if (info->pNext) {
            const auto* exportInfo = static_cast<const VkExportSemaphoreCreateInfo*>(info->pNext);
            assert(exportInfo->sType == VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
            assert(exportInfo->handleTypes == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        }
        *semaphore = fakeHandle<VkSemaphore>(0xE000 + ++trace.semaphoreCreates);
        return VK_SUCCESS;
    }

    void destroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}

    VkResult getSemaphoreFd(VkDevice, const VkSemaphoreGetFdInfoKHR* info, int* fd) {
        assert(info);
        assert(info->handleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        ++trace.fdExports;
        if (trace.fdExports == 1) {
            int pipeFds[2]{};
            assert(::pipe(pipeFds) == 0);
            ::close(pipeFds[1]);
            *fd = pipeFds[0];
            trace.events.emplace_back("EXPORT_A_FD");
        } else {
            *fd = -1;
            trace.events.emplace_back("EXPORT_B_SENTINEL");
        }
        return VK_SUCCESS;
    }

    VkResult importSemaphoreFd(VkDevice device, const VkImportSemaphoreFdInfoKHR* info) {
        assert(info);
        assert(info->handleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
        assert(info->flags == VK_SEMAPHORE_IMPORT_TEMPORARY_BIT);
        ++trace.tempImports;
        if (info->fd >= 0)
            ::close(info->fd); // emulate ownership transfer to Vulkan
        trace.events.emplace_back(device == DEVICE_B ? "IMPORT_B" : "IMPORT_A");
        return VK_SUCCESS;
    }

    VkResult queueSubmit(VkQueue queue, uint32_t count,
            const VkSubmitInfo* submits, VkFence fence) {
        assert(count == 1);
        assert(submits);
        const auto& submit = submits[0];
        if (queue == QUEUE_A && submit.waitSemaphoreCount == 0
                && submit.signalSemaphoreCount == 1 && fence == VK_NULL_HANDLE) {
            trace.events.emplace_back("SUBMIT_A_SIGNAL");
        } else if (queue == QUEUE_B && submit.waitSemaphoreCount == 1
                && submit.signalSemaphoreCount == 1 && fence == VK_NULL_HANDLE) {
            assert(submit.pWaitDstStageMask);
            trace.events.emplace_back("SUBMIT_B_WAIT_SIGNAL");
        } else if (queue == QUEUE_A && submit.waitSemaphoreCount == 1
                && submit.signalSemaphoreCount == 0 && fence != VK_NULL_HANDLE) {
            assert(submit.pWaitDstStageMask);
            trace.events.emplace_back("SUBMIT_A_FINAL_WAIT");
        } else {
            assert(false && "unexpected submit shape");
        }
        return VK_SUCCESS;
    }

    VkResult createFence(VkDevice, const VkFenceCreateInfo*,
            const VkAllocationCallbacks*, VkFence* fence) {
        *fence = fakeHandle<VkFence>(0xF001);
        return VK_SUCCESS;
    }

    void destroyFence(VkDevice, VkFence, const VkAllocationCallbacks*) {}

    VkResult waitForFences(VkDevice device, uint32_t count,
            const VkFence* fences, VkBool32 waitAll, uint64_t) {
        assert(device == DEVICE_A);
        assert(count == 1 && fences && *fences != VK_NULL_HANDLE);
        assert(waitAll == VK_TRUE);
        ++trace.hostWaits;
        trace.events.emplace_back("HOST_WAIT_FINAL");
        return VK_SUCCESS;
    }

    VkResult deviceWaitIdle(VkDevice) {
        ++trace.deviceIdleWaits;
        return VK_SUCCESS;
    }

    vk::RuntimeExchangeEndpoint makeEndpoint(VkDevice device, VkQueue queue, uint32_t memoryType) {
        VkPhysicalDeviceMemoryProperties properties{};
        properties.memoryTypeCount = memoryType + 1;
        properties.memoryTypes[memoryType].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        return {
            .bufferDevice = {
                .device = device,
                .memoryProperties = properties,
                .funcs = {
                    .CreateBuffer = createBuffer,
                    .DestroyBuffer = destroyBuffer,
                    .GetBufferMemoryRequirements = getBufferMemoryRequirements,
                    .GetMemoryFdPropertiesKHR = getMemoryFdProperties,
                    .AllocateMemory = allocateMemory,
                    .FreeMemory = freeMemory,
                    .BindBufferMemory = bindBufferMemory
                }
            },
            .semaphoreDevice = {
                .device = device,
                .funcs = {
                    .CreateSemaphore = createSemaphore,
                    .DestroySemaphore = destroySemaphore,
                    .GetSemaphoreFdKHR = getSemaphoreFd,
                    .ImportSemaphoreFdKHR = importSemaphoreFd
                }
            },
            .queue = queue,
            .queueFamilyIndex = 0,
            .QueueSubmit = queueSubmit,
            .CreateFence = createFence,
            .DestroyFence = destroyFence,
            .WaitForFences = waitForFences,
            .DeviceWaitIdle = deviceWaitIdle
        };
    }

    vk::PhysicalDeviceIdentity identity(const char* name, uint8_t uuidByte) {
        vk::PhysicalDeviceIdentity out{};
        out.name = name;
        out.deviceUuid[0] = uuidByte;
        return out;
    }

    ls::OwnedFd duplicatePipeReadEnd() {
        int pipeFds[2]{};
        assert(::pipe(pipeFds) == 0);
        ::close(pipeFds[1]);
        return ls::OwnedFd(pipeFds[0]);
    }
}

int main() {
    const vk::RuntimeDevicePair pair{
        .render = { identity("A", 0xA1), vk::RuntimeDeviceOwnership::ApplicationManaged },
        .generation = { identity("B", 0xB1), vk::RuntimeDeviceOwnership::BackendManaged },
        .mode = vk::RuntimeDevicePairMode::CrossPhysicalDevice,
        .requestedGenerationSelector = std::string("B")
    };

    auto channel = vk::createRuntimeExchangeChannel(
        pair,
        makeEndpoint(DEVICE_A, QUEUE_A, 3),
        makeEndpoint(DEVICE_B, QUEUE_B, 0),
        duplicatePipeReadEnd(),
        duplicatePipeReadEnd(),
        {
            .logicalSize = 64 * 1024,
            .backingSize = 64 * 1024,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        });

    assert(channel.renderBuffer() != VK_NULL_HANDLE);
    assert(channel.generationBuffer() != VK_NULL_HANDLE);
    assert(channel.renderBufferDiagnostics().memoryTypeIndex == 3);
    assert(channel.generationBufferDiagnostics().memoryTypeIndex == 0);

    const auto diagnostics = channel.validateSyncRoundTrip();
    assert(!diagnostics.renderToGenerationSentinel);
    assert(diagnostics.generationToRenderSentinel);
    assert(!diagnostics.hostWaitBeforeFinalSubmit);
    assert(trace.bufferCreates == 2);
    assert(trace.semaphoreCreates == 4);
    assert(trace.fdExports == 2);
    assert(trace.tempImports == 2);
    assert(trace.hostWaits == 1);
    assert(trace.deviceIdleWaits == 0);

    const std::vector<std::string> expected{
        "SUBMIT_A_SIGNAL",
        "EXPORT_A_FD",
        "IMPORT_B",
        "SUBMIT_B_WAIT_SIGNAL",
        "EXPORT_B_SENTINEL",
        "IMPORT_A",
        "SUBMIT_A_FINAL_WAIT",
        "HOST_WAIT_FINAL"
    };
    assert(trace.events == expected);

    bool rejectedSameDevice = false;
    try {
        auto samePair = pair;
        samePair.generation.identity = samePair.render.identity;
        samePair.mode = vk::RuntimeDevicePairMode::SamePhysicalDevice;
        auto invalid = vk::createRuntimeExchangeChannel(
            samePair,
            makeEndpoint(DEVICE_A, QUEUE_A, 3),
            makeEndpoint(DEVICE_A, QUEUE_A, 3),
            duplicatePipeReadEnd(), duplicatePipeReadEnd(),
            {64 * 1024, 64 * 1024,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
        static_cast<void>(invalid);
    } catch (const std::invalid_argument&) {
        rejectedSameDevice = true;
    }
    assert(rejectedSameDevice);

    return 0;
}
