/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "lsfg-vk-common/vulkan/external_buffer_transport.hpp"

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <type_traits>
#include <vector>

#include <unistd.h>
#include <vulkan/vulkan_core.h>

namespace {
    enum class Event {
        DestroyBuffer,
        FreeMemory
    };

    std::vector<Event> events;
    bool failAllocation{};

    template<typename T>
    T fakeHandle(uintptr_t value) {
        if constexpr (std::is_pointer_v<T>)
            return reinterpret_cast<T>(value);
        else
            return static_cast<T>(value);
    }

    VkResult createBuffer(VkDevice, const VkBufferCreateInfo* info,
            const VkAllocationCallbacks*, VkBuffer* buffer) {
        assert(info);
        assert(info->size == 64 * 1024);
        assert(info->pNext);
        const auto* external = static_cast<const VkExternalMemoryBufferCreateInfo*>(
            info->pNext);
        assert(external->handleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        *buffer = fakeHandle<VkBuffer>(1);
        return VK_SUCCESS;
    }

    void destroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
        assert(buffer == fakeHandle<VkBuffer>(1));
        events.push_back(Event::DestroyBuffer);
    }

    void getBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements* requirements) {
        requirements->size = 64 * 1024;
        requirements->alignment = 16;
        requirements->memoryTypeBits = 0b1100;
    }

    VkResult getMemoryFdProperties(VkDevice,
            VkExternalMemoryHandleTypeFlagBits handleType,
            int fd, VkMemoryFdPropertiesKHR* properties) {
        assert(handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        assert(fd >= 0);
        properties->memoryTypeBits = 0b1010;
        return VK_SUCCESS;
    }

    VkResult allocateMemory(VkDevice, const VkMemoryAllocateInfo* info,
            const VkAllocationCallbacks*, VkDeviceMemory* memory) {
        assert(info);
        assert(info->allocationSize == 64 * 1024);
        assert(info->memoryTypeIndex == 3);
        const auto* import = static_cast<const VkImportMemoryFdInfoKHR*>(info->pNext);
        assert(import);
        assert(import->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        assert(import->fd >= 0);

        if (failAllocation)
            return static_cast<VkResult>(-2);

        // Simulate Vulkan's successful fd-import ownership transfer.
        assert(::close(import->fd) == 0);
        *memory = fakeHandle<VkDeviceMemory>(2);
        return VK_SUCCESS;
    }

    void freeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
        assert(memory == fakeHandle<VkDeviceMemory>(2));
        events.push_back(Event::FreeMemory);
    }

    VkResult bindBufferMemory(VkDevice, VkBuffer buffer,
            VkDeviceMemory memory, VkDeviceSize offset) {
        assert(buffer == fakeHandle<VkBuffer>(1));
        assert(memory == fakeHandle<VkDeviceMemory>(2));
        assert(offset == 0);
        return VK_SUCCESS;
    }

    vk::ExternalBufferDevice makeDevice() {
        VkPhysicalDeviceMemoryProperties properties{};
        properties.memoryTypeCount = 4;
        properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        properties.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        properties.memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        properties.memoryTypes[3].propertyFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        return {
            fakeHandle<VkDevice>(3),
            properties,
            {
                createBuffer,
                destroyBuffer,
                getBufferMemoryRequirements,
                getMemoryFdProperties,
                allocateMemory,
                freeMemory,
                bindBufferMemory
            }
        };
    }

    vk::ExternalBufferImportInfo makeImportInfo() {
        return {
            64 * 1024,
            64 * 1024,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0,
            0
        };
    }
}

int main() {
    assert(vk::intersectExternalBufferMemoryTypes(0b1010, 0b1100) == 0b1000);

    {
        auto device = makeDevice();
        const auto selected = vk::selectExternalBufferMemoryType(
            0b1000, device.memoryProperties, 0);
        assert(selected && *selected == 3);
    }

    events.clear();
    failAllocation = false;
    int successPipe[2]{};
    assert(::pipe(successPipe) == 0);
    ::close(successPipe[1]);
    const int importedFd = successPipe[0];
    {
        auto imported = vk::importDmaBufBuffer(
            makeDevice(), ls::OwnedFd(importedFd), makeImportInfo());
        assert(imported);
        const auto& diagnostics = imported.diagnostics();
        assert(diagnostics.requirements.size == 64 * 1024);
        assert(diagnostics.fdMemoryTypeBits == 0b1010);
        assert(diagnostics.usableMemoryTypeBits == 0b1000);
        assert(diagnostics.memoryTypeIndex == 3);

        errno = 0;
        assert(::fcntl(importedFd, F_GETFD) == -1);
        assert(errno == EBADF);
    }
    assert((events == std::vector<Event>{Event::DestroyBuffer, Event::FreeMemory}));

    events.clear();
    failAllocation = true;
    int failurePipe[2]{};
    assert(::pipe(failurePipe) == 0);
    ::close(failurePipe[1]);
    const int failedFd = failurePipe[0];
    bool failed{};
    try {
        static_cast<void>(vk::importDmaBufBuffer(
            makeDevice(), ls::OwnedFd(failedFd), makeImportInfo()));
    } catch (const vk::ExternalBufferImportError& error) {
        failed = true;
        assert(error.failure() == vk::ExternalBufferImportFailure::MemoryImport);
        assert(error.result() == static_cast<VkResult>(-2));
    }
    assert(failed);
    errno = 0;
    assert(::fcntl(failedFd, F_GETFD) == -1);
    assert(errno == EBADF);
    assert((events == std::vector<Event>{Event::DestroyBuffer}));
}
