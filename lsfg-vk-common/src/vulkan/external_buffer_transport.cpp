/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/external_buffer_transport.hpp"

#include <utility>

#include <vulkan/vulkan_core.h>

using namespace vk;

ExternalBufferImportError::ExternalBufferImportError(
        ExternalBufferImportFailure failure, VkResult result, const char* message) :
    std::runtime_error(message), failureCode(failure), vkResult(result) {}

ImportedExternalBuffer::ImportedExternalBuffer(
        VkDevice device,
        PFN_vkDestroyBuffer destroyBuffer,
        PFN_vkFreeMemory freeMemory,
        VkBuffer buffer,
        VkDeviceMemory memory,
        ExternalBufferImportDiagnostics diagnostics) noexcept :
    deviceHandle(device),
    destroyBuffer(destroyBuffer),
    freeMemory(freeMemory),
    bufferHandle(buffer),
    memoryHandle(memory),
    importDiagnostics(diagnostics) {}

ImportedExternalBuffer::ImportedExternalBuffer(ImportedExternalBuffer&& other) noexcept :
    deviceHandle(std::exchange(other.deviceHandle, VK_NULL_HANDLE)),
    destroyBuffer(std::exchange(other.destroyBuffer, nullptr)),
    freeMemory(std::exchange(other.freeMemory, nullptr)),
    bufferHandle(std::exchange(other.bufferHandle, VK_NULL_HANDLE)),
    memoryHandle(std::exchange(other.memoryHandle, VK_NULL_HANDLE)),
    importDiagnostics(other.importDiagnostics) {}

ImportedExternalBuffer& ImportedExternalBuffer::operator=(
        ImportedExternalBuffer&& other) noexcept {
    if (this == &other)
        return *this;

    this->reset();
    this->deviceHandle = std::exchange(other.deviceHandle, VK_NULL_HANDLE);
    this->destroyBuffer = std::exchange(other.destroyBuffer, nullptr);
    this->freeMemory = std::exchange(other.freeMemory, nullptr);
    this->bufferHandle = std::exchange(other.bufferHandle, VK_NULL_HANDLE);
    this->memoryHandle = std::exchange(other.memoryHandle, VK_NULL_HANDLE);
    this->importDiagnostics = other.importDiagnostics;
    return *this;
}

ImportedExternalBuffer::~ImportedExternalBuffer() {
    this->reset();
}

void ImportedExternalBuffer::reset() noexcept {
    if (this->bufferHandle != VK_NULL_HANDLE && this->destroyBuffer)
        this->destroyBuffer(this->deviceHandle, this->bufferHandle, nullptr);
    if (this->memoryHandle != VK_NULL_HANDLE && this->freeMemory)
        this->freeMemory(this->deviceHandle, this->memoryHandle, nullptr);

    this->memoryHandle = VK_NULL_HANDLE;
    this->bufferHandle = VK_NULL_HANDLE;
    this->deviceHandle = VK_NULL_HANDLE;
}

uint32_t vk::intersectExternalBufferMemoryTypes(
        uint32_t fdMemoryTypeBits, uint32_t bufferMemoryTypeBits) noexcept {
    return fdMemoryTypeBits & bufferMemoryTypeBits;
}

std::optional<uint32_t> vk::selectExternalBufferMemoryType(
        uint32_t memoryTypeBits,
        const VkPhysicalDeviceMemoryProperties& memoryProperties,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred) noexcept {
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1U << i)) == 0)
                continue;

            const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((flags & required) != required)
                continue;
            if (pass == 0 && (flags & preferred) != preferred)
                continue;
            return i;
        }
    }
    return std::nullopt;
}

ImportedExternalBuffer vk::importDmaBufBuffer(
        const ExternalBufferDevice& device,
        ls::OwnedFd fd,
        const ExternalBufferImportInfo& info) {
    if (!fd)
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::InvalidFileDescriptor,
            VK_SUCCESS, "invalid DMA-BUF file descriptor");

    const VkExternalMemoryBufferCreateInfo externalInfo{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
    };
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &externalInfo,
        .size = info.logicalSize,
        .usage = info.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VkBuffer buffer{};
    VkResult result = device.funcs.CreateBuffer(
        device.device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS)
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::BufferCreate,
            result, "external buffer creation failed");

    VkMemoryRequirements requirements{};
    device.funcs.GetBufferMemoryRequirements(
        device.device, buffer, &requirements);

    VkMemoryFdPropertiesKHR fdProperties{
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR
    };
    result = device.funcs.GetMemoryFdPropertiesKHR(
        device.device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        fd.get(), &fdProperties);
    if (result != VK_SUCCESS) {
        device.funcs.DestroyBuffer(device.device, buffer, nullptr);
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::FdProperties,
            result, "DMA-BUF memory properties query failed");
    }

    if (info.backingSize < requirements.size) {
        device.funcs.DestroyBuffer(device.device, buffer, nullptr);
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::BackingTooSmall,
            VK_SUCCESS,
            "DMA-BUF backing is smaller than Vulkan buffer requirements");
    }

    const uint32_t usableBits = intersectExternalBufferMemoryTypes(
        fdProperties.memoryTypeBits, requirements.memoryTypeBits);
    const auto memoryType = selectExternalBufferMemoryType(
        usableBits, device.memoryProperties,
        info.requiredMemoryProperties, info.preferredMemoryProperties);
    if (!memoryType) {
        device.funcs.DestroyBuffer(device.device, buffer, nullptr);
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::NoMemoryTypeIntersection,
            VK_SUCCESS,
            "no compatible Vulkan memory type for DMA-BUF import");
    }

    const VkImportMemoryFdInfoKHR importInfo{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd.get()
    };
    const VkMemoryAllocateInfo allocationInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &importInfo,
        .allocationSize = info.backingSize,
        .memoryTypeIndex = *memoryType
    };

    VkDeviceMemory memory{};
    result = device.funcs.AllocateMemory(
        device.device, &allocationInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        device.funcs.DestroyBuffer(device.device, buffer, nullptr);
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::MemoryImport,
            result, "DMA-BUF memory import failed");
    }

    // Successful fd import transfers fd ownership to Vulkan.
    static_cast<void>(fd.release());

    result = device.funcs.BindBufferMemory(device.device, buffer, memory, 0);
    if (result != VK_SUCCESS) {
        device.funcs.FreeMemory(device.device, memory, nullptr);
        device.funcs.DestroyBuffer(device.device, buffer, nullptr);
        throw ExternalBufferImportError(
            ExternalBufferImportFailure::BufferBind,
            result, "imported DMA-BUF buffer bind failed");
    }

    ExternalBufferImportDiagnostics diagnostics{
        .requirements = requirements,
        .fdMemoryTypeBits = fdProperties.memoryTypeBits,
        .usableMemoryTypeBits = usableBits,
        .memoryTypeIndex = *memoryType,
        .memoryPropertyFlags =
            device.memoryProperties.memoryTypes[*memoryType].propertyFlags
    };

    return ImportedExternalBuffer{
        device.device,
        device.funcs.DestroyBuffer,
        device.funcs.FreeMemory,
        buffer,
        memory,
        diagnostics
    };
}
