/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "lsfg-vk-common/vulkan/external_memory_import.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include <stdexcept>
#include <iostream>

using namespace vk;

ImportedExternalMemory::ImportedExternalMemory(VkDevice device, PFN_vkFreeMemory free,
        VkDeviceMemory memory, uint32_t type, bool dedicated) noexcept
    : deviceHandle(device), freeMemory(free), memoryHandle(memory), typeIndex(type),
      dedicatedAllocation(dedicated) {}

ImportedExternalMemory::ImportedExternalMemory(ImportedExternalMemory&& other) noexcept
    : deviceHandle(std::exchange(other.deviceHandle, VK_NULL_HANDLE)),
      freeMemory(std::exchange(other.freeMemory, nullptr)),
      memoryHandle(std::exchange(other.memoryHandle, VK_NULL_HANDLE)),
      typeIndex(std::exchange(other.typeIndex, 0)),
      dedicatedAllocation(std::exchange(other.dedicatedAllocation, false)) {}

ImportedExternalMemory& ImportedExternalMemory::operator=(ImportedExternalMemory&& other) noexcept {
    if (this != &other) {
        reset();
        deviceHandle = std::exchange(other.deviceHandle, VK_NULL_HANDLE);
        freeMemory = std::exchange(other.freeMemory, nullptr);
        memoryHandle = std::exchange(other.memoryHandle, VK_NULL_HANDLE);
        typeIndex = std::exchange(other.typeIndex, 0);
        dedicatedAllocation = std::exchange(other.dedicatedAllocation, false);
    }
    return *this;
}

ImportedExternalMemory::~ImportedExternalMemory() { reset(); }

void ImportedExternalMemory::reset() noexcept {
    if (memoryHandle && freeMemory)
        freeMemory(deviceHandle, memoryHandle, nullptr);
    deviceHandle = VK_NULL_HANDLE;
    freeMemory = nullptr;
    memoryHandle = VK_NULL_HANDLE;
}

ImportedExternalMemory ImportedExternalMemory::import(const ExternalMemoryImportPlan& plan,
        ls::OwnedFd fd, ExternalMemoryImportDiagnostics* diagnostics) {
    if (!plan.device || !plan.AllocateMemory || !plan.FreeMemory
            || !plan.GetMemoryFdPropertiesKHR || !fd)
        throw std::invalid_argument("invalid external memory import plan");
    VkMemoryFdPropertiesKHR fdProperties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    auto result = plan.GetMemoryFdPropertiesKHR(plan.device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd.get(), &fdProperties);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "vkGetMemoryFdPropertiesKHR failed");
    const uint32_t intersection = plan.imageMemoryTypeBits & fdProperties.memoryTypeBits;
    if (!intersection)
        throw ls::vulkan_error("external image memory type intersection is empty");
    std::optional<uint32_t> selected;
    for (uint32_t i = 0; i < plan.memoryProperties.memoryTypeCount; ++i)
        if ((intersection & (1u << i)) != 0) {
            if (!selected)
                selected = i;
            if (plan.prefersDedicated && (plan.memoryProperties.memoryTypes[i].propertyFlags
                    & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) break;
        }
    if (!selected) throw ls::vulkan_error("no external image memory type");
    const bool dedicated = externalMemoryMustBeDedicated(
        plan.externalDedicatedOnly, plan.requiresDedicated);
    std::cerr << "[DG2X-P4B-A4] image allocation request size=" << plan.allocationSize
        << " imageBits=0x" << std::hex << plan.imageMemoryTypeBits
        << " fdBits=0x" << fdProperties.memoryTypeBits
        << " intersection=0x" << intersection << std::dec
        << " dedicated=" << (dedicated ? "YES" : "NO")
        << " fd=" << fd.get() << "\n";
    VkMemoryAllocateInfo plainAllocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    plainAllocate.allocationSize = plan.allocationSize;
    plainAllocate.memoryTypeIndex = *selected;
    VkDeviceMemory plainMemory{};
    const VkResult plainResult = plan.AllocateMemory(
        plan.device, &plainAllocate, nullptr, &plainMemory);
    std::cerr << "[DG2X-P4B-A4H2] plain allocation result="
        << static_cast<int>(plainResult) << " memoryType=" << *selected << "\n";
    if (plainResult == VK_SUCCESS)
        plan.FreeMemory(plan.device, plainMemory, nullptr);
    VkMemoryDedicatedAllocateInfo dedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.image = dedicated ? plan.dedicatedImage : VK_NULL_HANDLE;
    VkImportMemoryFdInfoKHR importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    importInfo.fd = fd.get();
    dedicatedInfo.pNext = dedicated ? &importInfo : nullptr;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = dedicated ? static_cast<const void*>(&dedicatedInfo)
        : static_cast<const void*>(&importInfo);
    allocate.allocationSize = plan.allocationSize;
    allocate.memoryTypeIndex = *selected;
    VkDeviceMemory memory{};
    result = plan.AllocateMemory(plan.device, &allocate, nullptr, &memory);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "vkAllocateMemory external image import failed");
    static_cast<void>(fd.release());
    if (diagnostics) *diagnostics = {*&fdProperties.memoryTypeBits, intersection, *selected, dedicated};
    return ImportedExternalMemory(plan.device, plan.FreeMemory, memory, *selected, dedicated);
}
