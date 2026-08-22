/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../helpers/owned_fd.hpp"
#include <optional>
#include <vulkan/vulkan_core.h>

namespace vk {

[[nodiscard]] constexpr bool externalMemoryMustBeDedicated(
        bool externalDedicatedOnly, bool requiresDedicated) noexcept {
    return externalDedicatedOnly || requiresDedicated;
}

struct ExternalMemoryImportPlan {
    VkDevice device{};
    PFN_vkAllocateMemory AllocateMemory{};
    PFN_vkFreeMemory FreeMemory{};
    PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkDeviceSize allocationSize{};
    uint32_t imageMemoryTypeBits{};
    VkImage dedicatedImage{};
    bool externalDedicatedOnly{};
    bool requiresDedicated{};
    bool prefersDedicated{};
};

struct ExternalMemoryImportDiagnostics {
    uint32_t fdMemoryTypeBits{};
    uint32_t intersectionMemoryTypeBits{};
    uint32_t memoryTypeIndex{};
    bool dedicated{};
};

class ImportedExternalMemory {
public:
    ImportedExternalMemory() noexcept = default;
    ImportedExternalMemory(const ImportedExternalMemory&) = delete;
    ImportedExternalMemory& operator=(const ImportedExternalMemory&) = delete;
    ImportedExternalMemory(ImportedExternalMemory&&) noexcept;
    ImportedExternalMemory& operator=(ImportedExternalMemory&&) noexcept;
    ~ImportedExternalMemory();

    [[nodiscard]] static ImportedExternalMemory import(
        const ExternalMemoryImportPlan&, ls::OwnedFd,
        ExternalMemoryImportDiagnostics* = nullptr);
    [[nodiscard]] VkDeviceMemory memory() const noexcept { return memoryHandle; }
    [[nodiscard]] uint32_t memoryTypeIndex() const noexcept { return typeIndex; }
    [[nodiscard]] bool dedicated() const noexcept { return dedicatedAllocation; }

private:
    friend struct RuntimeImageEndpointTestAccess;
    ImportedExternalMemory(VkDevice, PFN_vkFreeMemory, VkDeviceMemory,
        uint32_t, bool) noexcept;
    void reset() noexcept;
    VkDevice deviceHandle{};
    PFN_vkFreeMemory freeMemory{};
    VkDeviceMemory memoryHandle{};
    uint32_t typeIndex{};
    bool dedicatedAllocation{};
};
}
