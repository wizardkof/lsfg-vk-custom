/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../helpers/owned_fd.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <vulkan/vulkan_core.h>

namespace vk {

    // Minimal Vulkan device dispatch needed to import one DMA-BUF as a VkBuffer.
    // It is intentionally independent from the higher-level Vulkan runtime so
    // probes and the future dual-GPU path can share the same import contract.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    struct ExternalBufferDeviceFuncs {
        PFN_vkCreateBuffer CreateBuffer{};
        PFN_vkDestroyBuffer DestroyBuffer{};
        PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{};
        PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR{};
        PFN_vkAllocateMemory AllocateMemory{};
        PFN_vkFreeMemory FreeMemory{};
        PFN_vkBindBufferMemory BindBufferMemory{};
    };

    struct ExternalBufferDevice {
        VkDevice device{};
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        ExternalBufferDeviceFuncs funcs{};
    };

    struct ExternalBufferImportInfo {
        VkDeviceSize logicalSize{};
        VkDeviceSize backingSize{};
        VkBufferUsageFlags usage{};
        VkMemoryPropertyFlags requiredMemoryProperties{};
        VkMemoryPropertyFlags preferredMemoryProperties{};
    };

    struct ExternalBufferImportDiagnostics {
        VkMemoryRequirements requirements{};
        uint32_t fdMemoryTypeBits{};
        uint32_t usableMemoryTypeBits{};
        uint32_t memoryTypeIndex{};
        VkMemoryPropertyFlags memoryPropertyFlags{};
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    enum class ExternalBufferImportFailure {
        InvalidFileDescriptor,
        BufferCreate,
        FdProperties,
        BackingTooSmall,
        NoMemoryTypeIntersection,
        MemoryImport,
        BufferBind
    };

    class ExternalBufferImportError final : public std::runtime_error {
    public:
        ExternalBufferImportError(ExternalBufferImportFailure failure,
            VkResult result, const char* message);

        [[nodiscard]] ExternalBufferImportFailure failure() const noexcept {
            return this->failureCode;
        }
        [[nodiscard]] VkResult result() const noexcept { return this->vkResult; }

    private:
        ExternalBufferImportFailure failureCode;
        VkResult vkResult;
    };

    class ImportedExternalBuffer {
    public:
        ImportedExternalBuffer() noexcept = default;
        ImportedExternalBuffer(const ImportedExternalBuffer&) = delete;
        ImportedExternalBuffer& operator=(const ImportedExternalBuffer&) = delete;
        ImportedExternalBuffer(ImportedExternalBuffer&& other) noexcept;
        ImportedExternalBuffer& operator=(ImportedExternalBuffer&& other) noexcept;
        ~ImportedExternalBuffer();

        [[nodiscard]] VkBuffer buffer() const noexcept { return this->bufferHandle; }
        [[nodiscard]] VkDeviceMemory memory() const noexcept { return this->memoryHandle; }
        [[nodiscard]] const ExternalBufferImportDiagnostics& diagnostics() const noexcept {
            return this->importDiagnostics;
        }
        explicit operator bool() const noexcept {
            return this->bufferHandle != VK_NULL_HANDLE
                && this->memoryHandle != VK_NULL_HANDLE;
        }

    private:
        friend ImportedExternalBuffer importDmaBufBuffer(
            const ExternalBufferDevice&, ls::OwnedFd, const ExternalBufferImportInfo&);

        ImportedExternalBuffer(VkDevice device,
            PFN_vkDestroyBuffer destroyBuffer,
            PFN_vkFreeMemory freeMemory,
            VkBuffer buffer,
            VkDeviceMemory memory,
            ExternalBufferImportDiagnostics diagnostics) noexcept;

        void reset() noexcept;

        VkDevice deviceHandle{};
        PFN_vkDestroyBuffer destroyBuffer{};
        PFN_vkFreeMemory freeMemory{};
        VkBuffer bufferHandle{};
        VkDeviceMemory memoryHandle{};
        ExternalBufferImportDiagnostics importDiagnostics{};
    };

    [[nodiscard]] uint32_t intersectExternalBufferMemoryTypes(
        uint32_t fdMemoryTypeBits, uint32_t bufferMemoryTypeBits) noexcept;

    [[nodiscard]] std::optional<uint32_t> selectExternalBufferMemoryType(
        uint32_t memoryTypeBits,
        const VkPhysicalDeviceMemoryProperties& memoryProperties,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0) noexcept;

    /// Import one owned DMA-BUF fd into one Vulkan device as a VkBuffer.
    /// The fd is consumed by Vulkan only after a successful vkAllocateMemory;
    /// all failure paths retain normal OwnedFd cleanup semantics.
    [[nodiscard]] ImportedExternalBuffer importDmaBufBuffer(
        const ExternalBufferDevice& device,
        ls::OwnedFd fd,
        const ExternalBufferImportInfo& info);

}
