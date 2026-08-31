/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

/// Immutable, layer-owned projection of a VkPresentInfoKHR pNext chain for one
/// logical swapchain of an application batch. The application chain is never
/// mutated. Unsupported batched structures are rejected before any queue work.
class PresentPNextProjection final {
public:
    PresentPNextProjection() = default;
    PresentPNextProjection(const PresentPNextProjection&) = delete;
    PresentPNextProjection& operator=(const PresentPNextProjection&) = delete;
    PresentPNextProjection(PresentPNextProjection&&) noexcept = default;
    PresentPNextProjection& operator=(PresentPNextProjection&&) noexcept = default;

    [[nodiscard]] static PresentPNextProjection build(
        const void* pNext, uint32_t originalSwapchainCount, uint32_t index) noexcept;

    [[nodiscard]] bool supported() const noexcept { return supportedValue; }
    [[nodiscard]] void* head() const noexcept { return headValue; }
    [[nodiscard]] VkStructureType unsupportedSType() const noexcept {
        return unsupportedType;
    }

private:
    template <typename T>
    T* store(T value) {
        auto storage = std::make_shared<T>(std::move(value));
        auto* result = storage.get();
        allocations.emplace_back(std::move(storage));
        return result;
    }

    void append(VkBaseOutStructure* node) noexcept;

    std::vector<std::shared_ptr<void>> allocations;
    VkBaseOutStructure* tail{};
    void* headValue{};
    bool supportedValue{true};
    VkStructureType unsupportedType{VK_STRUCTURE_TYPE_MAX_ENUM};
};

/// Layer-owned projection of a complete N-swapchain pNext chain. Structure
/// nodes and any data patched by the layer are owned here. Immutable nested
/// application arrays may be referenced only through the synchronous final
/// QueuePresentKHR call and are never mutated or retained asynchronously.
class PresentBatchPNextProjection final {
public:
    enum class Path : uint8_t { Native, D2, VirtualPrepared };

    PresentBatchPNextProjection() = default;
    PresentBatchPNextProjection(const PresentBatchPNextProjection&) = delete;
    PresentBatchPNextProjection& operator=(const PresentBatchPNextProjection&) = delete;
    PresentBatchPNextProjection(PresentBatchPNextProjection&&) noexcept = default;
    PresentBatchPNextProjection& operator=(PresentBatchPNextProjection&&) noexcept = default;

    [[nodiscard]] static PresentBatchPNextProjection build(
        const void* pNext, uint32_t swapchainCount,
        const std::vector<Path>& paths) noexcept;
    [[nodiscard]] bool supported() const noexcept { return supportedValue; }
    [[nodiscard]] void* head() const noexcept { return headValue; }
    [[nodiscard]] VkStructureType unsupportedSType() const noexcept {
        return unsupportedType;
    }
    [[nodiscard]] VkFence applicationFence(uint32_t index) const noexcept;
    [[nodiscard]] bool patchInternalPresentFence(
        uint32_t index, VkFence fence) noexcept;

private:
    template <typename T>
    T* store(T value) {
        auto storage = std::make_shared<T>(std::move(value));
        auto* result = storage.get();
        allocations.emplace_back(std::move(storage));
        return result;
    }
    template <typename T>
    T* storeArray(const T* values, uint32_t count) {
        auto storage = std::make_shared<std::vector<T>>();
        if (count) storage->assign(values, values + count);
        auto* result = storage->data();
        allocations.emplace_back(std::move(storage));
        return result;
    }
    void append(VkBaseOutStructure*) noexcept;
    void reject(VkStructureType) noexcept;

    std::vector<std::shared_ptr<void>> allocations;
    std::vector<Path> pathsValue;
    std::vector<VkFence> applicationFences;
    std::vector<VkFence> finalFences;
    VkSwapchainPresentFenceInfoKHR* fenceNode{};
    VkBaseOutStructure* tail{};
    void* headValue{};
    bool supportedValue{true};
    VkStructureType unsupportedType{VK_STRUCTURE_TYPE_MAX_ENUM};
};

/// Vulkan-defined aggregate result precedence for a decomposed
/// vkQueuePresentKHR batch. Unknown fatal errors are preserved conservatively.
[[nodiscard]] VkResult aggregatePresentResults(
    const std::vector<VkResult>& results) noexcept;

} // namespace lsfgvk::layer
