#pragma once

#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"

#include <cstdint>

namespace lsfgvk::layer {

class D3B1TerminalPendingGpuBacking final {
public:
    D3B1TerminalPendingGpuBacking(const vk::Vulkan&, uint32_t queueFamily);

    D3B1TerminalPendingGpuBacking(const D3B1TerminalPendingGpuBacking&) = delete;
    D3B1TerminalPendingGpuBacking& operator=(
        const D3B1TerminalPendingGpuBacking&) = delete;

    [[nodiscard]] const vk::CommandBuffer& command() const noexcept {
        return commandBuffer;
    }
    [[nodiscard]] VkFence completionFence() const noexcept {
        return terminalFence.handle();
    }
    [[nodiscard]] VkSemaphore acquireSemaphore() const noexcept {
        return hiddenAcquire.handle();
    }
    [[nodiscard]] VkSemaphore presentSemaphore() const noexcept {
        return finalPresent.handle();
    }
    [[nodiscard]] VkFence presentationFence() const noexcept {
        return presentFence.handle();
    }
    [[nodiscard]] int exportCompletionFd(const vk::Vulkan& vk) const {
        return terminalFence.exportSyncFd(vk);
    }
    [[nodiscard]] int exportPresentationFd(const vk::Vulkan& vk) const {
        return presentFence.exportSyncFd(vk);
    }

private:
    ls::owned_ptr<VkCommandPool> commandPool;
    vk::CommandBuffer commandBuffer;
    vk::Fence terminalFence;
    vk::Semaphore hiddenAcquire;
    vk::Semaphore finalPresent;
    vk::Fence presentFence;
};

class D3B2TerminalPendingGpuBacking final {
public:
    D3B2TerminalPendingGpuBacking(const vk::Vulkan&, uint32_t queueFamily);

    D3B2TerminalPendingGpuBacking(const D3B2TerminalPendingGpuBacking&) = delete;
    D3B2TerminalPendingGpuBacking& operator=(
        const D3B2TerminalPendingGpuBacking&) = delete;

    [[nodiscard]] const vk::CommandBuffer& command() const noexcept {
        return commandBuffer;
    }
    [[nodiscard]] VkFence completionFence() const noexcept {
        return terminalFence.handle();
    }
    [[nodiscard]] VkSemaphore generatedAcquireSemaphore() const noexcept {
        return generatedAcquire.handle();
    }
    [[nodiscard]] VkSemaphore originalAcquireSemaphore() const noexcept {
        return originalAcquire.handle();
    }
    [[nodiscard]] VkSemaphore generatedPresentSemaphore() const noexcept {
        return generatedPresent.handle();
    }
    [[nodiscard]] VkSemaphore originalPresentSemaphore() const noexcept {
        return originalPresent.handle();
    }
    [[nodiscard]] VkFence generatedPresentationFence() const noexcept {
        return generatedPresentFence.handle();
    }
    [[nodiscard]] VkFence originalPresentationFence() const noexcept {
        return originalPresentFence.handle();
    }
    [[nodiscard]] int exportCompletionFd(const vk::Vulkan& vk) const {
        return terminalFence.exportSyncFd(vk);
    }
    [[nodiscard]] int exportGeneratedPresentationFd(const vk::Vulkan& vk) const {
        return generatedPresentFence.exportSyncFd(vk);
    }
    [[nodiscard]] int exportOriginalPresentationFd(const vk::Vulkan& vk) const {
        return originalPresentFence.exportSyncFd(vk);
    }

private:
    ls::owned_ptr<VkCommandPool> commandPool;
    vk::CommandBuffer commandBuffer;
    vk::Fence terminalFence;
    vk::Semaphore generatedAcquire;
    vk::Semaphore originalAcquire;
    vk::Semaphore generatedPresent;
    vk::Semaphore originalPresent;
    vk::Fence generatedPresentFence;
    vk::Fence originalPresentFence;
};

}
