#include "terminal_pending_gpu_operation.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"

namespace lsfgvk::layer {
namespace {
ls::owned_ptr<VkCommandPool> createCommandPool(
        const vk::Vulkan& vk, uint32_t family) {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = family;
    VkCommandPool pool{};
    const auto result = vk.df().CreateCommandPool(vk.dev(), &info, nullptr, &pool);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result,
            "terminal per-operation command pool creation failed");
    return ls::owned_ptr<VkCommandPool>(new VkCommandPool(pool),
        [device = vk.dev(), destroy = vk.df().DestroyCommandPool](VkCommandPool& value) {
            destroy(device, value, nullptr);
        });
}
}

D3B1TerminalPendingGpuBacking::D3B1TerminalPendingGpuBacking(
        const vk::Vulkan& vk, uint32_t queueFamily) :
    commandPool(createCommandPool(vk, queueFamily)),
    commandBuffer(vk, *commandPool),
    terminalFence(vk, vk::Fence::ExternalHandle::SyncFd),
    hiddenAcquire(vk),
    finalPresent(vk),
    presentFence(vk, vk::Fence::ExternalHandle::SyncFd) {}

D3B2TerminalPendingGpuBacking::D3B2TerminalPendingGpuBacking(
        const vk::Vulkan& vk, uint32_t queueFamily) :
    commandPool(createCommandPool(vk, queueFamily)),
    commandBuffer(vk, *commandPool),
    terminalFence(vk, vk::Fence::ExternalHandle::SyncFd),
    generatedAcquire(vk),
    originalAcquire(vk),
    generatedPresent(vk),
    originalPresent(vk),
    generatedPresentFence(vk, vk::Fence::ExternalHandle::SyncFd),
    originalPresentFence(vk, vk::Fence::ExternalHandle::SyncFd) {}

}
