/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cstdint>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    /// create a fence
    ls::owned_ptr<VkFence> createFence(const vk::Vulkan& vk,
            Fence::ExternalHandle externalHandle) {
        VkFence handle{};

        const VkExportFenceCreateInfo exportInfo{
            .sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
        };
        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = externalHandle == Fence::ExternalHandle::SyncFd
                ? &exportInfo : nullptr
        };
        auto res = vk.df().CreateFence(vk.dev(), &fenceInfo, VK_NULL_HANDLE, &handle);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res, "vkCreateFence() failed");

        return ls::owned_ptr<VkFence>(
            new VkFence(handle),
            [dev = vk.dev(), defunc = vk.df().DestroyFence](VkFence& fence) {
                defunc(dev, fence, VK_NULL_HANDLE);
            }
        );
    }
}

Fence::Fence(const vk::Vulkan& vk, ExternalHandle externalHandle)
    : fence(createFence(vk, externalHandle)) {}

int Fence::exportSyncFd(const vk::Vulkan& vk) const {
    if (!vk.df().GetFenceFdKHR)
        throw ls::vulkan_error(VK_ERROR_EXTENSION_NOT_PRESENT,
            "vkGetFenceFdKHR() is unavailable");
    const VkFenceGetFdInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
        .fence = this->handle(),
        .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
    };
    int fd{-1};
    const auto result = vk.df().GetFenceFdKHR(vk.dev(), &info, &fd);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "vkGetFenceFdKHR() failed");
    return fd;
}

void Fence::reset(const vk::Vulkan& vk) const {
    auto res = vk.df().ResetFences(vk.dev(), 1, &*this->fence);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkResetFences() failed");
}

bool Fence::wait(const vk::Vulkan& vk, uint64_t timeout) const {
    auto res = vk.df().WaitForFences(vk.dev(), 1, &*this->fence, VK_TRUE, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
        throw ls::vulkan_error(res, "vkWaitForFences() failed");

    return res == VK_SUCCESS;
}
