/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <vulkan/vulkan_core.h>

using namespace vk;

RuntimeExchangeEndpoint vk::makeRuntimeExchangeEndpoint(const Vulkan& vk) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vk.fi().GetPhysicalDeviceMemoryProperties(vk.physdev(), &memoryProperties);
    const auto getMemoryFdProperties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
        vk.fi().GetDeviceProcAddr(vk.dev(), "vkGetMemoryFdPropertiesKHR"));
    if (!getMemoryFdProperties)
        throw ls::vulkan_error("vkGetMemoryFdPropertiesKHR unavailable for runtime exchange endpoint");

    return {
        .bufferDevice = {
            .device = vk.dev(),
            .memoryProperties = memoryProperties,
            .funcs = {
                .CreateBuffer = vk.df().CreateBuffer,
                .DestroyBuffer = vk.df().DestroyBuffer,
                .GetBufferMemoryRequirements = vk.df().GetBufferMemoryRequirements,
                .GetMemoryFdPropertiesKHR = getMemoryFdProperties,
                .AllocateMemory = vk.df().AllocateMemory,
                .FreeMemory = vk.df().FreeMemory,
                .BindBufferMemory = vk.df().BindBufferMemory
            }
        },
        .semaphoreDevice = {
            .device = vk.dev(),
            .funcs = {
                .CreateSemaphore = vk.df().CreateSemaphore,
                .DestroySemaphore = vk.df().DestroySemaphore,
                .GetSemaphoreFdKHR = vk.df().GetSemaphoreFdKHR,
                .ImportSemaphoreFdKHR = vk.df().ImportSemaphoreFdKHR
            }
        },
        .queue = vk.queue(),
        .queueFamilyIndex = vk.queueFamilyIndex(),
        .QueueSubmit = vk.df().QueueSubmit,
        .CreateFence = vk.df().CreateFence,
        .DestroyFence = vk.df().DestroyFence,
        .WaitForFences = vk.df().WaitForFences,
        .DeviceWaitIdle = vk.df().DeviceWaitIdle
    };
}
