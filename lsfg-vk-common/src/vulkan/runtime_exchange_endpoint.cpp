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
    const auto getImageModifier = reinterpret_cast<PFN_vkGetImageDrmFormatModifierPropertiesEXT>(
        vk.fi().GetDeviceProcAddr(vk.dev(), "vkGetImageDrmFormatModifierPropertiesEXT"));
    const auto getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vk.fi().GetDeviceProcAddr(vk.dev(), "vkGetMemoryFdKHR"));
    const auto copyImage = reinterpret_cast<PFN_vkCmdCopyImage>(
        vk.fi().GetDeviceProcAddr(vk.dev(), "vkCmdCopyImage"));
    const auto getSubresourceLayout = reinterpret_cast<PFN_vkGetImageSubresourceLayout>(
        vk.fi().GetDeviceProcAddr(vk.dev(), "vkGetImageSubresourceLayout"));
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
        .ResetFences = vk.df().ResetFences,
        .DeviceWaitIdle = vk.df().DeviceWaitIdle,
        .CreateCommandPool = vk.df().CreateCommandPool,
        .DestroyCommandPool = vk.df().DestroyCommandPool,
        .AllocateCommandBuffers = vk.df().AllocateCommandBuffers,
        .FreeCommandBuffers = vk.df().FreeCommandBuffers,
        .BeginCommandBuffer = vk.df().BeginCommandBuffer,
        .EndCommandBuffer = vk.df().EndCommandBuffer,
        .CmdPipelineBarrier = vk.df().CmdPipelineBarrier,
        .CmdFillBuffer = vk.df().CmdFillBuffer,
        .CmdCopyBuffer = vk.df().CmdCopyBuffer,
        .MapMemory = vk.df().MapMemory,
        .UnmapMemory = vk.df().UnmapMemory,
        .InvalidateMappedMemoryRanges = vk.df().InvalidateMappedMemoryRanges
        ,.physicalDevice = vk.physdev()
        ,.identity = getPhysicalDeviceIdentity(vk.fi(), vk.physdev())
        ,.GetPhysicalDeviceFormatProperties2 = vk.fi().GetPhysicalDeviceFormatProperties2
        ,.GetPhysicalDeviceImageFormatProperties2 = vk.fi().GetPhysicalDeviceImageFormatProperties2
        ,.CreateImage = vk.df().CreateImage
        ,.DestroyImage = vk.df().DestroyImage
        ,.GetImageMemoryRequirements2 = vk.df().GetImageMemoryRequirements2
        ,.BindImageMemory = vk.df().BindImageMemory
        ,.CmdClearColorImage = vk.df().CmdClearColorImage
        ,.CmdCopyImageToBuffer = vk.df().CmdCopyImageToBuffer
        ,.CmdBlitImage = vk.df().CmdBlitImage
        ,.CmdCopyImage = copyImage
        ,.GetImageSubresourceLayout = getSubresourceLayout
        ,.GetImageDrmFormatModifierPropertiesEXT = getImageModifier
        ,.GetMemoryFdKHR = getMemoryFd
        ,.GetMemoryFdPropertiesKHR = getMemoryFdProperties
        ,.AllocateMemory = vk.df().AllocateMemory
        ,.FreeMemory = vk.df().FreeMemory
        ,.CreateBuffer = vk.df().CreateBuffer
        ,.DestroyBuffer = vk.df().DestroyBuffer
        ,.GetBufferMemoryRequirements = vk.df().GetBufferMemoryRequirements
        ,.BindBufferMemory = vk.df().BindBufferMemory
    };
}
