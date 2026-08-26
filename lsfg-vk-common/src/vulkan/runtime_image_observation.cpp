/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_image_observation.hpp"

#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/external_buffer_transport.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

    [[noreturn]] void throwVk(VkResult result, const char* message) {
        throw ls::vulkan_error(result, message);
    }

    void validateEndpoint(const vk::RuntimeExchangeEndpoint& endpoint) {
        const auto& buffer = endpoint.bufferDevice;
        if (buffer.device == VK_NULL_HANDLE || endpoint.queue == VK_NULL_HANDLE
                || endpoint.queueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
                || !buffer.funcs.CreateBuffer || !buffer.funcs.DestroyBuffer
                || !buffer.funcs.GetBufferMemoryRequirements
                || !buffer.funcs.AllocateMemory || !buffer.funcs.FreeMemory
                || !buffer.funcs.BindBufferMemory
                || !endpoint.MapMemory || !endpoint.UnmapMemory
                || !endpoint.InvalidateMappedMemoryRanges
                || !endpoint.CreateCommandPool || !endpoint.DestroyCommandPool
                || !endpoint.AllocateCommandBuffers || !endpoint.FreeCommandBuffers
                || !endpoint.ResetCommandBuffer || !endpoint.BeginCommandBuffer
                || !endpoint.EndCommandBuffer || !endpoint.CmdPipelineBarrier
                || !endpoint.CmdCopyImageToBuffer || !endpoint.QueueSubmit
                || !endpoint.CreateFence || !endpoint.DestroyFence
                || !endpoint.GetFenceStatus || !endpoint.ResetFences)
            throw std::invalid_argument(
                "incomplete runtime image observation Vulkan endpoint");
    }

}

struct vk::RuntimeImageObservationSession::Impl {
    struct Slot {
        VkCommandBuffer command{VK_NULL_HANDLE};
        VkBuffer stagingBuffer{VK_NULL_HANDLE};
        VkDeviceMemory stagingMemory{VK_NULL_HANDLE};
        void* stagingMapped{};
        bool hostCoherent{};
        VkFence fence{VK_NULL_HANDLE};
        bool inFlight{};
        uint64_t epoch{};
        RuntimeImageObservationDescriptor descriptor{};
        size_t byteCount{};
        std::shared_ptr<const uint8_t> imageLifetime;
    };

    RuntimeExchangeEndpoint endpoint;
    RuntimeImageObservationSessionInfo info;
    VkCommandPool commandPool{VK_NULL_HANDLE};
    bool commandsAllocated{};
    std::vector<VkCommandBuffer> commands;
    std::vector<Slot> slots;

    ~Impl() {
        const auto device = endpoint.bufferDevice.device;
        if (commandsAllocated && commandPool != VK_NULL_HANDLE && !commands.empty())
            endpoint.FreeCommandBuffers(device, commandPool,
                static_cast<uint32_t>(commands.size()), commands.data());
        if (commandPool != VK_NULL_HANDLE)
            endpoint.DestroyCommandPool(device, commandPool, nullptr);

        for (auto& slot : slots) {
            if (slot.stagingMapped && slot.stagingMemory != VK_NULL_HANDLE)
                endpoint.UnmapMemory(device, slot.stagingMemory);
            if (slot.stagingBuffer != VK_NULL_HANDLE)
                endpoint.bufferDevice.funcs.DestroyBuffer(
                    device, slot.stagingBuffer, nullptr);
            if (slot.stagingMemory != VK_NULL_HANDLE)
                endpoint.bufferDevice.funcs.FreeMemory(
                    device, slot.stagingMemory, nullptr);
            if (slot.fence != VK_NULL_HANDLE)
                endpoint.DestroyFence(device, slot.fence, nullptr);
        }
    }

    [[nodiscard]] uint32_t inFlightCount() const noexcept {
        return static_cast<uint32_t>(std::count_if(
            slots.begin(), slots.end(), [](const Slot& slot) {
                return slot.inFlight;
            }));
    }
};

std::optional<VkDeviceSize> vk::runtimeImageObservationByteCount(
        VkFormat format, VkExtent2D extent) noexcept {
    if (format != VK_FORMAT_R8G8B8A8_UNORM
            && format != VK_FORMAT_B8G8R8A8_UNORM)
        return std::nullopt;
    if (extent.width == 0 || extent.height == 0)
        return std::nullopt;

    constexpr VkDeviceSize bytesPerTexel = 4;
    constexpr auto max = std::numeric_limits<VkDeviceSize>::max();
    const VkDeviceSize width = extent.width;
    const VkDeviceSize height = extent.height;
    if (width > max / bytesPerTexel
            || height > max / (width * bytesPerTexel))
        return std::nullopt;
    return width * height * bytesPerTexel;
}

vk::RuntimeImageObservationSession::RuntimeImageObservationSession(
        std::unique_ptr<Impl> value) noexcept : impl(std::move(value)) {}

vk::RuntimeImageObservationSession::RuntimeImageObservationSession(
        RuntimeImageObservationSession&& other) noexcept :
    impl(std::move(other.impl)) {}

vk::RuntimeImageObservationSession&
vk::RuntimeImageObservationSession::operator=(
        RuntimeImageObservationSession&& other) noexcept {
    if (this != &other) {
        reset();
        impl = std::move(other.impl);
    }
    return *this;
}

vk::RuntimeImageObservationSession::~RuntimeImageObservationSession() {
    reset();
}

void vk::RuntimeImageObservationSession::reset() noexcept {
    if (impl && impl->inFlightCount() != 0) {
        // No host wait is permitted here.  Retaining the bounded block is the
        // only safe fallback when the caller violates the retirement contract:
        // command, fence, staging, and source lifetime must all remain valid.
        static_cast<void>(impl.release());
        return;
    }
    impl.reset();
}

bool vk::RuntimeImageObservationSession::valid() const noexcept {
    return impl != nullptr;
}

uint32_t vk::RuntimeImageObservationSession::slotCount() const noexcept {
    return impl ? static_cast<uint32_t>(impl->slots.size()) : 0;
}

uint32_t vk::RuntimeImageObservationSession::inFlightCount() const noexcept {
    return impl ? impl->inFlightCount() : 0;
}

VkDeviceSize vk::RuntimeImageObservationSession::maxByteCount() const noexcept {
    return impl ? impl->info.maxByteCount : 0;
}

vk::RuntimeImageObservationSubmitResult
vk::RuntimeImageObservationSession::submit(
        const RuntimeImageObservationDescriptor& descriptor) {
    if (!impl)
        throw std::logic_error("invalid runtime image observation session");

    const auto byteCount = runtimeImageObservationByteCount(
        descriptor.format, descriptor.extent);
    if (descriptor.image == VK_NULL_HANDLE)
        throw std::invalid_argument("observation image is null");
    if (descriptor.layout != VK_IMAGE_LAYOUT_GENERAL
            && descriptor.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        throw std::invalid_argument("unsupported observation image layout");
    if (!byteCount)
        throw std::invalid_argument("unsupported observation format or extent");
    if (*byteCount > impl->info.maxByteCount
            || *byteCount > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("observation image exceeds staging capacity");
    if (descriptor.queueFamilyIndex != impl->endpoint.queueFamilyIndex)
        throw std::invalid_argument("observation image queue family mismatch");
    auto lifetime = descriptor.lifetime.lock();
    if (!lifetime)
        throw std::invalid_argument("observation image lifetime expired");

    const auto freeSlot = std::find_if(impl->slots.begin(), impl->slots.end(),
        [](const Impl::Slot& slot) { return !slot.inFlight; });
    if (freeSlot == impl->slots.end())
        return {
            .status = RuntimeImageObservationSubmitStatus::TEMPORARILY_BLOCKED,
            .ticket = {}
        };

    auto& slot = *freeSlot;
    if (slot.epoch == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("runtime image observation ticket epoch exhausted");

    const auto device = impl->endpoint.bufferDevice.device;
    VkResult result = impl->endpoint.ResetFences(device, 1, &slot.fence);
    if (result != VK_SUCCESS)
        throwVk(result, "vkResetFences() failed for runtime image observation");
    result = impl->endpoint.ResetCommandBuffer(slot.command, 0);
    if (result != VK_SUCCESS)
        throwVk(result, "vkResetCommandBuffer() failed for runtime image observation");

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    result = impl->endpoint.BeginCommandBuffer(slot.command, &beginInfo);
    if (result != VK_SUCCESS)
        throwVk(result, "vkBeginCommandBuffer() failed for runtime image observation");

    const VkImageMemoryBarrier beforeCopy{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = descriptor.layout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = descriptor.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    impl->endpoint.CmdPipelineBarrier(slot.command,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &beforeCopy);

    const VkBufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {
            .width = descriptor.extent.width,
            .height = descriptor.extent.height,
            .depth = 1
        }
    };
    impl->endpoint.CmdCopyImageToBuffer(slot.command, descriptor.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.stagingBuffer,
        1, &copyRegion);

    const bool restoreGeneral = descriptor.layout == VK_IMAGE_LAYOUT_GENERAL;
    const VkAccessFlags restoredAccess = restoreGeneral
        ? static_cast<VkAccessFlags>(
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)
        : VK_ACCESS_TRANSFER_READ_BIT;
    const VkImageMemoryBarrier afterCopy{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = restoredAccess,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = descriptor.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = descriptor.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    impl->endpoint.CmdPipelineBarrier(slot.command,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        restoreGeneral ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                       : VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr, 0, nullptr, 1, &afterCopy);

    result = impl->endpoint.EndCommandBuffer(slot.command);
    if (result != VK_SUCCESS)
        throwVk(result, "vkEndCommandBuffer() failed for runtime image observation");

    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &slot.command
    };
    result = impl->endpoint.QueueSubmit(
        impl->endpoint.queue, 1, &submitInfo, slot.fence);
    if (result != VK_SUCCESS)
        throwVk(result, "vkQueueSubmit() failed for runtime image observation");

    ++slot.epoch;
    slot.inFlight = true;
    slot.descriptor = descriptor;
    slot.byteCount = static_cast<size_t>(*byteCount);
    slot.imageLifetime = std::move(lifetime);

    return {
        .status = RuntimeImageObservationSubmitStatus::SUBMITTED,
        .ticket = {
            .slotIndex = static_cast<uint32_t>(freeSlot - impl->slots.begin()),
            .epoch = slot.epoch
        }
    };
}

vk::RuntimeImageObservationRetireResult
vk::RuntimeImageObservationSession::tryRetire(
        RuntimeImageObservationTicket ticket) {
    if (!impl)
        throw std::logic_error("invalid runtime image observation session");
    if (!ticket.valid() || ticket.slotIndex >= impl->slots.size())
        throw std::invalid_argument("invalid runtime image observation ticket");
    auto& slot = impl->slots[ticket.slotIndex];
    if (!slot.inFlight || slot.epoch != ticket.epoch)
        throw std::invalid_argument("stale runtime image observation ticket");

    const auto device = impl->endpoint.bufferDevice.device;
    VkResult result = impl->endpoint.GetFenceStatus(device, slot.fence);
    if (result == VK_NOT_READY)
        return {
            .status = RuntimeImageObservationRetireStatus::NOT_READY,
            .observation = std::nullopt
        };
    if (result != VK_SUCCESS)
        throwVk(result, "vkGetFenceStatus() failed for runtime image observation");

    if (!slot.hostCoherent) {
        const VkMappedMemoryRange range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = slot.stagingMemory,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };
        result = impl->endpoint.InvalidateMappedMemoryRanges(device, 1, &range);
        if (result != VK_SUCCESS)
            throwVk(result,
                "vkInvalidateMappedMemoryRanges() failed for runtime image observation");
    }

    const auto* bytes = static_cast<const uint8_t*>(slot.stagingMapped);
    RuntimeImageObservation observation{
        .image = slot.descriptor.image,
        .layout = slot.descriptor.layout,
        .format = slot.descriptor.format,
        .extent = slot.descriptor.extent,
        .queueFamilyIndex = slot.descriptor.queueFamilyIndex,
        .byteCount = slot.byteCount,
        .nonzeroByteCount = static_cast<size_t>(std::count_if(
            bytes, bytes + slot.byteCount,
            [](uint8_t byte) { return byte != 0; })),
        .checksum = lsfgvk::common::fnv1a64(bytes, slot.byteCount)
    };

    slot.inFlight = false;
    slot.descriptor = {};
    slot.byteCount = 0;
    slot.imageLifetime.reset();

    return {
        .status = RuntimeImageObservationRetireStatus::RETIRED,
        .observation = observation
    };
}

vk::RuntimeImageObservationSession vk::createRuntimeImageObservationSession(
        RuntimeExchangeEndpoint endpoint, RuntimeImageObservationSessionInfo info) {
    validateEndpoint(endpoint);
    if (info.slotCount == 0)
        throw std::invalid_argument("runtime image observation requires at least one slot");
    if (info.maxByteCount == 0
            || info.maxByteCount > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("invalid runtime image observation staging capacity");

    auto impl = std::make_unique<RuntimeImageObservationSession::Impl>();
    impl->endpoint = std::move(endpoint);
    impl->info = info;
    impl->slots.resize(info.slotCount);

    const auto device = impl->endpoint.bufferDevice.device;
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = impl->endpoint.queueFamilyIndex
    };
    VkResult result = impl->endpoint.CreateCommandPool(
        device, &poolInfo, nullptr, &impl->commandPool);
    if (result != VK_SUCCESS)
        throwVk(result, "vkCreateCommandPool() failed for runtime image observation");

    impl->commands.resize(info.slotCount, VK_NULL_HANDLE);
    const VkCommandBufferAllocateInfo commandInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = impl->commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = info.slotCount
    };
    result = impl->endpoint.AllocateCommandBuffers(
        device, &commandInfo, impl->commands.data());
    if (result != VK_SUCCESS) {
        impl->commands.clear();
        throwVk(result,
            "vkAllocateCommandBuffers() failed for runtime image observation");
    }
    impl->commandsAllocated = true;

    for (uint32_t i = 0; i < info.slotCount; ++i) {
        auto& slot = impl->slots[i];
        slot.command = impl->commands[i];

        const VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = info.maxByteCount,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        result = impl->endpoint.bufferDevice.funcs.CreateBuffer(
            device, &bufferInfo, nullptr, &slot.stagingBuffer);
        if (result != VK_SUCCESS)
            throwVk(result,
                "vkCreateBuffer() failed for runtime image observation staging");

        VkMemoryRequirements requirements{};
        impl->endpoint.bufferDevice.funcs.GetBufferMemoryRequirements(
            device, slot.stagingBuffer, &requirements);
        const auto memoryType = selectExternalBufferMemoryType(
            requirements.memoryTypeBits,
            impl->endpoint.bufferDevice.memoryProperties,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!memoryType)
            throw std::runtime_error(
                "no host-visible memory for runtime image observation staging");

        const auto memoryFlags = impl->endpoint.bufferDevice.memoryProperties
            .memoryTypes[*memoryType].propertyFlags;
        slot.hostCoherent =
            (memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        const VkMemoryAllocateInfo memoryInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *memoryType
        };
        result = impl->endpoint.bufferDevice.funcs.AllocateMemory(
            device, &memoryInfo, nullptr, &slot.stagingMemory);
        if (result != VK_SUCCESS)
            throwVk(result,
                "vkAllocateMemory() failed for runtime image observation staging");
        result = impl->endpoint.bufferDevice.funcs.BindBufferMemory(
            device, slot.stagingBuffer, slot.stagingMemory, 0);
        if (result != VK_SUCCESS)
            throwVk(result,
                "vkBindBufferMemory() failed for runtime image observation staging");
        result = impl->endpoint.MapMemory(device, slot.stagingMemory,
            0, VK_WHOLE_SIZE, 0, &slot.stagingMapped);
        if (result != VK_SUCCESS || !slot.stagingMapped)
            throwVk(result == VK_SUCCESS ? VK_ERROR_MEMORY_MAP_FAILED : result,
                "vkMapMemory() failed for runtime image observation staging");

        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        result = impl->endpoint.CreateFence(
            device, &fenceInfo, nullptr, &slot.fence);
        if (result != VK_SUCCESS)
            throwVk(result,
                "vkCreateFence() failed for runtime image observation");
    }

    return RuntimeImageObservationSession(std::move(impl));
}
