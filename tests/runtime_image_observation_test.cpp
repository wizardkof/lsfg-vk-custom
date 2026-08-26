/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/vulkan/runtime_image_observation.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

    template<typename T>
    T handle(uintptr_t value) {
        if constexpr (std::is_pointer_v<T>)
            return reinterpret_cast<T>(value);
        else
            return static_cast<T>(value);
    }

    constexpr VkDeviceSize CAPACITY = 64;
    const VkDevice DEVICE = handle<VkDevice>(0x1000);
    const VkQueue QUEUE = handle<VkQueue>(0x2000);

    struct BufferState {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
    };

    struct MemoryState {
        VkDeviceMemory memory{VK_NULL_HANDLE};
        std::vector<uint8_t> bytes;
    };

    struct FenceState {
        VkFence fence{VK_NULL_HANDLE};
        bool signaled{};
    };

    struct CommandState {
        VkCommandBuffer command{VK_NULL_HANDLE};
        VkImage image{VK_NULL_HANDLE};
        VkBuffer buffer{VK_NULL_HANDLE};
    };

    struct BarrierRecord {
        VkPipelineStageFlags srcStage{};
        VkPipelineStageFlags dstStage{};
        VkImageMemoryBarrier barrier{};
    };

    struct Trace {
        std::vector<BufferState> buffers;
        std::vector<MemoryState> memories;
        std::vector<FenceState> fences;
        std::vector<CommandState> commands;
        std::vector<BarrierRecord> barriers;
        std::vector<std::string> events;
        uint32_t commandPoolCreates{};
        uint32_t commandPoolDestroys{};
        uint32_t commandAllocations{};
        uint32_t commandFrees{};
        uint32_t bufferDestroys{};
        uint32_t memoryFrees{};
        uint32_t maps{};
        uint32_t unmaps{};
        uint32_t fenceDestroys{};
        uint32_t fenceResets{};
        uint32_t commandResets{};
        uint32_t begins{};
        uint32_t ends{};
        uint32_t copies{};
        uint32_t submits{};
        uint32_t fenceStatusChecks{};
        uint32_t invalidates{};
        uint32_t waitForFences{};
        uint32_t queueWaitIdle{};
        uint32_t deviceWaitIdle{};
    } trace;

    BufferState& bufferState(VkBuffer buffer) {
        const auto found = std::find_if(trace.buffers.begin(), trace.buffers.end(),
            [buffer](const BufferState& state) { return state.buffer == buffer; });
        assert(found != trace.buffers.end());
        return *found;
    }

    MemoryState& memoryState(VkDeviceMemory memory) {
        const auto found = std::find_if(trace.memories.begin(), trace.memories.end(),
            [memory](const MemoryState& state) { return state.memory == memory; });
        assert(found != trace.memories.end());
        return *found;
    }

    FenceState& fenceState(VkFence fence) {
        const auto found = std::find_if(trace.fences.begin(), trace.fences.end(),
            [fence](const FenceState& state) { return state.fence == fence; });
        assert(found != trace.fences.end());
        return *found;
    }

    CommandState& commandState(VkCommandBuffer command) {
        const auto found = std::find_if(trace.commands.begin(), trace.commands.end(),
            [command](const CommandState& state) { return state.command == command; });
        assert(found != trace.commands.end());
        return *found;
    }

    VKAPI_ATTR VkResult VKAPI_CALL createBuffer(VkDevice device,
            const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* buffer) {
        assert(device == DEVICE && info && buffer);
        assert(info->size == CAPACITY);
        assert(info->usage == VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        assert(info->sharingMode == VK_SHARING_MODE_EXCLUSIVE);
        *buffer = handle<VkBuffer>(0x3000 + trace.buffers.size());
        trace.buffers.push_back({*buffer, VK_NULL_HANDLE});
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL destroyBuffer(VkDevice device, VkBuffer buffer,
            const VkAllocationCallbacks*) {
        assert(device == DEVICE && buffer != VK_NULL_HANDLE);
        ++trace.bufferDestroys;
    }

    VKAPI_ATTR void VKAPI_CALL getBufferMemoryRequirements(VkDevice device,
            VkBuffer buffer, VkMemoryRequirements* requirements) {
        assert(device == DEVICE && buffer != VK_NULL_HANDLE && requirements);
        *requirements = {
            .size = CAPACITY,
            .alignment = 8,
            .memoryTypeBits = 1
        };
    }

    VKAPI_ATTR VkResult VKAPI_CALL allocateMemory(VkDevice device,
            const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
            VkDeviceMemory* memory) {
        assert(device == DEVICE && info && memory);
        assert(info->allocationSize == CAPACITY && info->memoryTypeIndex == 0);
        *memory = handle<VkDeviceMemory>(0x4000 + trace.memories.size());
        trace.memories.push_back({*memory, std::vector<uint8_t>(CAPACITY)});
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL freeMemory(VkDevice device, VkDeviceMemory memory,
            const VkAllocationCallbacks*) {
        assert(device == DEVICE && memory != VK_NULL_HANDLE);
        ++trace.memoryFrees;
    }

    VKAPI_ATTR VkResult VKAPI_CALL bindBufferMemory(VkDevice device,
            VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
        assert(device == DEVICE && offset == 0);
        bufferState(buffer).memory = memory;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mapMemory(VkDevice device,
            VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
            VkMemoryMapFlags, void** mapped) {
        assert(device == DEVICE && offset == 0 && size == VK_WHOLE_SIZE && mapped);
        ++trace.maps;
        *mapped = memoryState(memory).bytes.data();
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL unmapMemory(VkDevice device, VkDeviceMemory memory) {
        assert(device == DEVICE && memory != VK_NULL_HANDLE);
        ++trace.unmaps;
    }

    VKAPI_ATTR VkResult VKAPI_CALL invalidateMappedMemoryRanges(VkDevice device,
            uint32_t count, const VkMappedMemoryRange* ranges) {
        assert(device == DEVICE && count == 1 && ranges);
        assert(ranges[0].offset == 0 && ranges[0].size == VK_WHOLE_SIZE);
        ++trace.invalidates;
        trace.events.emplace_back("invalidate");
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL createCommandPool(VkDevice device,
            const VkCommandPoolCreateInfo* info, const VkAllocationCallbacks*,
            VkCommandPool* pool) {
        assert(device == DEVICE && info && pool);
        assert(info->queueFamilyIndex == 7);
        assert((info->flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) != 0);
        ++trace.commandPoolCreates;
        *pool = handle<VkCommandPool>(0x5000);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL destroyCommandPool(VkDevice device,
            VkCommandPool pool, const VkAllocationCallbacks*) {
        assert(device == DEVICE && pool == handle<VkCommandPool>(0x5000));
        ++trace.commandPoolDestroys;
    }

    VKAPI_ATTR VkResult VKAPI_CALL allocateCommandBuffers(VkDevice device,
            const VkCommandBufferAllocateInfo* info, VkCommandBuffer* commands) {
        assert(device == DEVICE && info && commands);
        assert(info->commandPool == handle<VkCommandPool>(0x5000));
        assert(info->commandBufferCount == 1);
        ++trace.commandAllocations;
        for (uint32_t i = 0; i < info->commandBufferCount; ++i) {
            commands[i] = handle<VkCommandBuffer>(0x6000 + i);
            trace.commands.push_back({commands[i], VK_NULL_HANDLE, VK_NULL_HANDLE});
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL freeCommandBuffers(VkDevice device,
            VkCommandPool pool, uint32_t count, const VkCommandBuffer* commands) {
        assert(device == DEVICE && pool == handle<VkCommandPool>(0x5000));
        assert(count == 1 && commands && commands[0] != VK_NULL_HANDLE);
        ++trace.commandFrees;
    }

    VKAPI_ATTR VkResult VKAPI_CALL resetCommandBuffer(VkCommandBuffer command,
            VkCommandBufferResetFlags) {
        auto& state = commandState(command);
        state.image = VK_NULL_HANDLE;
        state.buffer = VK_NULL_HANDLE;
        ++trace.commandResets;
        trace.events.emplace_back("reset-command");
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL beginCommandBuffer(VkCommandBuffer command,
            const VkCommandBufferBeginInfo* info) {
        static_cast<void>(commandState(command));
        assert(info && info->flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        ++trace.begins;
        trace.events.emplace_back("begin");
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL endCommandBuffer(VkCommandBuffer command) {
        static_cast<void>(commandState(command));
        ++trace.ends;
        trace.events.emplace_back("end");
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL cmdPipelineBarrier(VkCommandBuffer command,
            VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
            VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t,
            const VkBufferMemoryBarrier*, uint32_t imageBarrierCount,
            const VkImageMemoryBarrier* imageBarriers) {
        static_cast<void>(commandState(command));
        assert(imageBarrierCount == 1 && imageBarriers);
        trace.barriers.push_back({srcStage, dstStage, imageBarriers[0]});
        trace.events.emplace_back("barrier");
    }

    VKAPI_ATTR void VKAPI_CALL cmdCopyImageToBuffer(VkCommandBuffer command,
            VkImage image, VkImageLayout layout, VkBuffer buffer,
            uint32_t regionCount, const VkBufferImageCopy* regions) {
        assert(layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(regionCount == 1 && regions);
        assert(regions[0].bufferOffset == 0 && regions[0].bufferRowLength == 0
            && regions[0].bufferImageHeight == 0);
        auto& state = commandState(command);
        state.image = image;
        state.buffer = buffer;
        ++trace.copies;
        trace.events.emplace_back("copy");
    }

    VKAPI_ATTR VkResult VKAPI_CALL createFence(VkDevice device,
            const VkFenceCreateInfo* info, const VkAllocationCallbacks*, VkFence* fence) {
        assert(device == DEVICE && info && fence);
        assert(info->flags == VK_FENCE_CREATE_SIGNALED_BIT);
        *fence = handle<VkFence>(0x7000 + trace.fences.size());
        trace.fences.push_back({*fence, true});
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL destroyFence(VkDevice device, VkFence fence,
            const VkAllocationCallbacks*) {
        assert(device == DEVICE && fence != VK_NULL_HANDLE);
        ++trace.fenceDestroys;
    }

    VKAPI_ATTR VkResult VKAPI_CALL resetFences(VkDevice device, uint32_t count,
            const VkFence* fences) {
        assert(device == DEVICE && count == 1 && fences);
        fenceState(fences[0]).signaled = false;
        ++trace.fenceResets;
        trace.events.emplace_back("reset-fence");
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL getFenceStatus(VkDevice device, VkFence fence) {
        assert(device == DEVICE);
        ++trace.fenceStatusChecks;
        trace.events.emplace_back("get-fence-status");
        return fenceState(fence).signaled ? VK_SUCCESS : VK_NOT_READY;
    }

    VKAPI_ATTR VkResult VKAPI_CALL queueSubmit(VkQueue queue, uint32_t count,
            const VkSubmitInfo* submits, VkFence fence) {
        assert(queue == QUEUE && count == 1 && submits && fence != VK_NULL_HANDLE);
        assert(submits[0].waitSemaphoreCount == 0);
        assert(submits[0].signalSemaphoreCount == 0);
        assert(submits[0].commandBufferCount == 1);
        const auto& command = commandState(submits[0].pCommandBuffers[0]);
        assert(command.image != VK_NULL_HANDLE && command.buffer != VK_NULL_HANDLE);
        auto& bytes = memoryState(bufferState(command.buffer).memory).bytes;
        if (command.image == handle<VkImage>(0x8000)) {
            const uint8_t values[]{0, 1, 2, 3, 4, 0, 6, 7};
            std::copy(std::begin(values), std::end(values), bytes.begin());
        } else if (command.image == handle<VkImage>(0x8001)) {
            const uint8_t values[]{9, 0, 9, 0};
            std::copy(std::begin(values), std::end(values), bytes.begin());
        } else {
            assert(command.image == handle<VkImage>(0x8002));
            const uint8_t values[]{0, 5, 0, 5};
            std::copy(std::begin(values), std::end(values), bytes.begin());
        }
        ++trace.submits;
        trace.events.emplace_back("submit");
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL waitForFences(VkDevice, uint32_t,
            const VkFence*, VkBool32, uint64_t) {
        ++trace.waitForFences;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL queueWaitIdle(VkQueue) {
        ++trace.queueWaitIdle;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL deviceWaitIdle(VkDevice) {
        ++trace.deviceWaitIdle;
        return VK_SUCCESS;
    }

    vk::RuntimeExchangeEndpoint endpoint() {
        VkPhysicalDeviceMemoryProperties properties{};
        properties.memoryTypeCount = 1;
        properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        vk::RuntimeExchangeEndpoint result{};
        result.bufferDevice = {
            .device = DEVICE,
            .memoryProperties = properties,
            .funcs = {
                .CreateBuffer = createBuffer,
                .DestroyBuffer = destroyBuffer,
                .GetBufferMemoryRequirements = getBufferMemoryRequirements,
                .AllocateMemory = allocateMemory,
                .FreeMemory = freeMemory,
                .BindBufferMemory = bindBufferMemory
            }
        };
        result.queue = QUEUE;
        result.queueFamilyIndex = 7;
        result.QueueSubmit = queueSubmit;
        result.CreateFence = createFence;
        result.DestroyFence = destroyFence;
        result.GetFenceStatus = getFenceStatus;
        result.WaitForFences = waitForFences;
        result.ResetFences = resetFences;
        result.DeviceWaitIdle = deviceWaitIdle;
        result.CreateCommandPool = createCommandPool;
        result.DestroyCommandPool = destroyCommandPool;
        result.AllocateCommandBuffers = allocateCommandBuffers;
        result.FreeCommandBuffers = freeCommandBuffers;
        result.BeginCommandBuffer = beginCommandBuffer;
        result.EndCommandBuffer = endCommandBuffer;
        result.CmdPipelineBarrier = cmdPipelineBarrier;
        result.MapMemory = mapMemory;
        result.UnmapMemory = unmapMemory;
        result.InvalidateMappedMemoryRanges = invalidateMappedMemoryRanges;
        result.CmdCopyImageToBuffer = cmdCopyImageToBuffer;
        result.ResetCommandBuffer = resetCommandBuffer;
        // queueWaitIdle deliberately cannot be wired: the endpoint has no such
        // dispatch member.  Keeping the fake lets the final zero-use assertion
        // cover the complete forbidden-wait inventory.
        static_cast<void>(&queueWaitIdle);
        return result;
    }

    void expectInvalid(vk::RuntimeImageObservationSession& session,
            const vk::RuntimeImageObservationDescriptor& descriptor) {
        const auto resets = trace.fenceResets;
        const auto commandResetCount = trace.commandResets;
        const auto submits = trace.submits;
        bool rejected{};
        try {
            static_cast<void>(session.submit(descriptor));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
        assert(trace.fenceResets == resets);
        assert(trace.commandResets == commandResetCount);
        assert(trace.submits == submits);
    }

}

int main() {
    assert(vk::runtimeImageObservationByteCount(
        VK_FORMAT_R8G8B8A8_UNORM, {2, 1}) == 8);
    assert(vk::runtimeImageObservationByteCount(
        VK_FORMAT_B8G8R8A8_UNORM, {1, 1}) == 4);
    assert(!vk::runtimeImageObservationByteCount(VK_FORMAT_R8_UNORM, {1, 1}));
    assert(!vk::runtimeImageObservationByteCount(
        VK_FORMAT_R8G8B8A8_UNORM, {0, 1}));

    {
        auto session = vk::createRuntimeImageObservationSession(
            endpoint(), {.slotCount = 1, .maxByteCount = CAPACITY});
        assert(session.valid() && session.slotCount() == 1);
        assert(session.maxByteCount() == CAPACITY && session.inFlightCount() == 0);
        assert(trace.commandPoolCreates == 1 && trace.commandAllocations == 1);
        assert(trace.buffers.size() == 1 && trace.memories.size() == 1);
        assert(trace.fences.size() == 1 && trace.maps == 1);

        auto validLifetime = std::make_shared<const uint8_t>(0);
        vk::RuntimeImageObservationDescriptor valid{
            .image = handle<VkImage>(0x8000),
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {2, 1},
            .queueFamilyIndex = 7,
            .lifetime = validLifetime
        };

        auto invalid = valid;
        invalid.image = VK_NULL_HANDLE;
        expectInvalid(session, invalid);
        invalid = valid;
        invalid.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        expectInvalid(session, invalid);
        invalid = valid;
        invalid.format = VK_FORMAT_R8_UNORM;
        expectInvalid(session, invalid);
        invalid = valid;
        invalid.extent = {20, 1};
        expectInvalid(session, invalid);
        invalid = valid;
        invalid.queueFamilyIndex = 8;
        expectInvalid(session, invalid);
        invalid = valid;
        invalid.lifetime.reset();
        expectInvalid(session, invalid);

        const auto first = session.submit(valid);
        assert(first.status == vk::RuntimeImageObservationSubmitStatus::SUBMITTED);
        assert(first.ticket.valid() && first.ticket.slotIndex == 0
            && first.ticket.epoch == 1);
        assert(session.inFlightCount() == 1);
        assert(trace.fenceResets == 1 && trace.commandResets == 1
            && trace.begins == 1 && trace.ends == 1 && trace.copies == 1
            && trace.submits == 1);
        assert(trace.barriers.size() == 2);
        assert(trace.barriers[0].srcStage == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        assert(trace.barriers[0].dstStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
        assert(trace.barriers[0].barrier.oldLayout == VK_IMAGE_LAYOUT_GENERAL);
        assert(trace.barriers[0].barrier.newLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[1].barrier.oldLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[1].barrier.newLayout == VK_IMAGE_LAYOUT_GENERAL);
        assert(trace.barriers[1].dstStage == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        assert(trace.barriers[1].barrier.dstAccessMask
            == (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT));
        for (const auto& barrier : trace.barriers) {
            assert(barrier.barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
            assert(barrier.barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        }
        const std::vector<std::string> expectedSubmitEvents{
            "reset-fence", "reset-command", "begin", "barrier", "copy",
            "barrier", "end", "submit"
        };
        assert(trace.events == expectedSubmitEvents);

        std::weak_ptr<const uint8_t> retained = validLifetime;
        validLifetime.reset();
        assert(!retained.expired());

        auto blockedLifetime = std::make_shared<const uint8_t>(0);
        auto blockedDescriptor = valid;
        blockedDescriptor.lifetime = blockedLifetime;
        const auto resetsBeforeBlock = trace.fenceResets;
        const auto commandResetsBeforeBlock = trace.commandResets;
        const auto submitsBeforeBlock = trace.submits;
        const auto blocked = session.submit(blockedDescriptor);
        assert(blocked.status
            == vk::RuntimeImageObservationSubmitStatus::TEMPORARILY_BLOCKED);
        assert(!blocked.ticket.valid());
        assert(trace.fenceResets == resetsBeforeBlock);
        assert(trace.commandResets == commandResetsBeforeBlock);
        assert(trace.submits == submitsBeforeBlock);

        const auto pending = session.tryRetire(first.ticket);
        assert(pending.status == vk::RuntimeImageObservationRetireStatus::NOT_READY);
        assert(!pending.observation && session.inFlightCount() == 1);
        assert(trace.fenceStatusChecks == 1 && trace.invalidates == 0);
        fenceState(trace.fences[0].fence).signaled = true;

        const uint8_t expectedABytes[]{0, 1, 2, 3, 4, 0, 6, 7};
        const auto retired = session.tryRetire(first.ticket);
        assert(retired.status == vk::RuntimeImageObservationRetireStatus::RETIRED);
        assert(retired.observation && retired.observation->image == valid.image);
        assert(retired.observation->layout == VK_IMAGE_LAYOUT_GENERAL);
        assert(retired.observation->format == VK_FORMAT_R8G8B8A8_UNORM);
        assert(retired.observation->extent.width == 2
            && retired.observation->extent.height == 1);
        assert(retired.observation->queueFamilyIndex == 7);
        assert(retired.observation->byteCount == 8);
        assert(retired.observation->nonzeroByteCount == 6);
        assert(retired.observation->checksum == lsfgvk::common::fnv1a64(
            expectedABytes, std::size(expectedABytes)));
        assert(session.inFlightCount() == 0 && retained.expired());
        assert(trace.fenceStatusChecks == 2 && trace.invalidates == 1);

        const auto checksBeforeStale = trace.fenceStatusChecks;
        bool staleRejected{};
        try {
            static_cast<void>(session.tryRetire(first.ticket));
        } catch (const std::invalid_argument&) {
            staleRejected = true;
        }
        assert(staleRejected && trace.fenceStatusChecks == checksBeforeStale);

        auto secondLifetime = std::make_shared<const uint8_t>(0);
        const vk::RuntimeImageObservationDescriptor secondDescriptor{
            .image = handle<VkImage>(0x8001),
            .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .format = VK_FORMAT_B8G8R8A8_UNORM,
            .extent = {1, 1},
            .queueFamilyIndex = 7,
            .lifetime = secondLifetime
        };
        const auto second = session.submit(secondDescriptor);
        assert(second.status == vk::RuntimeImageObservationSubmitStatus::SUBMITTED);
        assert(second.ticket.slotIndex == first.ticket.slotIndex);
        assert(second.ticket.epoch == first.ticket.epoch + 1);
        assert(trace.buffers.size() == 1 && trace.memories.size() == 1
            && trace.fences.size() == 1 && trace.commandAllocations == 1);
        assert(trace.barriers.size() == 4);
        assert(trace.barriers[2].barrier.oldLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[2].barrier.newLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[3].barrier.oldLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[3].barrier.newLayout
            == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(trace.barriers[3].dstStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
        assert(trace.barriers[3].barrier.dstAccessMask
            == VK_ACCESS_TRANSFER_READ_BIT);

        bool oldEpochRejected{};
        try {
            static_cast<void>(session.tryRetire(first.ticket));
        } catch (const std::invalid_argument&) {
            oldEpochRejected = true;
        }
        assert(oldEpochRejected && trace.fenceStatusChecks == checksBeforeStale);

        fenceState(trace.fences[0].fence).signaled = true;
        const uint8_t expectedBBytes[]{9, 0, 9, 0};
        const auto secondRetired = session.tryRetire(second.ticket);
        assert(secondRetired.status
            == vk::RuntimeImageObservationRetireStatus::RETIRED);
        assert(secondRetired.observation->byteCount == 4);
        assert(secondRetired.observation->nonzeroByteCount == 2);
        assert(secondRetired.observation->checksum == lsfgvk::common::fnv1a64(
            expectedBBytes, std::size(expectedBBytes)));

        auto thirdLifetime = std::make_shared<const uint8_t>(0);
        auto thirdDescriptor = secondDescriptor;
        thirdDescriptor.image = handle<VkImage>(0x8002);
        thirdDescriptor.lifetime = thirdLifetime;
        const auto third = session.submit(thirdDescriptor);
        assert(third.status == vk::RuntimeImageObservationSubmitStatus::SUBMITTED);
        assert(third.ticket.slotIndex == first.ticket.slotIndex);
        assert(third.ticket.epoch == second.ticket.epoch + 1);
        assert(trace.buffers.size() == 1 && trace.memories.size() == 1
            && trace.fences.size() == 1 && trace.commandAllocations == 1);
        fenceState(trace.fences[0].fence).signaled = true;
        const uint8_t expectedCBytes[]{0, 5, 0, 5};
        const auto thirdRetired = session.tryRetire(third.ticket);
        assert(thirdRetired.status
            == vk::RuntimeImageObservationRetireStatus::RETIRED);
        assert(thirdRetired.observation->byteCount == 4);
        assert(thirdRetired.observation->nonzeroByteCount == 2);
        assert(thirdRetired.observation->checksum == lsfgvk::common::fnv1a64(
            expectedCBytes, std::size(expectedCBytes)));
        assert(retired.observation->checksum != secondRetired.observation->checksum);
        assert(retired.observation->checksum != thirdRetired.observation->checksum);
        assert(secondRetired.observation->checksum
            != thirdRetired.observation->checksum);
    }

    assert(trace.commandFrees == 1 && trace.commandPoolDestroys == 1);
    assert(trace.bufferDestroys == 1 && trace.memoryFrees == 1);
    assert(trace.unmaps == 1 && trace.fenceDestroys == 1);
    assert(trace.waitForFences == 0 && trace.queueWaitIdle == 0
        && trace.deviceWaitIdle == 0);
    return 0;
}
