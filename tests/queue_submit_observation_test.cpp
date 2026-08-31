/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/queue_submit.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"

#include <cassert>
#include <cstdint>
#include <array>

namespace {
template <typename T> T handle(uintptr_t value) { return reinterpret_cast<T>(value); }

const VkSubmitInfo* expectedSubmits{};
uint32_t expectedCount{};
VkQueue expectedQueue{};
VkFence expectedFence{};
VkResult scriptedResult{VK_SUCCESS};

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count,
        const VkSubmitInfo* submits, VkFence fence) {
    assert(queue == expectedQueue && count == expectedCount);
    assert(submits == expectedSubmits && fence == expectedFence);
    return scriptedResult;
}
}

int main() {
    const VkSemaphore wait = handle<VkSemaphore>(0x10);
    const VkSemaphore signal = handle<VkSemaphore>(0x11);
    const VkCommandBuffer command = handle<VkCommandBuffer>(0x12);
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &wait,
        &stage, 1, &command, 1, &signal};
    expectedQueue = handle<VkQueue>(0x20);
    expectedFence = handle<VkFence>(0x21);
    expectedSubmits = &info;
    expectedCount = 1;
    scriptedResult = VK_ERROR_DEVICE_LOST;
    bool before{};
    bool after{};
    const auto result = vk::executeQueueSubmit(submit, expectedQueue, 1, &info,
        expectedFence,
        [&](VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
            assert(queue == expectedQueue && count == 1 && submits == &info);
            assert(fence == expectedFence && submits->pWaitSemaphores == &wait);
            assert(submits->pSignalSemaphores == &signal
                && submits->pCommandBuffers == &command);
            before = true;
        },
        [&](VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                VkFence fence, VkResult observed) {
            assert(before && queue == expectedQueue && count == 1 && submits == &info);
            assert(fence == expectedFence && observed == scriptedResult);
            after = true;
        });
    assert(result == scriptedResult && before && after);

    const std::array<VkSemaphore, 2> binaryWaits{
        handle<VkSemaphore>(0x30), handle<VkSemaphore>(0x31)};
    const std::array<VkPipelineStageFlags, 2> binaryStages{
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT};
    const std::array<vk::TimelineWait, 2> timelineWaits{
        vk::TimelineWait{handle<VkSemaphore>(0x32), 41,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT},
        vk::TimelineWait{handle<VkSemaphore>(0x33), 7,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};
    const std::array<VkSemaphore, 2> binarySignals{
        handle<VkSemaphore>(0x34), handle<VkSemaphore>(0x35)};
    const std::array<vk::TimelineSignal, 2> timelineSignals{
        vk::TimelineSignal{handle<VkSemaphore>(0x36), 43},
        vk::TimelineSignal{handle<VkSemaphore>(0x37), 7}};
    const vk::CommandBufferSubmit multi{
        .binaryWaits = binaryWaits,
        .binaryWaitStages = binaryStages,
        .timelineWaits = timelineWaits,
        .binarySignals = binarySignals,
        .timelineSignals = timelineSignals};
    vk::PreparedCommandBufferSubmit prepared;
    vk::prepareCommandBufferSubmit(command, multi, prepared);
    const auto& observed = prepared.submitInfo;
    const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(
        observed.pNext);
    assert(observed.waitSemaphoreCount == 4);
    assert(observed.pWaitSemaphores[0] == binaryWaits[0]
        && observed.pWaitSemaphores[1] == binaryWaits[1]);
    assert(observed.pWaitSemaphores[2] == timelineWaits[0].semaphore
        && observed.pWaitSemaphores[3] == timelineWaits[1].semaphore);
    assert(observed.pWaitDstStageMask[2] == timelineWaits[0].stage
        && observed.pWaitDstStageMask[3] == timelineWaits[1].stage);
    assert(timeline->pWaitSemaphoreValues[0] == 0
        && timeline->pWaitSemaphoreValues[1] == 0
        && timeline->pWaitSemaphoreValues[2] == 41
        && timeline->pWaitSemaphoreValues[3] == 7);
    assert(observed.signalSemaphoreCount == 4);
    assert(observed.pSignalSemaphores[0] == binarySignals[0]
        && observed.pSignalSemaphores[1] == binarySignals[1]);
    assert(observed.pSignalSemaphores[2] == timelineSignals[0].semaphore
        && observed.pSignalSemaphores[3] == timelineSignals[1].semaphore);
    assert(timeline->pSignalSemaphoreValues[0] == 0
        && timeline->pSignalSemaphoreValues[1] == 0
        && timeline->pSignalSemaphoreValues[2] == 43
        && timeline->pSignalSemaphoreValues[3] == 7);
    assert(observed.pCommandBuffers[0] == command);
}
