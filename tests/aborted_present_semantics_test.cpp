#include "aborted_present_semantics.hpp"
#include <array>
#include <cassert>

using namespace lsfgvk::layer;

int main() {
    assert(classifyPresentResult(VK_ERROR_OUT_OF_HOST_MEMORY)
        == PresentResultClass::PreEnqueueFailure);
    assert(classifyPresentResult(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        == PresentResultClass::PreEnqueueFailure);
    assert(classifyPresentResult(VK_ERROR_OUT_OF_DATE_KHR)
        == PresentResultClass::EnqueuedRejection);
    assert(classifyPresentResult(VK_ERROR_SURFACE_LOST_KHR)
        == PresentResultClass::EnqueuedRejection);
    assert(classifyPresentResult(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
        == PresentResultClass::EnqueuedRejection);
    assert(classifyPresentResult(VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT)
        == PresentResultClass::EnqueuedRejection);
    assert(classifyPresentResult(VK_ERROR_DEVICE_LOST)
        == PresentResultClass::DeviceLost);
    assert(classifyPresentResult(VK_SUCCESS) == PresentResultClass::Normal);
    assert(classifyPresentResult(VK_SUBOPTIMAL_KHR) == PresentResultClass::Normal);
    assert(classifyPresentResult(VK_ERROR_VALIDATION_FAILED_EXT)
        == PresentResultClass::Indeterminate);

    const auto queue = reinterpret_cast<VkQueue>(uintptr_t{0xA001});
    const auto otherQueue = reinterpret_cast<VkQueue>(uintptr_t{0xA002});
    const auto wait = reinterpret_cast<VkSemaphore>(uintptr_t{0xA003});
    const auto fence = reinterpret_cast<VkFence>(uintptr_t{0xA004});
    struct Counters {
        uint32_t waitConsumptions{};
        uint32_t bridgeSubmits{};
        uint32_t compensationSubmits{};
        uint32_t downstreamPresents{};
        uint32_t fenceUses{};
        uint32_t hiddenWaits{};
        VkQueue bridgeQueue{};
        VkQueue compensationQueue{};
        VkSemaphore consumedWait{};
    };

    // Class A: the synthetic error occurs before any queue operation.
    for (const auto result : std::array{VK_ERROR_OUT_OF_HOST_MEMORY,
            VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        Counters counters;
        assert(classifyPresentResult(result) == PresentResultClass::PreEnqueueFailure);
        assert(counters.waitConsumptions == 0 && counters.bridgeSubmits == 0
            && counters.compensationSubmits == 0
            && counters.downstreamPresents == 0 && counters.fenceUses == 0
            && counters.hiddenWaits == 0);
    }

    // Class B: model the production bridge as the unique consumer of S, then
    // invoke the same compensation helper used by Swapchain on the same queue.
    for (const auto result : std::array{VK_ERROR_OUT_OF_DATE_KHR,
            VK_ERROR_SURFACE_LOST_KHR,
            VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
            VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT}) {
        Counters counters;
        counters.bridgeSubmits++;
        counters.waitConsumptions++;
        counters.bridgeQueue = queue;
        counters.consumedWait = wait;
        const auto compensation = compensateEnqueuedPresentAbort(queue, fence,
            [&](VkQueue actualQueue, const VkSubmitInfo& submit, VkFence actualFence) {
                ++counters.compensationSubmits;
                counters.compensationQueue = actualQueue;
                assert(submit.waitSemaphoreCount == 0
                    && submit.signalSemaphoreCount == 0);
                assert(actualFence == fence);
                ++counters.fenceUses;
                return VK_SUCCESS;
            });
        assert(classifyPresentResult(result) == PresentResultClass::EnqueuedRejection);
        assert(compensation == VK_SUCCESS && counters.waitConsumptions == 1
            && counters.bridgeSubmits == 1 && counters.compensationSubmits == 1
            && counters.downstreamPresents == 0 && counters.fenceUses == 1
            && counters.hiddenWaits == 0 && counters.consumedWait == wait
            && counters.bridgeQueue == counters.compensationQueue
            && counters.compensationQueue != otherQueue);

        Counters noFence;
        noFence.bridgeSubmits = noFence.waitConsumptions = 1;
        noFence.bridgeQueue = queue;
        assert(compensateEnqueuedPresentAbort(queue, VK_NULL_HANDLE,
            [&](VkQueue, const VkSubmitInfo&, VkFence) {
                ++noFence.compensationSubmits;
                return VK_SUCCESS;
            }) == VK_SUCCESS);
        assert(noFence.compensationSubmits == 0 && noFence.hiddenWaits == 0);
    }

    // Class C never routes through ordinary-fence compensation.
    Counters deviceLost;
    assert(classifyPresentResult(VK_ERROR_DEVICE_LOST)
        == PresentResultClass::DeviceLost);
    assert(deviceLost.compensationSubmits == 0 && deviceLost.fenceUses == 0);
    return 0;
}
