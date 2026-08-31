#include "present_batch_projection.hpp"

#include <cassert>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <vector>

using namespace lsfgvk::layer;

static std::atomic_bool failAlloc{};
void* operator new(std::size_t size) {
    if (failAlloc.load(std::memory_order_relaxed)) throw std::bad_alloc{};
    if (auto* value = std::malloc(size)) return value;
    throw std::bad_alloc{};
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

namespace {
template <typename Handle>
Handle fakeHandle(uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}
}

int main() {
    {
        const std::vector<VkResult> values{VK_SUCCESS, VK_SUBOPTIMAL_KHR};
        assert(aggregatePresentResults(values) == VK_SUBOPTIMAL_KHR);
    }
    {
        const std::vector<VkResult> values{VK_ERROR_OUT_OF_DATE_KHR, VK_SUCCESS};
        assert(aggregatePresentResults(values) == VK_ERROR_OUT_OF_DATE_KHR);
    }
    {
        const std::vector<VkResult> values{VK_SUBOPTIMAL_KHR, VK_ERROR_SURFACE_LOST_KHR};
        assert(aggregatePresentResults(values) == VK_ERROR_SURFACE_LOST_KHR);
    }
    {
        const std::vector<VkResult> values{VK_ERROR_OUT_OF_DATE_KHR, VK_ERROR_DEVICE_LOST};
        assert(aggregatePresentResults(values) == VK_ERROR_DEVICE_LOST);
    }

#if defined(VK_KHR_present_id)
    {
        VkPresentIdKHR presentIds{
            .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .pNext = nullptr,
            .swapchainCount = 2,
            .pPresentIds = nullptr};
        auto projected = PresentPNextProjection::build(&presentIds, 2, 0);
        assert(projected.supported());
        auto* value = reinterpret_cast<VkPresentIdKHR*>(projected.head());
        assert(value && value->swapchainCount == 1 && value->pPresentIds == nullptr);
    }
    {
        const uint64_t ids[]{101, 102};
        VkPresentIdKHR input{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
        input.swapchainCount = 2; input.pPresentIds = ids;
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        assert(batch.supported());
        auto* output = reinterpret_cast<VkPresentIdKHR*>(batch.head());
        assert(output && output->swapchainCount == 2);
        assert(output->pPresentIds == ids && ids[0] == 101 && ids[1] == 102);

        failAlloc.store(true, std::memory_order_relaxed);
        auto failed = PresentBatchPNextProjection::build(&input, 2, paths);
        failAlloc.store(false, std::memory_order_relaxed);
        assert(!failed.supported() && failed.head() == nullptr);
    }
#endif

#if defined(VK_KHR_present_id2)
    {
        const uint64_t ids[]{201, 202};
        VkPresentId2KHR input{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
        input.swapchainCount = 2; input.pPresentIds = ids;
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        assert(batch.supported());
        auto* output = reinterpret_cast<VkPresentId2KHR*>(batch.head());
        assert(output && output->swapchainCount == 2
            && output->pPresentIds == ids);
    }
#endif

#if defined(VK_KHR_device_group)
    {
        VkDeviceGroupPresentInfoKHR deviceGroup{
            .sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .swapchainCount = 0,
            .pDeviceMasks = nullptr,
            .mode = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR};
        auto projected = PresentPNextProjection::build(&deviceGroup, 2, 1);
        assert(projected.supported());
        auto* value = reinterpret_cast<VkDeviceGroupPresentInfoKHR*>(projected.head());
        assert(value && value->swapchainCount == 0 && value->pDeviceMasks == nullptr);
    }
    {
        const uint32_t masks[]{1, 2};
        VkDeviceGroupPresentInfoKHR input{
            .sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
            .swapchainCount = 2, .pDeviceMasks = masks,
            .mode = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR};
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        auto* output = reinterpret_cast<VkDeviceGroupPresentInfoKHR*>(batch.head());
        assert(batch.supported() && output && output->pDeviceMasks == masks);
    }
#endif

#if defined(VK_KHR_incremental_present)
    {
        const VkPresentRegionKHR regions[2]{};
        VkPresentRegionsKHR input{VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR};
        input.swapchainCount = 2; input.pRegions = regions;
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        auto* output = reinterpret_cast<VkPresentRegionsKHR*>(batch.head());
        assert(batch.supported() && output && output->pRegions == regions);
    }
#endif

#if defined(VK_GOOGLE_display_timing)
    {
        const VkPresentTimeGOOGLE times[2]{};
        VkPresentTimesInfoGOOGLE input{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
        input.swapchainCount = 2; input.pTimes = times;
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        auto* output = reinterpret_cast<VkPresentTimesInfoGOOGLE*>(batch.head());
        assert(batch.supported() && output && output->pTimes == times);
    }
#endif

#if defined(VK_EXT_present_timing)
    {
        const VkPresentTimingInfoEXT timings[2]{};
        VkPresentTimingsInfoEXT input{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
        input.swapchainCount = 2; input.pTimingInfos = timings;
        std::vector<PresentBatchPNextProjection::Path> paths(2,
            PresentBatchPNextProjection::Path::VirtualPrepared);
        auto batch = PresentBatchPNextProjection::build(&input, 2, paths);
        auto* output = reinterpret_cast<VkPresentTimingsInfoEXT*>(batch.head());
        assert(batch.supported() && output && output->pTimingInfos == timings);
    }
#endif

#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
    VkFence fences[] = {
        fakeHandle<VkFence>(0x11),
        fakeHandle<VkFence>(0x22)};
    VkPresentModeKHR modes[] = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR};
    VkSwapchainPresentModeInfoKHR modeInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
        .pNext = nullptr,
        .swapchainCount = 2,
        .pPresentModes = modes};
    VkSwapchainPresentFenceInfoKHR fenceInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .pNext = &modeInfo,
        .swapchainCount = 2,
        .pFences = fences};

    const auto originalFenceNext = fenceInfo.pNext;
    const auto originalMode0 = modes[0];
    const auto originalMode1 = modes[1];

    auto projected = PresentPNextProjection::build(&fenceInfo, 2, 1);
    assert(projected.supported());
    auto* projectedFence = reinterpret_cast<VkSwapchainPresentFenceInfoKHR*>(projected.head());
    assert(projectedFence);
    assert(projectedFence != &fenceInfo);
    assert(projectedFence->swapchainCount == 1);
    assert(projectedFence->pFences && projectedFence->pFences[0] == fences[1]);
    auto* projectedMode = reinterpret_cast<VkSwapchainPresentModeInfoKHR*>(
        const_cast<void*>(projectedFence->pNext));
    assert(projectedMode);
    assert(projectedMode != &modeInfo);
    assert(projectedMode->swapchainCount == 1);
    assert(projectedMode->pPresentModes[0] == modes[1]);

    // Caller memory is unchanged.
    assert(fenceInfo.swapchainCount == 2);
    assert(fenceInfo.pNext == originalFenceNext);
    assert(modeInfo.swapchainCount == 2);
    assert(modes[0] == originalMode0 && modes[1] == originalMode1);

    const std::vector<PresentBatchPNextProjection::Path> paths{
        PresentBatchPNextProjection::Path::VirtualPrepared,
        PresentBatchPNextProjection::Path::Native};
    auto batch = PresentBatchPNextProjection::build(&fenceInfo, 2, paths);
    assert(batch.supported());
    assert(batch.applicationFence(0) == fences[0]);
    assert(batch.applicationFence(1) == fences[1]);
    auto* batchFence = reinterpret_cast<VkSwapchainPresentFenceInfoKHR*>(
        batch.head());
    assert(batchFence && batchFence->swapchainCount == 2);
    assert(batchFence->pFences != fences);
    assert(batchFence->pFences[0] == fences[0]);
    assert(!batch.patchInternalPresentFence(0, fakeHandle<VkFence>(0x33)));

    VkFence nullFences[]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkSwapchainPresentFenceInfoKHR nullFenceInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .swapchainCount = 2,
        .pFences = nullFences};
    auto patchable = PresentBatchPNextProjection::build(
        &nullFenceInfo, 2, paths);
    assert(patchable.supported());
    assert(patchable.patchInternalPresentFence(
        0, fakeHandle<VkFence>(0x44)));
    assert(!patchable.patchInternalPresentFence(
        1, fakeHandle<VkFence>(0x55)));
    auto* patched = reinterpret_cast<VkSwapchainPresentFenceInfoKHR*>(
        patchable.head());
    assert(patched->pFences[0] == fakeHandle<VkFence>(0x44));
    assert(patched->pFences[1] == VK_NULL_HANDLE);
    assert(nullFences[0] == VK_NULL_HANDLE && nullFences[1] == VK_NULL_HANDLE);

    VkFence mixedFences[]{fakeHandle<VkFence>(0x66), VK_NULL_HANDLE,
        VK_NULL_HANDLE};
    VkSwapchainPresentFenceInfoKHR mixedInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .swapchainCount = 3, .pFences = mixedFences};
    std::vector<PresentBatchPNextProjection::Path> mixedPaths{
        PresentBatchPNextProjection::Path::VirtualPrepared,
        PresentBatchPNextProjection::Path::Native,
        PresentBatchPNextProjection::Path::D2};
    auto mixed = PresentBatchPNextProjection::build(&mixedInfo, 3, mixedPaths);
    assert(mixed.supported());
    assert(!mixed.patchInternalPresentFence(0, fakeHandle<VkFence>(0x70)));
    assert(!mixed.patchInternalPresentFence(1, fakeHandle<VkFence>(0x71)));
    assert(!mixed.patchInternalPresentFence(2, fakeHandle<VkFence>(0x72)));
    auto* mixedOutput = reinterpret_cast<VkSwapchainPresentFenceInfoKHR*>(
        mixed.head());
    assert(mixedOutput->pFences[0] == mixedFences[0]);
    assert(mixedOutput->pFences[1] == VK_NULL_HANDLE);
    assert(mixedOutput->pFences[2] == VK_NULL_HANDLE);

    auto allocationFree = PresentBatchPNextProjection::build(
        &nullFenceInfo, 2, paths);
    failAlloc.store(true, std::memory_order_relaxed);
    assert(allocationFree.patchInternalPresentFence(
        0, fakeHandle<VkFence>(0x73)));
    failAlloc.store(false, std::memory_order_relaxed);
#endif

    std::vector<PresentBatchPNextProjection::Path> noChainPaths{
        PresentBatchPNextProjection::Path::VirtualPrepared,
        PresentBatchPNextProjection::Path::D2};
    auto noChain = PresentBatchPNextProjection::build(nullptr, 2, noChainPaths);
    assert(noChain.supported());
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
    assert(noChain.head() != nullptr);
    assert(noChain.applicationFence(0) == VK_NULL_HANDLE);
    assert(noChain.patchInternalPresentFence(
        0, fakeHandle<VkFence>(0x74)));
#else
    assert(noChain.head() == nullptr);
#endif

    VkBaseInStructure unsupported{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = nullptr};
    auto rejected = PresentBatchPNextProjection::build(
        &unsupported, 2, noChainPaths);
    assert(!rejected.supported());
    assert(rejected.unsupportedSType() == VK_STRUCTURE_TYPE_APPLICATION_INFO);

    return 0;
}
