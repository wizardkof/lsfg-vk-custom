#include "entrypoint_test_seam.hpp"
#include "aborted_present_semantics.hpp"
#include "batch_application_present_bridge_authority.hpp"
#include "batch_present_transaction.hpp"
#include "present_batch_projection.hpp"
#include "shadow_b_return_execution_harness.hpp"

#include <cassert>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>

#include <sys/mman.h>
#include <unistd.h>

using namespace lsfgvk::layer;

std::atomic_bool failGlobalAllocation{};
void* operator new(std::size_t size) {
    if (failGlobalAllocation.load(std::memory_order_relaxed)) throw std::bad_alloc();
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

namespace {
void failAdaptiveExecuteAllocations() {
    failGlobalAllocation.store(true, std::memory_order_relaxed);
}
void allowAdaptiveExecuteAllocations() {
    failGlobalAllocation.store(false, std::memory_order_relaxed);
}
struct Capture {
    uint32_t createCalls{}, destroyCalls{}, statusCalls{}, waitCalls{}, resetCalls{}, importCalls{};
    VkDevice device{};
    VkFence returnedFence{};
    VkFence lastFence{};
    VkFence signaledFence{};
    std::vector<VkFence> waits;
    VkBool32 waitAll{};
    uint64_t timeout{};
    VkResult statusResult{VK_NOT_READY};
    VkResult waitResult{VK_SUCCESS};
    VkResult resetResult{VK_SUCCESS};
    VkResult importResult{VK_SUCCESS};
} capture;

VKAPI_ATTR VkResult VKAPI_CALL createFence(VkDevice device,
        const VkFenceCreateInfo* info, const VkAllocationCallbacks*, VkFence* fence) {
    ++capture.createCalls;
    capture.device = device;
    assert(info && info->sType == VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    *fence = capture.returnedFence;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroyFence(VkDevice device, VkFence fence,
        const VkAllocationCallbacks*) {
    ++capture.destroyCalls;
    assert(device == capture.device);
    capture.lastFence = fence;
}
VKAPI_ATTR VkResult VKAPI_CALL getFenceStatus(VkDevice device, VkFence fence) {
    ++capture.statusCalls;
    assert(device == capture.device);
    capture.lastFence = fence;
    return capture.signaledFence != VK_NULL_HANDLE
        ? (fence == capture.signaledFence ? VK_SUCCESS : VK_NOT_READY)
        : capture.statusResult;
}
VKAPI_ATTR VkResult VKAPI_CALL waitForFences(VkDevice device, uint32_t count,
        const VkFence* fences, VkBool32 waitAll, uint64_t timeout) {
    ++capture.waitCalls;
    assert(device == capture.device);
    capture.waits.assign(fences, fences + count);
    capture.waitAll = waitAll;
    capture.timeout = timeout;
    return capture.waitResult;
}
VKAPI_ATTR VkResult VKAPI_CALL resetFences(VkDevice device, uint32_t count,
        const VkFence* fences) {
    ++capture.resetCalls;
    assert(device == capture.device && count > 0 && fences);
    capture.lastFence = fences[0];
    return capture.resetResult;
}
VKAPI_ATTR VkResult VKAPI_CALL importFenceFd(VkDevice device,
        const VkImportFenceFdInfoKHR* info) {
    ++capture.importCalls;
    assert(device == capture.device && info);
    capture.lastFence = info->fence;
    return capture.importResult;
}

PresentedPhysicalImageIdentity identity(uint64_t operation, uint32_t image = 0) {
    return {71, 81, 0x1234, image, operation};
}

struct AllocatorProbe {
    bool insidePublicDestroy{};
    bool valid{true};
    uint32_t callbacks{};
    bool callbackInsideCommand{};
    bool callbackOnCallingThread{};
    bool callbackWhileValid{};
    std::thread::id callingThread;
};

VKAPI_ATTR void* VKAPI_CALL probeAllocate(
        void*, size_t, size_t, VkSystemAllocationScope) { return nullptr; }
VKAPI_ATTR void* VKAPI_CALL probeReallocate(
        void*, void*, size_t, size_t, VkSystemAllocationScope) { return nullptr; }
VKAPI_ATTR void VKAPI_CALL probeFree(void* userData, void*) {
    auto& probe = *static_cast<AllocatorProbe*>(userData);
    ++probe.callbacks;
    probe.callbackInsideCommand = probe.insidePublicDestroy;
    probe.callbackOnCallingThread =
        std::this_thread::get_id() == probe.callingThread;
    probe.callbackWhileValid = probe.valid;
}
}

int main() {
    {
        const auto pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        void* page = mmap(nullptr, pageSize, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(page != MAP_FAILED);
        auto* maintenance = new (page)
            VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
                .swapchainMaintenance1 = VK_FALSE};
        const auto maintenanceBefore = *maintenance;
        assert(mprotect(page, pageSize, PROT_READ) == 0);
        const VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = maintenance};
        assert(!lsfgvk::layer::test::maintenanceFeatureUsable(createInfo));
        assert(std::memcmp(maintenance, &maintenanceBefore,
            sizeof(maintenanceBefore)) == 0);
        assert(maintenance->sType == maintenanceBefore.sType
            && maintenance->pNext == maintenanceBefore.pNext
            && maintenance->swapchainMaintenance1
                == maintenanceBefore.swapchainMaintenance1);
        assert(munmap(page, pageSize) == 0);
    }

    {
        const auto pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        void* page = mmap(nullptr, pageSize, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(page != MAP_FAILED);
        auto* timeline = new (page) VkPhysicalDeviceVulkan12Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .timelineSemaphore = VK_FALSE};
        const auto timelineBefore = *timeline;
        assert(mprotect(page, pageSize, PROT_READ) == 0);
        const VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = timeline};
        assert(!lsfgvk::layer::test::timelineFeatureUsable(createInfo));
        assert(std::memcmp(timeline, &timelineBefore, sizeof(timelineBefore)) == 0);
        assert(timeline->sType == timelineBefore.sType
            && timeline->pNext == timelineBefore.pNext
            && timeline->timelineSemaphore == timelineBefore.timelineSemaphore);
        assert(munmap(page, pageSize) == 0);
    }

    const auto device = reinterpret_cast<VkDevice>(uintptr_t{0x7100});
    capture = {};
    capture.device = device;
    capture.returnedFence = reinterpret_cast<VkFence>(uintptr_t{0x7200});
    vk::VulkanDeviceFuncs downstream{};
    downstream.CreateFence = createFence;
    downstream.DestroyFence = destroyFence;
    downstream.GetFenceStatus = getFenceStatus;
    downstream.WaitForFences = waitForFences;
    downstream.ResetFences = resetFences;
    downstream.ImportFenceFdKHR = importFenceFd;
    assert(lsfgvk::layer::test::installDeviceEntrypointHarness(
        {device, downstream, 71}));

    const auto create = reinterpret_cast<PFN_vkCreateFence>(
        lsfgvk::layer::test::deviceEntrypoint("vkCreateFence"));
    const auto destroy = reinterpret_cast<PFN_vkDestroyFence>(
        lsfgvk::layer::test::deviceEntrypoint("vkDestroyFence"));
    const auto status = reinterpret_cast<PFN_vkGetFenceStatus>(
        lsfgvk::layer::test::deviceEntrypoint("vkGetFenceStatus"));
    const auto wait = reinterpret_cast<PFN_vkWaitForFences>(
        lsfgvk::layer::test::deviceEntrypoint("vkWaitForFences"));
    const auto reset = reinterpret_cast<PFN_vkResetFences>(
        lsfgvk::layer::test::deviceEntrypoint("vkResetFences"));
    const auto importFd = reinterpret_cast<PFN_vkImportFenceFdKHR>(
        lsfgvk::layer::test::deviceEntrypoint("vkImportFenceFdKHR"));
    assert(create && destroy && status && wait && reset && importFd);

    const VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    VkFence fence{};
    assert(create(device, &createInfo, nullptr, &fence) == VK_SUCCESS);
    assert(capture.createCalls == 1 && fence == capture.returnedFence);

    auto first = std::make_shared<int>(1);
    const auto firstIdentity = identity(91);
    assert(lsfgvk::layer::test::installPresentedLease(firstIdentity, first));
    assert(lsfgvk::layer::test::associateBorrowedFence(fence, firstIdentity));
    capture.statusResult = VK_NOT_READY;
    assert(status(device, fence) == VK_NOT_READY);
    assert(capture.statusCalls == 1
        && lsfgvk::layer::test::containsPresentedLease(firstIdentity));
    capture.statusResult = VK_SUCCESS;
    assert(status(device, fence) == VK_SUCCESS);
    assert(capture.statusCalls == 2
        && !lsfgvk::layer::test::containsPresentedLease(firstIdentity));
    assert(status(device, fence) == VK_SUCCESS && capture.statusCalls == 3);

    // waitAll retires all named tracked fences with one transparent call.
    const auto secondFence = reinterpret_cast<VkFence>(uintptr_t{0x7201});
    capture.returnedFence = secondFence;
    VkFence createdSecond{};
    assert(create(device, &createInfo, nullptr, &createdSecond) == VK_SUCCESS);
    const auto waitA = identity(92, 1), waitB = identity(93, 2);
    assert(lsfgvk::layer::test::installPresentedLease(waitA, std::make_shared<int>(2)));
    assert(lsfgvk::layer::test::installPresentedLease(waitB, std::make_shared<int>(3)));
    assert(lsfgvk::layer::test::associateBorrowedFence(fence, waitA));
    assert(lsfgvk::layer::test::associateBorrowedFence(secondFence, waitB));
    const VkFence waitFences[]{fence, secondFence};
    assert(wait(device, 2, waitFences, VK_TRUE, 987654321) == VK_SUCCESS);
    assert(capture.waitCalls == 1 && capture.waits.size() == 2
        && capture.waits[0] == fence && capture.waits[1] == secondFence
        && capture.waitAll == VK_TRUE && capture.timeout == 987654321
        && !lsfgvk::layer::test::containsPresentedLease(waitA)
        && !lsfgvk::layer::test::containsPresentedLease(waitB));

    // waitAny is followed by observation-only status queries and retires only
    // the generation whose exact fence is signaled.
    const auto anyA = identity(94, 3), anyB = identity(95, 4);
    assert(lsfgvk::layer::test::installPresentedLease(anyA, std::make_shared<int>(4)));
    assert(lsfgvk::layer::test::installPresentedLease(anyB, std::make_shared<int>(5)));
    assert(lsfgvk::layer::test::associateBorrowedFence(fence, anyA));
    assert(lsfgvk::layer::test::associateBorrowedFence(secondFence, anyB));
    capture.signaledFence = secondFence;
    const auto statusCallsBeforeWaitAny = capture.statusCalls;
    assert(wait(device, 2, waitFences, VK_FALSE, 123) == VK_SUCCESS);
    assert(capture.waitCalls == 2 && capture.waitAll == VK_FALSE
        && capture.timeout == 123
        && lsfgvk::layer::test::containsPresentedLease(anyA)
        && !lsfgvk::layer::test::containsPresentedLease(anyB)
        && capture.statusCalls == statusCallsBeforeWaitAny + 2);
    capture.signaledFence = VK_NULL_HANDLE;
    capture.statusResult = VK_NOT_READY;

    capture.resetResult = VK_ERROR_DEVICE_LOST;
    assert(reset(device, 1, &fence) == VK_ERROR_DEVICE_LOST);
    assert(capture.resetCalls == 1
        && lsfgvk::layer::test::containsPresentedLease(anyA));
    capture.resetResult = VK_SUCCESS;
    assert(reset(device, 1, &fence) == VK_SUCCESS);
    assert(capture.resetCalls == 2
        && !lsfgvk::layer::test::containsPresentedLease(anyA));

    destroy(device, secondFence, nullptr);
    assert(capture.destroyCalls == 1 && capture.lastFence == secondFence
        && !lsfgvk::layer::test::containsPresentedLease(anyB));

    // Reuse the same raw handle after destroy; the old generation is closed.
    destroy(device, fence, nullptr);
    const auto destroysBeforeReuse = capture.destroyCalls;
    capture.returnedFence = fence;
    VkFence reused{};
    assert(create(device, &createInfo, nullptr, &reused) == VK_SUCCESS
        && reused == fence && capture.destroyCalls == destroysBeforeReuse);
    const auto reusedIdentity = identity(96, 5);
    assert(lsfgvk::layer::test::installPresentedLease(
        reusedIdentity, std::make_shared<int>(6)));
    assert(lsfgvk::layer::test::associateBorrowedFence(reused, reusedIdentity));
    capture.statusResult = VK_SUCCESS;
    assert(status(device, reused) == VK_SUCCESS
        && !lsfgvk::layer::test::containsPresentedLease(reusedIdentity));

    // 4B1C: logical Acquire completion is projected to one layer-owned proxy
    // without modifying the application's permanent fence payload.
    const auto proxy = reinterpret_cast<VkFence>(uintptr_t{0x72F0});
    assert(lsfgvk::layer::test::associateAcquireFenceProxy(
        reused, proxy, std::make_shared<int>(7)));
    capture.signaledFence = proxy;
    assert(status(device, reused) == VK_SUCCESS && capture.lastFence == proxy);
    const VkFence mixedProxy[]{secondFence, reused};
    const auto waitsBeforeProjectionOom = capture.waitCalls;
    failGlobalAllocation.store(true, std::memory_order_relaxed);
    const auto projectionOom = wait(
        device, 2, mixedProxy, VK_FALSE, 455);
    failGlobalAllocation.store(false, std::memory_order_relaxed);
    assert(projectionOom == VK_ERROR_OUT_OF_HOST_MEMORY
        && capture.waitCalls == waitsBeforeProjectionOom
        && lsfgvk::layer::test::acquireFenceProxyAssociationCount() == 1);
    assert(wait(device, 2, mixedProxy, VK_FALSE, 456) == VK_SUCCESS);
    assert(capture.waits == std::vector<VkFence>({secondFence, proxy})
        && capture.waitAll == VK_FALSE && capture.timeout == 456);
    capture.resetResult = VK_ERROR_DEVICE_LOST;
    assert(reset(device, 1, &reused) == VK_ERROR_DEVICE_LOST
        && lsfgvk::layer::test::acquireFenceProxyAssociationCount() == 1);
    capture.resetResult = VK_SUCCESS;
    assert(reset(device, 1, &reused) == VK_SUCCESS && capture.lastFence == reused
        && lsfgvk::layer::test::acquireFenceProxyAssociationCount() == 0);
    assert(lsfgvk::layer::test::associateAcquireFenceProxy(
        reused, proxy, std::make_shared<int>(8)));
    const auto importLease = identity(97, 6);
    assert(lsfgvk::layer::test::installPresentedLease(
        importLease, std::make_shared<int>(9)));
    assert(lsfgvk::layer::test::associateBorrowedFence(reused, importLease));
    const VkImportFenceFdInfoKHR importInfo{
        .sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
        .fence = reused,
        .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = -1};
    capture.importResult = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    assert(importFd(device, &importInfo) == VK_ERROR_INVALID_EXTERNAL_HANDLE
        && lsfgvk::layer::test::acquireFenceProxyAssociationCount() == 1
        && lsfgvk::layer::test::containsPresentedLease(importLease));
    capture.importResult = VK_SUCCESS;
    assert(importFd(device, &importInfo) == VK_SUCCESS
        && lsfgvk::layer::test::acquireFenceProxyAssociationCount() == 0
        && !lsfgvk::layer::test::containsPresentedLease(importLease)
        && lsfgvk::layer::test::borrowedAssociationCount() == 0);
    destroy(device, reused, nullptr);
    assert(capture.destroyCalls == destroysBeforeReuse + 1);
    lsfgvk::layer::test::uninstallDeviceEntrypointHarness();

    // Actual exported-layer implementation: concrete vk::Vulkan, backend,
    // Root and Swapchain state installed in the same production registries.
    {
        lsfgvk::test::ShadowBReturnExecutionHarness presentHarness;
        auto backend = presentHarness.createB0B3Instance();
        auto vulkan = presentHarness.createQueuePresentVulkan();
        const auto presentQueue = presentHarness.endpoint().queue;
        const auto presentDevice = presentHarness.queuePresentDevice();
        const auto presentSwapchain = reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8100});
        SwapchainInfo swapchainInfo{};
        swapchainInfo.images = {reinterpret_cast<VkImage>(uintptr_t{0x8200})};
        swapchainInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        swapchainInfo.extent = {1280, 720};
        swapchainInfo.arrayLayers = 2;
        swapchainInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        swapchainInfo.adaptivePresentMode = VK_PRESENT_MODE_FIFO_KHR;
        swapchainInfo.d2Foundation = true;
        swapchainInfo.d2State = std::make_shared<D2RealWsiState>(3);
        const auto d2State = swapchainInfo.d2State;
        assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
            .device = presentDevice,
            .queue = presentQueue,
            .queueFamily = presentHarness.endpoint().queueFamilyIndex,
            .swapchain = presentSwapchain,
            .vulkan = vulkan.get(),
            .backend = backend.get(),
            .devicePair = presentHarness.sameDevicePair(),
            .swapchainInfo = swapchainInfo,
            .deviceLifetimeIdentity = 101,
            .frameGenerationMultiplier = 2}));
        // Ownership moved into the real InstanceInfo device map.
        vulkan.reset();
        const auto present = reinterpret_cast<PFN_vkQueuePresentKHR>(
            lsfgvk::layer::test::queuePresentEntrypoint());
        const auto d2Acquire = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
            lsfgvk::layer::test::queuePresentAcquireEntrypoint());
        assert(present && d2Acquire);
        const VkSemaphore waits[]{
            reinterpret_cast<VkSemaphore>(uintptr_t{0x8300}),
            reinterpret_cast<VkSemaphore>(uintptr_t{0x8301})};
        const VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        const VkSwapchainPresentModeInfoKHR modeInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
            .swapchainCount = 1,
            .pPresentModes = &mode};
        uint32_t imageIndex = UINT32_MAX;
        const auto submitsBeforeD2Acquire =
            presentHarness.observedApplicationSubmitCount();
        assert(d2Acquire(presentDevice, presentSwapchain, 123456,
            waits[0], VK_NULL_HANDLE, &imageIndex) == VK_SUCCESS);
        assert(imageIndex == 0 && presentHarness.observedApplicationSubmitCount()
            == submitsBeforeD2Acquire);
        assert(presentHarness.observedAcquireTimeout() == 123456
            && presentHarness.observedAcquireSemaphore() == waits[0]
            && presentHarness.observedAcquireFence() == VK_NULL_HANDLE);
        VkResult perSwapchain = VK_NOT_READY;
        const VkPresentInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = &modeInfo,
            .waitSemaphoreCount = 2,
            .pWaitSemaphores = waits,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &imageIndex,
            .pResults = &perSwapchain};
        assert(present(presentQueue, &info) == VK_SUCCESS);
        assert(perSwapchain == VK_SUCCESS);
        assert(presentHarness.observedQueuePresentCalls() == 1);
        assert(presentHarness.observedPresentedQueue() == presentQueue);
        assert(presentHarness.observedPresentedSwapchain() == presentSwapchain);
        assert(presentHarness.observedPresentedImageIndex() == imageIndex);
        assert(presentHarness.observedPresentedWaits()
            == std::vector<VkSemaphore>(std::begin(waits), std::end(waits)));
        assert(presentHarness.observedPresentedNext() == &modeInfo);
        assert(presentHarness.observedApplicationSubmitCount()
            == submitsBeforeD2Acquire);
        assert(!lsfgvk::layer::test::productionPresentedLeaseInstalled());

        // G-D2-1/2/3: KHR and EXT share one prevalidated, atomic state
        // transaction and forward the unchanged physical-index batch.
        const auto releaseKhr = reinterpret_cast<PFN_vkReleaseSwapchainImagesKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkReleaseSwapchainImagesKHR"));
        const auto releaseExt = reinterpret_cast<PFN_vkReleaseSwapchainImagesEXT>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkReleaseSwapchainImagesEXT"));
        assert(releaseKhr && releaseExt);
        assert(d2Acquire(presentDevice, presentSwapchain, 1, waits[0],
            VK_NULL_HANDLE, &imageIndex) == VK_SUCCESS);
        assert(d2State->acquired(2));
        const uint32_t releaseIndices[]{0, 2};
        const VkReleaseSwapchainImagesInfoKHR releaseInfo{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain,
            .imageIndexCount = 2,
            .pImageIndices = releaseIndices};
        presentHarness.setReleaseSwapchainImagesResult(VK_ERROR_SURFACE_LOST_KHR);
        assert(releaseKhr(presentDevice, &releaseInfo) == VK_ERROR_SURFACE_LOST_KHR);
        assert(d2State->state(0)->appAcquired && d2State->state(2)->appAcquired);
        presentHarness.setReleaseSwapchainImagesResult(VK_SUCCESS);
        assert(releaseKhr(presentDevice, &releaseInfo) == VK_SUCCESS);
        assert(presentHarness.observedReleasedIndices()
            == std::vector<uint32_t>(std::begin(releaseIndices),
                std::end(releaseIndices)));
        assert(!d2State->state(0)->appAcquired
            && !d2State->state(2)->appAcquired);
        assert(d2State->acquired(1));
        const uint32_t extIndex = 1;
        const VkReleaseSwapchainImagesInfoKHR extReleaseInfo{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain,
            .imageIndexCount = 1,
            .pImageIndices = &extIndex};
        assert(releaseExt(presentDevice, &extReleaseInfo) == VK_SUCCESS);
        assert(d2State->state(1)->lastApplicationExit
            == D2ApplicationExit::Release);

        // 4B1C native fast path remains exact with a D2 runtime installed and
        // no hidden-clobbered image: fence-only and both payloads are passed
        // through, with no layer QueueSubmit.
        const auto nativeFence = reinterpret_cast<VkFence>(uintptr_t{0x830F});
        const auto nativeSubmits = presentHarness.observedApplicationSubmitCount();
        presentHarness.setAcquiredImageIndex(0);
        assert(d2Acquire(presentDevice, presentSwapchain, 55,
            VK_NULL_HANDLE, nativeFence, &imageIndex) == VK_SUCCESS);
        assert(presentHarness.observedAcquireSemaphore() == VK_NULL_HANDLE
            && presentHarness.observedAcquireFence() == nativeFence
            && presentHarness.observedApplicationSubmitCount() == nativeSubmits);
        const uint32_t nativeFenceIndex = imageIndex;
        const VkReleaseSwapchainImagesInfoKHR nativeFenceRelease{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain, .imageIndexCount = 1,
            .pImageIndices = &nativeFenceIndex};
        assert(releaseKhr(presentDevice, &nativeFenceRelease) == VK_SUCCESS);
        presentHarness.setAcquiredImageIndex(1);
        assert(d2Acquire(presentDevice, presentSwapchain, 56,
            waits[0], nativeFence, &imageIndex) == VK_SUCCESS);
        assert(presentHarness.observedAcquireSemaphore() == waits[0]
            && presentHarness.observedAcquireFence() == nativeFence
            && presentHarness.observedApplicationSubmitCount() == nativeSubmits);
        const uint32_t nativeBothIndex = imageIndex;
        const VkReleaseSwapchainImagesInfoKHR nativeBothRelease{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain, .imageIndexCount = 1,
            .pImageIndices = &nativeBothIndex};
        assert(releaseKhr(presentDevice, &nativeBothRelease) == VK_SUCCESS);

        // A-D2-3: Acquire2 is forwarded as one immutable structure, including
        // the device mask and application synchronization payload.
        const auto d2Acquire2 = reinterpret_cast<PFN_vkAcquireNextImage2KHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkAcquireNextImage2KHR"));
        assert(d2Acquire2);
        const auto acquire2Fence =
            reinterpret_cast<VkFence>(uintptr_t{0x8310});
        const VkAcquireNextImageInfoKHR acquire2Info{
            .sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
            .swapchain = presentSwapchain,
            .timeout = 7654321,
            .semaphore = waits[1],
            .fence = acquire2Fence,
            .deviceMask = 0x5};
        presentHarness.setAcquiredImageIndex(0);
        imageIndex = UINT32_MAX;
        assert(d2Acquire2(presentDevice, &acquire2Info, &imageIndex) == VK_SUCCESS
            && imageIndex == 0);
        const auto& observedAcquire2 = presentHarness.observedAcquire2();
        assert(observedAcquire2.swapchain == presentSwapchain
            && observedAcquire2.timeout == acquire2Info.timeout
            && observedAcquire2.semaphore == acquire2Info.semaphore
            && observedAcquire2.fence == acquire2Info.fence
            && observedAcquire2.deviceMask == acquire2Info.deviceMask);
        uint32_t duplicateAcquireIndex = UINT32_MAX;
        assert(d2Acquire(presentDevice, presentSwapchain, 1, waits[0],
            VK_NULL_HANDLE, &duplicateAcquireIndex) == VK_ERROR_DEVICE_LOST);

        VkResult stalePerResult = VK_SUCCESS;
        const VkPresentInfoKHR preEnqueuePresent{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &imageIndex,
            .pResults = &stalePerResult};
        presentHarness.setQueuePresentResult(VK_ERROR_OUT_OF_HOST_MEMORY);
        assert(present(presentQueue, &preEnqueuePresent)
            == VK_ERROR_OUT_OF_HOST_MEMORY);
        assert(stalePerResult == VK_SUCCESS && d2State->state(0)->appAcquired);
        presentHarness.setQueuePresentResult(VK_SUCCESS);

        const uint32_t invalidReleaseIndex = 2;
        const VkReleaseSwapchainImagesInfoKHR invalidRelease{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain,
            .imageIndexCount = 1,
            .pImageIndices = &invalidReleaseIndex};
        const auto releasesBeforeInvalid = presentHarness.observedReleaseCalls();
        assert(releaseKhr(presentDevice, &invalidRelease) == VK_ERROR_UNKNOWN
            && presentHarness.observedReleaseCalls() == releasesBeforeInvalid);

        // Preserve the queue-metadata-absent coverage before retiring A.
        presentHarness.setAcquiredImageIndex(2);
        uint32_t unknownFamilyIndex = UINT32_MAX;
        assert(d2Acquire(presentDevice, presentSwapchain, 1, waits[0],
            VK_NULL_HANDLE, &unknownFamilyIndex) == VK_SUCCESS
            && unknownFamilyIndex == 2);
        lsfgvk::layer::test::removeQueueMetadata(presentQueue);
        const VkPresentInfoKHR unknownFamilyPresent{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &unknownFamilyIndex};
        assert(present(presentQueue, &unknownFamilyPresent) == VK_SUCCESS);
        assert(d2State->state(2)->lastApplicationExit
                == D2ApplicationExit::Present
            && !d2State->state(2)->lastApplicationPresentFamily.has_value());

        // Acquire both old-swapchain images while A is still valid. They are
        // the only authorities exercised after Create B retires A.
        presentHarness.setAcquiredImageIndex(1);
        uint32_t heldReleaseIndex = UINT32_MAX;
        assert(d2Acquire(presentDevice, presentSwapchain, 1, waits[0],
            VK_NULL_HANDLE, &heldReleaseIndex) == VK_SUCCESS
            && heldReleaseIndex == 1);

        // CREATE-D2-1..5: exercise the real public create entrypoint. The
        // downstream sees the original contract, exposes its real images, and
        // no additional Root/backend context is created even at requested 2x.
        const auto createSwapchain = reinterpret_cast<PFN_vkCreateSwapchainKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkCreateSwapchainKHR"));
        const auto getSwapchainImages = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkGetSwapchainImagesKHR"));
        const auto d2DestroySwapchain = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkDestroySwapchainKHR"));
        assert(createSwapchain && getSwapchainImages && d2DestroySwapchain);
        const uint32_t createFamilies[]{3, 7};
        const VkFormat createViewFormat = VK_FORMAT_B8G8R8A8_SRGB;
        const VkImageFormatListCreateInfo createNext{
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .viewFormatCount = 1,
            .pViewFormats = &createViewFormat};
        const VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = &createNext,
            .flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR,
            .surface = reinterpret_cast<VkSurfaceKHR>(uintptr_t{0x8820}),
            .minImageCount = 3,
            .imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
            .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            .imageExtent = {1920, 1080},
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_CONCURRENT,
            .queueFamilyIndexCount = 2,
            .pQueueFamilyIndices = createFamilies,
            .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_MAILBOX_KHR,
            .clipped = VK_TRUE,
            .oldSwapchain = presentSwapchain};
        const auto contextsBeforeCreate =
            lsfgvk::layer::test::rootSwapchainContextCount();
        VkSwapchainKHR created{};
        assert(createSwapchain(presentDevice, &createInfo, nullptr, &created)
            == VK_SUCCESS && created != VK_NULL_HANDLE);
        const auto& observedCreate = presentHarness.observedSwapchainCreateInfo();
        assert(observedCreate.pNext == createInfo.pNext
            && observedCreate.flags == createInfo.flags
            && observedCreate.surface == createInfo.surface
            && observedCreate.minImageCount == createInfo.minImageCount
            && observedCreate.imageFormat == createInfo.imageFormat
            && observedCreate.imageColorSpace == createInfo.imageColorSpace
            && observedCreate.imageExtent.width == createInfo.imageExtent.width
            && observedCreate.imageExtent.height == createInfo.imageExtent.height
            && observedCreate.imageArrayLayers == createInfo.imageArrayLayers
            && observedCreate.imageUsage == createInfo.imageUsage
            && observedCreate.imageSharingMode == createInfo.imageSharingMode
            && observedCreate.preTransform == createInfo.preTransform
            && observedCreate.compositeAlpha == createInfo.compositeAlpha
            && observedCreate.presentMode == createInfo.presentMode
            && observedCreate.clipped == createInfo.clipped
            && observedCreate.oldSwapchain == createInfo.oldSwapchain);
        assert(presentHarness.observedCreateQueueFamilies()
            == std::vector<uint32_t>(std::begin(createFamilies),
                std::end(createFamilies)));
        assert(lsfgvk::layer::test::rootSwapchainContextCount()
            == contextsBeforeCreate);
        assert(d2State->retiredForLsfgWork());

        // N-D2-1/2: the old swapchain was retired by successful creation, but
        // its already-held physical image remains a genuine downstream WSI
        // present/release authority.
        VkResult heldPresentResult = VK_NOT_READY;
        const VkPresentInfoKHR heldPresent{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &imageIndex,
            .pResults = &heldPresentResult};
        assert(present(presentQueue, &heldPresent) == VK_SUCCESS
            && heldPresentResult == VK_SUCCESS
            && presentHarness.observedPresentedSwapchain() == presentSwapchain
            && presentHarness.observedPresentedImageIndex() == 0
            && d2State->state(0)->lastApplicationExit
                == D2ApplicationExit::Present);
        const VkReleaseSwapchainImagesInfoKHR heldRelease{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = presentSwapchain,
            .imageIndexCount = 1,
            .pImageIndices = &heldReleaseIndex};
        assert(releaseKhr(presentDevice, &heldRelease) == VK_SUCCESS
            && presentHarness.observedReleasedIndices()
                == std::vector<uint32_t>{1}
            && d2State->state(1)->lastApplicationExit
                == D2ApplicationExit::Release);
        uint32_t realCount{};
        assert(getSwapchainImages(presentDevice, created, &realCount, nullptr)
            == VK_SUCCESS && realCount == 3);
        std::array<VkImage, 3> realImages{};
        assert(getSwapchainImages(presentDevice, created, &realCount,
            realImages.data()) == VK_SUCCESS);
        assert(realImages[0] == reinterpret_cast<VkImage>(uintptr_t{0x8810})
            && realImages[1] == reinterpret_cast<VkImage>(uintptr_t{0x8811})
            && realImages[2] == reinterpret_cast<VkImage>(uintptr_t{0x8812}));
        const auto destroysBeforeNoalloc =
            presentHarness.observedDestroySwapchainCalls();
        failGlobalAllocation.store(true, std::memory_order_relaxed);
        d2DestroySwapchain(presentDevice, created, nullptr);
        failGlobalAllocation.store(false, std::memory_order_relaxed);
        assert(presentHarness.observedDestroySwapchainCalls()
            == destroysBeforeNoalloc + 1);

        AllocatorProbe cleanupAllocatorProbe{};
        cleanupAllocatorProbe.callingThread = std::this_thread::get_id();
        cleanupAllocatorProbe.insidePublicDestroy = true;
        const VkAllocationCallbacks cleanupAllocator{
            .pUserData = &cleanupAllocatorProbe,
            .pfnAllocation = probeAllocate,
            .pfnReallocation = probeReallocate,
            .pfnFree = probeFree};
        const auto destroysBeforeFailedCreate =
            presentHarness.observedDestroySwapchainCalls();
        lsfgvk::layer::test::setD2CreateBookkeepingFailure(true);
        created = VK_NULL_HANDLE;
        assert(createSwapchain(presentDevice, &createInfo, &cleanupAllocator,
            &created) == VK_ERROR_INITIALIZATION_FAILED);
        assert(created == VK_NULL_HANDLE
            && presentHarness.observedDestroySwapchainCalls()
                == destroysBeforeFailedCreate + 1
            && cleanupAllocatorProbe.callbacks == 1
            && !lsfgvk::layer::test::trackedSwapchain(
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8800}))
            && lsfgvk::layer::test::rootSwapchainContextCount()
                == contextsBeforeCreate);
        lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();

        auto virtualVulkan = presentHarness.createQueuePresentVulkan();
        SwapchainInfo virtualInfo = swapchainInfo;
        virtualInfo.d2Foundation = false;
        virtualInfo.d2State.reset();
        virtualInfo.arrayLayers = 1;
        virtualInfo.realImages = {
            reinterpret_cast<VkImage>(uintptr_t{0x8400}),
            reinterpret_cast<VkImage>(uintptr_t{0x8401})};
        virtualInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        virtualInfo.surfacePresentFamilies = {
            presentHarness.endpoint().queueFamilyIndex};
        virtualInfo.releaseBackend = SwapchainReleaseBackend::Khr;
        assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
            .device = presentDevice,
            .queue = presentQueue,
            .queueFamily = presentHarness.endpoint().queueFamilyIndex,
            .swapchain = presentSwapchain,
            .vulkan = virtualVulkan.get(),
            .backend = backend.get(),
            .devicePair = presentHarness.sameDevicePair(),
            .swapchainInfo = virtualInfo,
            .deviceLifetimeIdentity = 102,
            .virtualized = true}));
        virtualVulkan.reset();
        const auto acquire = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
            lsfgvk::layer::test::queuePresentAcquireEntrypoint());
        const auto virtualPresent = reinterpret_cast<PFN_vkQueuePresentKHR>(
            lsfgvk::layer::test::queuePresentEntrypoint());
        assert(acquire && virtualPresent);
        const auto queueCreateFence = reinterpret_cast<PFN_vkCreateFence>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint("vkCreateFence"));
        const auto queueFenceStatus = reinterpret_cast<PFN_vkGetFenceStatus>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint("vkGetFenceStatus"));
        assert(queueCreateFence && queueFenceStatus);
        presentHarness.enableAdaptiveProducerFenceModel();
        const VkFenceCreateInfo applicationFenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence applicationFence{};
        assert(queueCreateFence(presentDevice, &applicationFenceInfo, nullptr,
            &applicationFence) == VK_SUCCESS);
        presentHarness.observeApplicationFence(applicationFence);

        // Actual-builder B matrix. Every injected failure happens before the
        // reservation is returned and therefore before any physical Acquire or
        // queue side effect. B-15 runs after both registry reservations exist.
        const auto verifyBuilderFailure = [&](auto inject) {
            const auto acquireBefore = presentHarness.observedAcquireCalls();
            const auto submitsBeforeFailure =
                presentHarness.observedApplicationSubmitCount();
            const auto presentsBeforeFailure =
                presentHarness.observedQueuePresentCalls();
            const auto waitsBeforeFailure = presentHarness.frontFenceWaitCount();
            const auto idleBeforeFailure =
                presentHarness.observedDeviceWaitIdleCalls();
            const auto leaseBaseline =
                lsfgvk::layer::test::installedPresentedLeaseCount();
            const auto borrowedBaseline =
                lsfgvk::layer::test::installedBorrowedAssociationCount();
            inject();
            bool failed{};
            try {
                auto unused = lsfgvk::layer::test::
                    reserveInstalledAdaptive1xForTesting(applicationFence);
                (void)unused;
            } catch (...) { failed = true; }
            assert(failed);
            assert(presentHarness.observedAcquireCalls() == acquireBefore);
            assert(presentHarness.observedApplicationSubmitCount()
                == submitsBeforeFailure);
            assert(presentHarness.observedQueuePresentCalls()
                == presentsBeforeFailure);
            assert(presentHarness.frontFenceWaitCount() == waitsBeforeFailure);
            assert(presentHarness.observedDeviceWaitIdleCalls()
                == idleBeforeFailure);
            assert(lsfgvk::layer::test::installedPresentedLeaseCount()
                == leaseBaseline);
            assert(lsfgvk::layer::test::installedBorrowedAssociationCount()
                == borrowedBaseline);
        };
        using BuilderFailure =
            lsfgvk::layer::test::AdaptiveBuilderFailurePoint;
        for (const auto point : {BuilderFailure::AcquireBacking,
                BuilderFailure::ProducerBacking, BuilderFailure::PresentBacking,
                BuilderFailure::CompositeRecoveryBacking,
                BuilderFailure::ConsumedAcquireStorage,
                BuilderFailure::ReactorPendingOperation,
                BuilderFailure::PendingHolder, BuilderFailure::PreparedCallbacks,
                BuilderFailure::RecoveryCallbacks,
                BuilderFailure::PhysicalLeaseReserve,
                BuilderFailure::BorrowedFenceReserve,
                BuilderFailure::AfterBothRegistryReservations}) {
            verifyBuilderFailure([&] {
                lsfgvk::layer::test::setAdaptiveBuilderFailurePoint(point);
            });
        }
        lsfgvk::layer::test::resetAdaptiveCommandPoolLifecycleCounts();
        lsfgvk::layer::test::setAdaptiveBuilderFailurePoint(
            BuilderFailure::None);
        verifyBuilderFailure([&] { presentHarness.setShadowFailurePoint(
            lsfgvk::test::ShadowHarnessFailurePoint::COMMAND_POOL_CREATE); });
        assert(lsfgvk::layer::test::adaptiveCommandPoolCreateCount() == 0);
        assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 0);
        presentHarness.setShadowFailurePoint(
            lsfgvk::test::ShadowHarnessFailurePoint::NONE);
        lsfgvk::layer::test::resetAdaptiveCommandPoolLifecycleCounts();
        lsfgvk::layer::test::setAdaptiveBuilderFailurePoint(
            BuilderFailure::None);
        verifyBuilderFailure([&] { presentHarness.setShadowFailurePoint(
            lsfgvk::test::ShadowHarnessFailurePoint::COMMAND_BUFFER_ALLOCATION); });
        assert(lsfgvk::layer::test::adaptiveCommandPoolCreateCount() == 1);
        assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 1);
        presentHarness.setShadowFailurePoint(
            lsfgvk::test::ShadowHarnessFailurePoint::NONE);

        // AR-1/AR-2: the real builder returns a valid reservation, and dropping
        // it before execute unwinds both host reservations without GPU work.
        {
            const auto acquireBefore = presentHarness.observedAcquireCalls();
            const auto leaseBaseline =
                lsfgvk::layer::test::installedPresentedLeaseCount();
            const auto borrowedBaseline =
                lsfgvk::layer::test::installedBorrowedAssociationCount();
            auto unused = lsfgvk::layer::test::
                reserveInstalledAdaptive1xForTesting(applicationFence);
            assert(unused.valid());
            assert(lsfgvk::layer::test::installedPresentedLeaseCount()
                == leaseBaseline + 1);
            assert(lsfgvk::layer::test::installedBorrowedAssociationCount()
                == borrowedBaseline + 1);
            (void)acquireBefore;
        }
        assert(lsfgvk::layer::test::installedPresentedLeaseCount() == 0);
        assert(lsfgvk::layer::test::installedBorrowedAssociationCount() == 0);

        // AR-6: maintenance-release backend None is rejected by the actual
        // builder before physical Acquire or any registry reservation.
        {
            const auto acquireBefore = presentHarness.observedAcquireCalls();
            const auto previous = lsfgvk::layer::test::
                setInstalledAdaptiveReleaseBackendForTesting(
                    SwapchainReleaseBackend::None);
            assert(previous != SwapchainReleaseBackend::None);
            try {
                static_cast<void>(lsfgvk::layer::test::
                    reserveInstalledAdaptive1xForTesting(applicationFence));
                assert(false);
            } catch (const ls::vulkan_error& error) {
                assert(error.error() == VK_ERROR_FEATURE_NOT_PRESENT);
            }
            assert(presentHarness.observedAcquireCalls() == acquireBefore);
            assert(lsfgvk::layer::test::installedPresentedLeaseCount() == 0);
            assert(lsfgvk::layer::test::installedBorrowedAssociationCount() == 0);
            assert(lsfgvk::layer::test::setInstalledAdaptiveReleaseBackendForTesting(
                previous) == SwapchainReleaseBackend::None);
        }

        // Actual-builder AR-3/4/5/13/14 and CP-4. Move transfers the host
        // reservations, the moved-from and already-executed objects reject
        // execution, and explicit abandon retires the generation exactly once.
        const auto cp4FreeBefore = presentHarness.commandBufferFreeCalls();
        {
            lsfgvk::layer::test::resetAdaptiveCommandPoolLifecycleCounts();
            const auto waitsBefore = presentHarness.frontFenceWaitCount();
            const auto releasesBefore = presentHarness.observedReleaseCalls();
            const auto presentsBefore = presentHarness.observedQueuePresentCalls();
            auto source = lsfgvk::layer::test::
                reserveInstalledAdaptive1xForTesting();
            Adaptive1xPreparationReservation moved(std::move(source));
            assert(!source.valid() && moved.valid());
            try {
                static_cast<void>(source.execute(
                    reinterpret_cast<VkSemaphore>(uintptr_t{0x84f0})));
                assert(false);
            } catch (const ls::vulkan_error&) {}
            auto prepared = moved.execute(
                reinterpret_cast<VkSemaphore>(uintptr_t{0x84f1}));
            assert(prepared.valid());
            assert(moved.state()
                == Adaptive1xPreparationReservation::State::PreparedTransferred);
            try {
                static_cast<void>(moved.execute(
                    reinterpret_cast<VkSemaphore>(uintptr_t{0x84f2})));
                assert(false);
            } catch (const ls::vulkan_error&) {}
            assert(presentHarness.observedQueuePresentCalls() == presentsBefore);
            assert(lsfgvk::layer::test::adaptiveCommandPoolCreateCount() == 1);
            assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 0);
            assert(presentHarness.commandBufferFreeCalls() == cp4FreeBefore);
            assert(prepared.abandonWithoutLogicalPresent() == VK_SUCCESS);
            assert(presentHarness.frontFenceWaitCount() == waitsBefore + 1);
            assert(presentHarness.observedReleaseCalls() == releasesBefore + 1);
            assert(!prepared.valid());
        }
        assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 1);
        assert(presentHarness.commandBufferFreeCalls() == cp4FreeBefore + 1);
        assert(lsfgvk::layer::test::installedPresentedLeaseCount() == 0);

        // Actual-builder AR-7..AR-12. These cases share one fixture but assert
        // the precise Acquire/producer boundary and recovery terminal.
        const auto runExecuteFailure = [&](VkFence fence, auto configure,
                Adaptive1xPreparationReservation::State expectedState,
                uint32_t expectedAcquireDelta, size_t expectedSubmitDelta,
                uint32_t expectedWaitDelta, uint32_t expectedReleaseDelta) {
            const auto acquiresBefore = presentHarness.observedAcquireCalls();
            const auto submitsBeforeFailure = presentHarness.frontSubmitCount();
            const auto waitsBeforeFailure = presentHarness.frontFenceWaitCount();
            const auto releasesBeforeFailure = presentHarness.observedReleaseCalls();
            const auto presentsBeforeFailure =
                presentHarness.observedQueuePresentCalls();
            configure();
            auto reservation = lsfgvk::layer::test::
                reserveInstalledAdaptive1xForTesting(fence);
            try {
                static_cast<void>(reservation.execute(
                    reinterpret_cast<VkSemaphore>(uintptr_t{0x84f3})));
                assert(false);
            } catch (const ls::vulkan_error&) {}
            assert(reservation.state() == expectedState);
            assert(presentHarness.observedAcquireCalls()
                == acquiresBefore + expectedAcquireDelta);
            assert(presentHarness.frontSubmitCount()
                == submitsBeforeFailure + expectedSubmitDelta);
            assert(presentHarness.frontFenceWaitCount()
                == waitsBeforeFailure + expectedWaitDelta);
            assert(presentHarness.observedReleaseCalls()
                == releasesBeforeFailure + expectedReleaseDelta);
            assert(presentHarness.observedQueuePresentCalls()
                == presentsBeforeFailure);
            presentHarness.setAcquireNextImageResult(VK_SUCCESS);
            presentHarness.setQueueSubmitResult(VK_SUCCESS);
            presentHarness.setShadowFailurePoint(
                lsfgvk::test::ShadowHarnessFailurePoint::NONE);
            lsfgvk::layer::test::setAdaptiveExecuteFailurePoint(
                lsfgvk::layer::test::AdaptiveExecuteFailurePoint::None);
            assert(lsfgvk::layer::test::installedPresentedLeaseCount() == 0);
            assert(lsfgvk::layer::test::installedBorrowedAssociationCount() == 0);
        };
        runExecuteFailure(VK_NULL_HANDLE, [&] {
            presentHarness.setAcquireNextImageResult(VK_ERROR_OUT_OF_DATE_KHR);
        }, Adaptive1xPreparationReservation::State::CleanlyAborted,
            1, 0, 0, 0);
        runExecuteFailure(VK_NULL_HANDLE, [&] {
            presentHarness.setShadowFailurePoint(
                lsfgvk::test::ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER);
        }, Adaptive1xPreparationReservation::State::Recovered,
            1, 0, 1, 1);
        runExecuteFailure(applicationFence, [&] {
            lsfgvk::layer::test::setAdaptiveExecuteFailurePoint(
                lsfgvk::layer::test::AdaptiveExecuteFailurePoint::PhysicalLeaseBind);
        }, Adaptive1xPreparationReservation::State::Recovered,
            1, 0, 1, 1);
        runExecuteFailure(applicationFence, [&] {
            lsfgvk::layer::test::setAdaptiveExecuteFailurePoint(
                lsfgvk::layer::test::AdaptiveExecuteFailurePoint::BorrowedFenceBind);
        }, Adaptive1xPreparationReservation::State::Recovered,
            1, 0, 1, 1);
        runExecuteFailure(VK_NULL_HANDLE, [&] {
            presentHarness.setQueueSubmitResult(VK_ERROR_OUT_OF_HOST_MEMORY);
        }, Adaptive1xPreparationReservation::State::Recovered,
            1, 1, 1, 1);
        runExecuteFailure(VK_NULL_HANDLE, [&] {
            presentHarness.setQueueSubmitResult(VK_ERROR_OUT_OF_DEVICE_MEMORY);
        }, Adaptive1xPreparationReservation::State::Recovered,
            1, 1, 1, 1);
        uint32_t virtualIndex{};
        const auto acquireReady = reinterpret_cast<VkSemaphore>(uintptr_t{0x8500});
        presentHarness.setAcquiredImageIndex(0);
        assert(acquire(presentDevice, presentSwapchain, 0, acquireReady,
            VK_NULL_HANDLE, &virtualIndex) == VK_SUCCESS);
        const auto submitsBefore = presentHarness.observedApplicationSubmitCount();
        const VkSwapchainPresentFenceInfoKHR presentFenceInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
            .swapchainCount = 1,
            .pFences = &applicationFence};
        const VkPresentInfoKHR virtualPresentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = &presentFenceInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &acquireReady,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &virtualIndex};
        const auto presentCallsBeforeVirtual =
            presentHarness.observedQueuePresentCalls();
        // The controlled QueueSubmit observer itself records two submits for
        // this logical present (wait bridge + producer). Reserve that test-only
        // storage before AR-17 disables allocation so harness bookkeeping cannot
        // masquerade as a production allocation.
        presentHarness.reserveFrontSubmitCapacity(submitsBefore + 2);
        // AR-17: reservation has completed every host allocation. Keep global
        // allocation disabled across the complete real execute interval, from
        // Acquire through PreparedLogicalFinal return.
        lsfgvk::layer::test::setAdaptiveReservationExecuteHooks(
            failAdaptiveExecuteAllocations, allowAdaptiveExecuteAllocations);
        lsfgvk::layer::test::resetAdaptiveCommandPoolLifecycleCounts();
        const auto cp5FreeBefore = presentHarness.commandBufferFreeCalls();
        const auto virtualResult = virtualPresent(presentQueue, &virtualPresentInfo);
        lsfgvk::layer::test::setAdaptiveReservationExecuteHooks(nullptr, nullptr);
        assert(!failGlobalAllocation.load(std::memory_order_relaxed));
        if (virtualResult != VK_SUCCESS)
            std::cerr << "virtual QueuePresent result=" << virtualResult << '\n';
        assert(virtualResult == VK_SUCCESS);
        assert(presentHarness.observedAcquireFence() != VK_NULL_HANDLE);
        const auto produced =
            lsfgvk::layer::test::lastProductionPresentedIdentity();
        assert(produced.deviceLifetimeIdentity != 0);
        assert(produced.swapchainLifecycleIdentity != 0);
        assert(produced.physicalSwapchainIdentity
            == reinterpret_cast<uintptr_t>(presentSwapchain));
        assert(produced.physicalImageIndex
            == presentHarness.observedPresentedImageIndex());
        assert(produced.presentOperationIdentity != 0);
        assert(lsfgvk::layer::test::productionPresentedLeaseInstalled());
        assert(presentHarness.observedApplicationSubmitCount()
            == submitsBefore + 2);
        assert(presentHarness.observedQueuePresentCalls()
            == presentCallsBeforeVirtual + 1);
        const auto firstProducerPool = presentHarness.latestCommandPool();
        const auto firstProducerCommand = presentHarness.latestCommandBuffer();
        assert(firstProducerPool != VK_NULL_HANDLE);
        assert(firstProducerCommand != VK_NULL_HANDLE);
        assert(presentHarness.observedPresentedSwapchain() == presentSwapchain);
        assert(presentHarness.observedPresentedImageIndex() == 0);
        assert(presentHarness.observedPresentedWaits().size() == 1);
        assert(presentHarness.frontSubmit(submitsBefore + 1).signal
            == presentHarness.observedPresentedWaits().front());
        assert(presentHarness.observedPresentedNext() == &presentFenceInfo);
        assert(presentHarness.observedPresentFenceStructCount() == 1);
        assert(presentHarness.observedPresentedFence() == applicationFence);
        presentHarness.setQueuePresentFenceStatus(VK_NOT_READY);
        assert(queueFenceStatus(presentDevice, applicationFence) == VK_NOT_READY);
        assert(lsfgvk::layer::test::productionPresentedLeaseInstalled());

        presentHarness.setQueuePresentFenceStatus(VK_SUCCESS);
        assert(queueFenceStatus(presentDevice, applicationFence) == VK_SUCCESS);
        assert(!lsfgvk::layer::test::productionPresentedLeaseInstalled());
        assert(lsfgvk::layer::test::adaptiveCommandPoolCreateCount() == 1);
        assert(lsfgvk::layer::test::waitForAdaptiveCommandPoolDestroyCount(
            1, 1000));
        assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 1);
        assert(presentHarness.commandBufferFreeCalls() == cp5FreeBefore + 1);
        assert(queueFenceStatus(presentDevice, applicationFence) == VK_SUCCESS);
        assert(presentHarness.applicationFenceExportCalls() == 0);
        assert(presentHarness.applicationFenceResetCalls() == 0);
        assert(presentHarness.applicationFenceWaitCalls() == 0);
        assert(presentHarness.applicationFenceDestroyCalls() == 0);

        // A1X-PG-1/2/3/12: SYNC_FD export copy-transfers the first producer
        // fence payload, so its source is no longer completion authority.  A
        // second generation must use a distinct producer fence and complete
        // without resetting, waiting for, or reusing the exported source.
        const auto firstProducerFence =
            presentHarness.frontSubmit(submitsBefore + 1).fence;
        assert(firstProducerFence != VK_NULL_HANDLE);
        assert(queueFenceStatus(presentDevice, firstProducerFence)
            == VK_NOT_READY);

        uint32_t secondVirtualIndex{};
        const auto secondAcquireReady =
            reinterpret_cast<VkSemaphore>(uintptr_t{0x8501});
        assert(acquire(presentDevice, presentSwapchain, 0, secondAcquireReady,
            VK_NULL_HANDLE, &secondVirtualIndex) == VK_SUCCESS);
        const auto secondSubmitsBefore = presentHarness.frontSubmitCount();
        const auto secondPresentsBefore =
            presentHarness.observedQueuePresentCalls();
        const auto secondReleasesBefore = presentHarness.observedReleaseCalls();
        const auto secondFenceWaitsBefore =
            presentHarness.frontFenceWaitCount();
        const auto secondFenceResetsBefore =
            presentHarness.controlledFenceResetCalls();
        VkResult internalFenceResult = VK_SUCCESS;
        const VkPresentInfoKHR internalFencePresent{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &secondAcquireReady,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &secondVirtualIndex,
            .pResults = &internalFenceResult};
        assert(virtualPresent(presentQueue, &internalFencePresent) == VK_SUCCESS);
        assert(internalFenceResult == VK_SUCCESS);
        assert(presentHarness.frontSubmitCount() == secondSubmitsBefore + 2);
        assert(presentHarness.latestCommandPool() != firstProducerPool);
        assert(presentHarness.latestCommandBuffer() != firstProducerCommand);
        assert(presentHarness.observedQueuePresentCalls()
            == secondPresentsBefore + 1);
        assert(presentHarness.observedReleaseCalls() == secondReleasesBefore);
        assert(presentHarness.frontFenceWaitCount() == secondFenceWaitsBefore);
        assert(presentHarness.controlledFenceResetCalls()
            == secondFenceResetsBefore);
        const auto secondProducerFence =
            presentHarness.frontSubmit(secondSubmitsBefore + 1).fence;
        assert(secondProducerFence != VK_NULL_HANDLE);
        assert(secondProducerFence != firstProducerFence);
        assert(queueFenceStatus(presentDevice, secondProducerFence)
            == VK_NOT_READY);

        // P4 real Adaptive composition: P1/P2/P3 are built before the accepted
        // bridge, then two production reservations execute allocation-free.
        {
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 3);
            const VkSwapchainKHR logicalSwapchains[]{
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x86e0}),
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x86e1})};
            const uint32_t logicalIndices[]{91, 92};
            const VkSemaphore applicationWaits[]{
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8700}),
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8701})};
            const VkFence applicationFences[]{applicationFence, VK_NULL_HANDLE};
            const VkSwapchainPresentFenceInfoKHR fenceInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .swapchainCount = 2, .pFences = applicationFences};
            const VkPresentInfoKHR applicationInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = &fenceInfo,
                .waitSemaphoreCount = 2,
                .pWaitSemaphores = applicationWaits,
                .swapchainCount = 2,
                .pSwapchains = logicalSwapchains,
                .pImageIndices = logicalIndices};
            auto transaction = BatchPresentTransaction::create(
                presentDevice, presentQueue, 0x8702, applicationInfo);
            assert(transaction);
            transaction->entry(0).path =
                BatchPresentTransaction::Path::VirtualPrepared;
            transaction->entry(1).path =
                BatchPresentTransaction::Path::VirtualPrepared;
            std::vector<PresentBatchPNextProjection::Path> paths(2,
                PresentBatchPNextProjection::Path::VirtualPrepared);
            auto projection = PresentBatchPNextProjection::build(
                applicationInfo.pNext, 2, paths);
            assert(projection.supported());

            uint32_t nextSignal = 0x8710;
            uint32_t bridgeSubmitCalls{};
            uint32_t bridgeHostWaits{};
            std::shared_ptr<void> asyncBridgeLifetime;
            auto bridge = BatchApplicationPresentBridgeAuthority::create(
                presentQueue, applicationWaits, 2, 2,
                std::make_shared<int>(1), {
                    .createSemaphore = [&]() -> std::optional<
                            BatchApplicationPresentBridgeAuthority::OwnedSemaphore> {
                        auto backing = std::make_shared<uint32_t>(nextSignal++);
                        return BatchApplicationPresentBridgeAuthority::OwnedSemaphore{
                            reinterpret_cast<VkSemaphore>(
                                static_cast<uintptr_t>(*backing)), backing};
                    },
                    .createFence = []() -> std::optional<
                            BatchApplicationPresentBridgeAuthority::OwnedFence> {
                        auto backing = std::make_shared<uint32_t>(0x871f);
                        return BatchApplicationPresentBridgeAuthority::OwnedFence{
                            reinterpret_cast<VkFence>(uintptr_t{0x871f}), backing};
                    },
                    .submit = [&](VkQueue queue, const VkSubmitInfo& info,
                            VkFence fence) {
                        ++bridgeSubmitCalls;
                        assert(info.waitSemaphoreCount == 2);
                        assert(info.pWaitSemaphores[0] == applicationWaits[0]);
                        assert(info.pWaitSemaphores[1] == applicationWaits[1]);
                        assert(info.signalSemaphoreCount == 2);
                        return lsfgvk::test::ShadowBReturnExecutionHarness::
                            controlledDeviceFunctions().QueueSubmit(
                                queue, 1, &info, fence);
                    },
                    .waitBridgeFence = [&](VkFence) {
                        ++bridgeHostWaits; return VK_SUCCESS;
                    },
                    .retireAsync = [&](VkFence, std::shared_ptr<void> value) {
                        asyncBridgeLifetime = std::move(value); return true;
                    },
                    .retainConservatively = [&](std::shared_ptr<void> value) {
                        asyncBridgeLifetime = std::move(value); return true;
                    }});
            assert(bridge && bridge->signalCount() == 2);
            assert(bridge->signal(0) != bridge->signal(1));
            auto s0 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting(
                applicationFence);
            const auto pool0 = presentHarness.latestCommandPool();
            const auto command0 = presentHarness.latestCommandBuffer();
            auto s1 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting();
            const auto pool1 = presentHarness.latestCommandPool();
            const auto command1 = presentHarness.latestCommandBuffer();
            assert(pool0 != pool1 && command0 != command1);
            assert(s0.attachWaitBacking(bridge->signalBacking(0)));
            assert(s1.attachWaitBacking(bridge->signalBacking(1)));
            assert(transaction->markWaitBridgePrepared());
            const auto presentsBefore = presentHarness.observedQueuePresentCalls();
            const auto bridgeSubmit = presentHarness.frontSubmitCount();
            assert(bridge->submit() == VK_SUCCESS);
            assert(bridge->accepted() && transaction->markWaitBridgeAccepted());
            assert(presentHarness.frontSubmitCount() == bridgeSubmit + 1);
            std::optional<Adaptive1xPreparedLogicalFinal> prepared0;
            std::optional<Adaptive1xPreparedLogicalFinal> prepared1;
            failGlobalAllocation.store(true, std::memory_order_relaxed);
            try {
                assert(transaction->markEntriesPreparing());
                presentHarness.setAcquiredImageIndex(0);
                prepared0.emplace(s0.execute(bridge->signal(0)));
                assert(bridge->markSignalConsumed(0));
                presentHarness.setAcquiredImageIndex(1);
                prepared1.emplace(s1.execute(bridge->signal(1)));
                assert(bridge->markSignalConsumed(1));
                Adaptive1xPreparedLogicalFinal* prepared[]{
                    &*prepared0, &*prepared1};
                for (uint32_t i = 0; i < 2; ++i) {
                    const auto& identity = prepared[i]->presentedIdentity();
                    assert(lsfgvk::layer::test::
                        installedPresentedLeasePreparationState(identity)
                        == PresentedPhysicalImageLeaseRegistry::PreparationState::
                            BoundUncommitted);
                    if (prepared[i]->applicationPresentFence()
                            != VK_NULL_HANDLE)
                        assert(lsfgvk::layer::test::
                            installedBorrowedPreparationState(
                                prepared[i]->applicationPresentFence(), identity)
                            == BorrowedPresentFenceRegistry::PreparationState::
                                BoundUncommitted);
                    auto& entry = transaction->entry(i);
                    entry.physicalSwapchain = prepared[i]->physicalSwapchain();
                    entry.physicalImageIndex = prepared[i]->physicalImageIndex();
                    entry.completionSemaphore = prepared[i]->presentReadySemaphore();
                    entry.deviceLifetimeIdentity = identity.deviceLifetimeIdentity;
                    entry.lifecycleGeneration = identity.swapchainLifecycleIdentity;
                    entry.presentOperationIdentity = identity.presentOperationIdentity;
                    entry.preparationAccepted = true;
                    if (prepared[i]->requiresInternalPresentFence()) {
                        assert(projection.patchInternalPresentFence(
                            i, prepared[i]->internalPresentFence()));
                    } else {
                        assert(projection.applicationFence(i)
                            == prepared[i]->applicationPresentFence());
                    }
                }
                assert(transaction->markEntriesReady());
                for (uint32_t i = 0; i < 2; ++i) {
                    const auto& prepared = i ? *prepared1 : *prepared0;
                    assert(lsfgvk::layer::test::
                        installedPresentedLeasePreparationState(
                            prepared.presentedIdentity())
                        == PresentedPhysicalImageLeaseRegistry::PreparationState::
                            BoundUncommitted);
                    if (prepared.applicationPresentFence() != VK_NULL_HANDLE)
                        assert(lsfgvk::layer::test::
                            installedBorrowedPreparationState(
                                prepared.applicationPresentFence(),
                                prepared.presentedIdentity())
                            == BorrowedPresentFenceRegistry::PreparationState::
                                BoundUncommitted);
                    assert(transaction->setFinalEntry(i,
                        prepared.physicalSwapchain(), prepared.physicalImageIndex()));
                    assert(transaction->setFinalWait(
                        i, prepared.presentReadySemaphore()));
                }
                const auto finalInfo = transaction->prepareFinalPresentInfo(
                    projection.head(), nullptr, 2);
                assert(finalInfo && finalInfo->swapchainCount == 2);
                assert(finalInfo->waitSemaphoreCount == 2);
                assert(finalInfo->pNext == projection.head());
                assert(finalInfo->pImageIndices[0]
                    == prepared0->physicalImageIndex());
                assert(finalInfo->pImageIndices[1]
                    == prepared1->physicalImageIndex());
                assert(finalInfo->pImageIndices[0] != logicalIndices[0]);
                assert(finalInfo->pImageIndices[1] != logicalIndices[1]);
            } catch (...) {
                failGlobalAllocation.store(false, std::memory_order_relaxed);
                throw;
            }
            failGlobalAllocation.store(false, std::memory_order_relaxed);
            assert(prepared0 && prepared1);
            assert(prepared0->acquireFence() != prepared1->acquireFence());
            assert(prepared0->producerFence() != prepared1->producerFence());
            assert(prepared0->presentReadySemaphore()
                != prepared1->presentReadySemaphore());
            assert(presentHarness.frontSubmit(bridgeSubmit + 1).wait
                == bridge->signal(0));
            assert(presentHarness.frontSubmit(bridgeSubmit + 2).wait
                == bridge->signal(1));
            assert(presentHarness.frontSubmit(bridgeSubmit + 1).wait
                != applicationWaits[0]);
            assert(presentHarness.frontSubmit(bridgeSubmit + 2).wait
                != applicationWaits[1]);
            assert(presentHarness.observedQueuePresentCalls() == presentsBefore);
            assert(bridgeSubmitCalls == 1 && bridgeHostWaits == 0);
            assert(bridge->retireNormalAsync());
            assert(asyncBridgeLifetime && bridgeHostWaits == 0);
            // Real two-authority pre-enqueue final-result simulation. The
            // transaction classifies only metadata; each prepared authority
            // performs its own exact producer recovery and lease abort.
            assert(prepared0->markLogicalPresentCalled());
            assert(prepared1->markLogicalPresentCalled());
            assert(transaction->markFinalPresentCalled());
            assert(transaction->markFinalPresentResult(
                VK_ERROR_OUT_OF_HOST_MEMORY));
            assert(prepared0->finalizeLogicalPresent(
                VK_ERROR_OUT_OF_HOST_MEMORY) == VK_SUCCESS);
            assert(prepared1->finalizeLogicalPresent(
                VK_ERROR_OUT_OF_DEVICE_MEMORY) == VK_SUCCESS);
            assert(lsfgvk::layer::test::installedPresentedLeasePreparationState(
                prepared0->presentedIdentity())
                == PresentedPhysicalImageLeaseRegistry::PreparationState::Missing);
            assert(lsfgvk::layer::test::installedPresentedLeasePreparationState(
                prepared1->presentedIdentity())
                == PresentedPhysicalImageLeaseRegistry::PreparationState::Missing);
        }

        // Failure after an accepted bridge: S0 is explicitly abandoned, S1
        // performs exact Acquire-fence recovery, and bridge acceptance is not
        // rolled back or followed by a partial logical present.
        {
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            auto s0 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting();
            auto s1 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting();
            const VkSemaphore applicationWait =
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8720});
            uint32_t nextPartialSignal = 0x8730;
            uint32_t partialBridgeWaits{};
            std::shared_ptr<void> partialRetained;
            auto bridge = BatchApplicationPresentBridgeAuthority::create(
                presentQueue, &applicationWait, 1, 2,
                std::make_shared<int>(2), {
                    .createSemaphore = [&]() -> std::optional<
                            BatchApplicationPresentBridgeAuthority::OwnedSemaphore> {
                        auto backing = std::make_shared<uint32_t>(
                            nextPartialSignal++);
                        return BatchApplicationPresentBridgeAuthority::OwnedSemaphore{
                            reinterpret_cast<VkSemaphore>(
                                static_cast<uintptr_t>(*backing)), backing};
                    },
                    .createFence = []() -> std::optional<
                            BatchApplicationPresentBridgeAuthority::OwnedFence> {
                        auto backing = std::make_shared<uint32_t>(0x873f);
                        return BatchApplicationPresentBridgeAuthority::OwnedFence{
                            reinterpret_cast<VkFence>(uintptr_t{0x873f}), backing};
                    },
                    .submit = [](VkQueue queue, const VkSubmitInfo& info,
                            VkFence fence) {
                        return lsfgvk::test::ShadowBReturnExecutionHarness::
                            controlledDeviceFunctions().QueueSubmit(
                                queue, 1, &info, fence);
                    },
                    .waitBridgeFence = [&](VkFence) {
                        ++partialBridgeWaits; return VK_SUCCESS;
                    },
                    .retireAsync = [&](VkFence, std::shared_ptr<void> value) {
                        partialRetained = std::move(value); return true;
                    },
                    .retainConservatively = [&](std::shared_ptr<void> value) {
                        partialRetained = std::move(value); return true;
                    }});
            assert(bridge);
            assert(s0.attachWaitBacking(bridge->signalBacking(0)));
            assert(s1.attachWaitBacking(bridge->signalBacking(1)));
            const auto presentsBefore = presentHarness.observedQueuePresentCalls();
            assert(bridge->submit() == VK_SUCCESS);
            assert(bridge->accepted());
            presentHarness.setAcquiredImageIndex(0);
            auto prepared0 = s0.execute(bridge->signal(0));
            assert(bridge->markSignalConsumed(0));
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.setShadowFailurePoint(
                lsfgvk::test::ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER);
            try {
                static_cast<void>(s1.execute(bridge->signal(1)));
                assert(false);
            } catch (const ls::vulkan_error&) {}
            presentHarness.setShadowFailurePoint(
                lsfgvk::test::ShadowHarnessFailurePoint::NONE);
            assert(s1.state()
                == Adaptive1xPreparationReservation::State::Recovered);
            assert(prepared0.abandonWithoutLogicalPresent() == VK_SUCCESS);
            assert(bridge->recoverPartial() == VK_SUCCESS);
            assert(partialBridgeWaits == 1);
            assert(bridge->accepted());
            assert(presentHarness.observedQueuePresentCalls() == presentsBefore);
        }

        // Real two-authority SUCCESS ownership simulation. No downstream
        // QueuePresentKHR is invoked; only PreparedLogicalFinal may commit the
        // two already-bound physical generations.
        {
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            const VkSwapchainKHR logical[]{
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8740}),
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8741})};
            const uint32_t logicalIndices[]{7, 8};
            const VkPresentInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = logical,
                .pImageIndices = logicalIndices};
            auto transaction = BatchPresentTransaction::create(
                presentDevice, presentQueue, 0x8742, info);
            assert(transaction && transaction->markWaitBridgePrepared());
            assert(transaction->markWaitBridgeAccepted());
            assert(transaction->markEntriesPreparing());
            auto s0 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting();
            auto s1 = lsfgvk::layer::test::reserveInstalledAdaptive1xForTesting();
            presentHarness.setAcquiredImageIndex(0);
            auto p0 = s0.execute(
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8743}));
            presentHarness.setAcquiredImageIndex(1);
            auto p1 = s1.execute(
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8744}));
            Adaptive1xPreparedLogicalFinal* prepared[]{&p0, &p1};
            for (uint32_t i = 0; i < 2; ++i) {
                const auto& identity = prepared[i]->presentedIdentity();
                assert(lsfgvk::layer::test::
                    installedPresentedLeasePreparationState(identity)
                    == PresentedPhysicalImageLeaseRegistry::PreparationState::
                        BoundUncommitted);
                auto& entry = transaction->entry(i);
                entry.path = BatchPresentTransaction::Path::VirtualPrepared;
                entry.physicalSwapchain = prepared[i]->physicalSwapchain();
                entry.physicalImageIndex = prepared[i]->physicalImageIndex();
                entry.completionSemaphore = prepared[i]->presentReadySemaphore();
                entry.deviceLifetimeIdentity = identity.deviceLifetimeIdentity;
                entry.lifecycleGeneration = identity.swapchainLifecycleIdentity;
                entry.presentOperationIdentity = identity.presentOperationIdentity;
                entry.preparationAccepted = true;
            }
            assert(transaction->markEntriesReady());
            for (uint32_t i = 0; i < 2; ++i) {
                assert(transaction->setFinalEntry(i,
                    prepared[i]->physicalSwapchain(),
                    prepared[i]->physicalImageIndex()));
                assert(transaction->setFinalWait(
                    i, prepared[i]->presentReadySemaphore()));
            }
            assert(transaction->prepareFinalPresentInfo(nullptr, nullptr, 2));
            assert(p0.markLogicalPresentCalled());
            assert(p1.markLogicalPresentCalled());
            assert(transaction->markFinalPresentCalled());
            assert(transaction->markFinalPresentResult(VK_SUBOPTIMAL_KHR));
            assert(presentHarness.simulatePresentFenceEnqueued(
                p0.internalPresentFence()));
            assert(presentHarness.simulatePresentFenceEnqueued(
                p1.internalPresentFence()));
            const auto p0Final = p0.finalizeLogicalPresent(VK_SUCCESS);
            const auto p1Final = p1.finalizeLogicalPresent(VK_SUBOPTIMAL_KHR);
            if (p0Final != VK_SUCCESS || p1Final != VK_SUCCESS)
                std::cerr << "real batch success finalization p0=" << p0Final
                    << " p1=" << p1Final << '\n';
            assert(p0Final == VK_SUCCESS);
            assert(p1Final == VK_SUCCESS);
            assert(lsfgvk::layer::test::installedPresentedLeasePreparationState(
                p0.presentedIdentity())
                == PresentedPhysicalImageLeaseRegistry::PreparationState::Committed);
            assert(lsfgvk::layer::test::installedPresentedLeasePreparationState(
                p1.presentedIdentity())
                == PresentedPhysicalImageLeaseRegistry::PreparationState::Committed);
        }

        // AR-15/AR-16 and CP-6: an accepted/indeterminate producer submit is
        // never reported as a clean abort. Its pool, command and physical lease
        // remain owned by the conservative device-lifetime authority.
        for (const auto terminal : {VK_ERROR_DEVICE_LOST,
                VK_ERROR_INITIALIZATION_FAILED}) {
            lsfgvk::layer::test::resetAdaptiveCommandPoolLifecycleCounts();
            const auto leasesBefore =
                lsfgvk::layer::test::installedPresentedLeaseCount();
            presentHarness.setQueueSubmitResult(terminal);
            auto reservation = lsfgvk::layer::test::
                reserveInstalledAdaptive1xForTesting();
            try {
                static_cast<void>(reservation.execute(
                    reinterpret_cast<VkSemaphore>(uintptr_t{0x84f4})));
                assert(false);
            } catch (const ls::vulkan_error&) {}
            assert(reservation.state()
                == Adaptive1xPreparationReservation::State::ConservativelyRetained);
            assert(lsfgvk::layer::test::adaptiveCommandPoolCreateCount() == 1);
            assert(lsfgvk::layer::test::adaptiveCommandPoolDestroyCount() == 0);
            assert(lsfgvk::layer::test::installedPresentedLeaseCount()
                == leasesBefore + 1);
        }
        presentHarness.setQueueSubmitResult(VK_SUCCESS);

        // VB production entrypoint smoke: two distinct virtual runtimes share
        // one application wait bridge, execute two reserved Adaptive producers,
        // and reach exactly one final downstream QueuePresentKHR.
        {
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto virtualA =
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8750});
            const auto virtualB =
                reinterpret_cast<VkSwapchainKHR>(uintptr_t{0x8751});
            assert(lsfgvk::layer::test::addInstalledVirtualSwapchainForTesting(
                virtualA));
            assert(lsfgvk::layer::test::addInstalledVirtualSwapchainForTesting(
                virtualB));
            uint32_t indices[2]{};
            const VkSemaphore acquireSignals[]{
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8752}),
                reinterpret_cast<VkSemaphore>(uintptr_t{0x8753})};
            assert(acquire(presentDevice, virtualA, 0, acquireSignals[0],
                VK_NULL_HANDLE, &indices[0]) == VK_SUCCESS);
            assert(acquire(presentDevice, virtualB, 0, acquireSignals[1],
                VK_NULL_HANDLE, &indices[1]) == VK_SUCCESS);
            const VkSwapchainKHR swapchains[]{virtualA, virtualB};
            VkResult results[]{VK_NOT_READY, VK_NOT_READY};
            const VkPresentInfoKHR batchInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 2,
                .pWaitSemaphores = acquireSignals,
                .swapchainCount = 2,
                .pSwapchains = swapchains,
                .pImageIndices = indices,
                .pResults = results};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 3);
            presentHarness.setAcquiredImageIndex(0);
            presentHarness.failAllocationsAfterFrontSubmit(
                &failGlobalAllocation, submits + 1);
            presentHarness.restoreAllocationFailureAfterPresent(true);
            const auto result = virtualPresent(presentQueue, &batchInfo);
            presentHarness.restoreAllocationFailureAfterPresent(false);
            presentHarness.clearPostSubmitAllocationFailure();
            if (result != VK_SUCCESS)
                std::cerr << "production virtual batch result=" << result
                    << " per=[" << results[0] << ',' << results[1] << "]\n";
            assert(result == VK_SUCCESS);
            assert(results[0] == VK_SUCCESS && results[1] == VK_SUCCESS);
            assert(presentHarness.frontSubmitCount() == submits + 3);
            assert(presentHarness.observedQueuePresentCalls() == presents + 1);
            assert(presentHarness.observedPresentedWaits().size() == 2);
            assert(presentHarness.observedPresentFenceStructCount() == 1);
        }

        // Public VB qualification groups. Every transformed case below enters
        // myvkQueuePresentKHR; component authorities are not invoked directly.
        uintptr_t nextVbHandle{0x9000};
        uintptr_t nextVbSemaphore{0x9800};
        const auto makeVirtual = [&](bool acquireImage = true) {
            const auto swapchain = reinterpret_cast<VkSwapchainKHR>(
                nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledVirtualSwapchainForTesting(
                swapchain));
            uint32_t index{};
            if (acquireImage) {
                const auto signal = reinterpret_cast<VkSemaphore>(
                    nextVbSemaphore++);
                assert(acquire(presentDevice, swapchain, 0, signal,
                    VK_NULL_HANDLE, &index) == VK_SUCCESS);
            }
            return std::pair{swapchain, index};
        };
        const auto runVirtualBatch = [&](uint32_t virtualCount,
                uint32_t waitCount, const void* pNext, bool withResults,
                VkResult expected) {
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            std::vector<VkSwapchainKHR> swapchains;
            std::vector<uint32_t> indices;
            std::vector<VkSemaphore> waits;
            std::vector<VkResult> results(virtualCount, VK_NOT_READY);
            swapchains.reserve(virtualCount);
            indices.reserve(virtualCount);
            waits.reserve(waitCount);
            for (uint32_t i = 0; i < virtualCount; ++i) {
                const auto [swapchain, index] = makeVirtual();
                swapchains.push_back(swapchain);
                indices.push_back(index);
            }
            for (uint32_t i = 0; i < waitCount; ++i)
                waits.push_back(reinterpret_cast<VkSemaphore>(nextVbSemaphore++));
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + virtualCount + 1);
            presentHarness.setAcquiredImageIndex(1);
            const VkPresentInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = pNext,
                .waitSemaphoreCount = waitCount,
                .pWaitSemaphores = waits.empty() ? nullptr : waits.data(),
                .swapchainCount = virtualCount,
                .pSwapchains = swapchains.data(),
                .pImageIndices = indices.data(),
                .pResults = withResults ? results.data() : nullptr};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            const auto result = virtualPresent(presentQueue, &info);
            assert(result == expected);
            if (expected == VK_SUCCESS || expected == VK_SUBOPTIMAL_KHR) {
                assert(presentHarness.frontSubmitCount()
                    == submits + virtualCount + 1);
                assert(presentHarness.observedQueuePresentCalls() == presents + 1);
                assert(presentHarness.observedPresentedWaits().size()
                    == virtualCount);
                if (withResults)
                    for (const auto value : results) assert(value == expected);
            }
            return result;
        };

        // VB-2/6/10/11 and VB-35..44: three virtuals, zero application
        // waits, no pNext/pResults, distinct real producer resources, one
        // bridge, one final present, dense layer-owned waits.
        static_cast<void>(runVirtualBatch(3, 0, nullptr, false, VK_SUCCESS));
        // VB-7 and VB-8/9.
        static_cast<void>(runVirtualBatch(2, 1, nullptr, true, VK_SUCCESS));
        static_cast<void>(runVirtualBatch(2, 3, nullptr, true, VK_SUCCESS));

        // VB-3/39/40: Native, Virtual, Native, Virtual produces exactly two
        // dense final waits and preserves native positions.
        {
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto native0 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            const auto native2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledNativeSwapchainForTesting(
                native0));
            assert(lsfgvk::layer::test::addInstalledNativeSwapchainForTesting(
                native2));
            const auto v1 = makeVirtual();
            const auto v3 = makeVirtual();
            const VkSwapchainKHR swapchains[]{native0, v1.first, native2, v3.first};
            const uint32_t indices[]{17, v1.second, 19, v3.second};
            const auto appWait = reinterpret_cast<VkSemaphore>(nextVbSemaphore++);
            VkResult results[4]{};
            const VkPresentInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1, .pWaitSemaphores = &appWait,
                .swapchainCount = 4, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 3);
            presentHarness.setAcquiredImageIndex(1);
            assert(virtualPresent(presentQueue, &info) == VK_SUCCESS);
            assert(presentHarness.frontSubmitCount() == submits + 3);
            assert(presentHarness.observedQueuePresentCalls() == presents + 1);
            assert(presentHarness.observedPresentedWaits().size() == 2);
            for (const auto wait : presentHarness.observedPresentedWaits())
                assert(wait != appWait && wait != VK_NULL_HANDLE);
        }

        // VB-4/VB-5/VB-34: D2 entries join the existing whole-call authority;
        // their generation bookkeeping is committed only from their exact
        // final per-entry result.
        {
            const auto virtualEntry = makeVirtual();
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const VkSwapchainKHR swapchains[]{virtualEntry.first, d2};
            const uint32_t indices[]{virtualEntry.second, 0};
            VkResult results[2]{};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 2);
            presentHarness.setAcquiredImageIndex(1);
            assert(virtualPresent(presentQueue, &info) == VK_SUCCESS);
            assert(presentHarness.frontSubmitCount() == submits + 2);
            assert(presentHarness.observedQueuePresentCalls() == presents + 1);
            assert(results[0] == VK_SUCCESS && results[1] == VK_SUCCESS);
            const auto d2After = lsfgvk::layer::test::installedD2StateForTesting(
                d2, 0);
            assert(d2After && !d2After->appAcquired
                && !d2After->batchPresentPrepared
                && d2After->applicationGeneration == 1);
        }
        {
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            const auto native0 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            const auto native3 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            assert(lsfgvk::layer::test::addInstalledNativeSwapchainForTesting(native0));
            assert(lsfgvk::layer::test::addInstalledNativeSwapchainForTesting(native3));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, native0,
                virtualEntry.first, native3};
            const uint32_t indices[]{0, 7, virtualEntry.second, 9};
            const VkSemaphore waits[]{
                reinterpret_cast<VkSemaphore>(nextVbSemaphore++),
                reinterpret_cast<VkSemaphore>(nextVbSemaphore++),
                reinterpret_cast<VkSemaphore>(nextVbSemaphore++)};
            VkResult results[4]{};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 3, .pWaitSemaphores = waits,
                .swapchainCount = 4, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 2);
            presentHarness.setAcquiredImageIndex(1);
            assert(virtualPresent(presentQueue, &info) == VK_SUCCESS);
            assert(presentHarness.frontSubmitCount() == submits + 2
                && presentHarness.observedQueuePresentCalls() == presents + 1);
            assert(presentHarness.observedPresentedWaits().size() == 1);
            const auto d2After = lsfgvk::layer::test::installedD2StateForTesting(
                d2, 0);
            assert(d2After && d2After->applicationGeneration == 1);
        }
        {
            // Invalid D2 ownership/index remains a zero-side-effect firewall.
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{1, virtualEntry.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            assert(virtualPresent(presentQueue, &info) == VK_ERROR_DEVICE_LOST);
            assert(presentHarness.frontSubmitCount() == submits
                && presentHarness.observedQueuePresentCalls() == presents);
        }
        {
            // Reverse-order D2 + Virtual, pResults == nullptr, and the R0 cold
            // rule after the bridge has been accepted.
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{0, virtualEntry.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 2);
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.failAllocationsAfterFrontSubmit(
                &failGlobalAllocation, submits + 1);
            presentHarness.restoreAllocationFailureAfterPresent(true);
            assert(virtualPresent(presentQueue, &info) == VK_SUCCESS);
            presentHarness.restoreAllocationFailureAfterPresent(false);
            presentHarness.clearPostSubmitAllocationFailure();
            assert(presentHarness.frontSubmitCount() == submits + 2
                && presentHarness.observedQueuePresentCalls() == presents + 1);
        }

        // VB-12: currently supported whole-batch present-mode projection.
        {
            const VkPresentModeKHR modes[]{VK_PRESENT_MODE_FIFO_KHR,
                VK_PRESENT_MODE_FIFO_KHR};
            VkSwapchainPresentModeInfoKHR modesInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
                .swapchainCount = 2, .pPresentModes = modes};
            const void* supportedChain = &modesInfo;
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            const VkFence fences[]{VK_NULL_HANDLE, VK_NULL_HANDLE};
            VkSwapchainPresentFenceInfoKHR fenceInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .pNext = supportedChain, .swapchainCount = 2,
                .pFences = fences};
            supportedChain = &fenceInfo;
#endif
#if defined(VK_EXT_present_timing)
            const VkPresentTimingInfoEXT timings[2]{};
            VkPresentTimingsInfoEXT timingInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT,
                .pNext = supportedChain, .swapchainCount = 2,
                .pTimingInfos = timings};
            supportedChain = &timingInfo;
#endif
#if defined(VK_GOOGLE_display_timing)
            const VkPresentTimeGOOGLE times[2]{};
            VkPresentTimesInfoGOOGLE timesInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE,
                .pNext = supportedChain, .swapchainCount = 2,
                .pTimes = times};
            supportedChain = &timesInfo;
#endif
#if defined(VK_KHR_incremental_present)
            const VkPresentRegionKHR regions[2]{};
            VkPresentRegionsKHR regionsInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
                .pNext = supportedChain, .swapchainCount = 2,
                .pRegions = regions};
            supportedChain = &regionsInfo;
#endif
#if defined(VK_KHR_present_id2)
            const uint64_t ids2[]{201, 202};
            VkPresentId2KHR id2Info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR,
                .pNext = supportedChain, .swapchainCount = 2,
                .pPresentIds = ids2};
            supportedChain = &id2Info;
#endif
#if defined(VK_KHR_present_id)
            const uint64_t ids[]{101, 102};
            VkPresentIdKHR idInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
                .pNext = supportedChain, .swapchainCount = 2,
                .pPresentIds = ids};
            supportedChain = &idInfo;
#endif
#if defined(VK_KHR_device_group)
            const uint32_t masks[]{1, 1};
            VkDeviceGroupPresentInfoKHR groupInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
                .pNext = supportedChain, .swapchainCount = 2,
                .pDeviceMasks = masks,
                .mode = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR};
            supportedChain = &groupInfo;
#endif
            static_cast<void>(runVirtualBatch(
                2, 1, supportedChain, true, VK_SUCCESS));
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR mixedSwapchains[]{d2, virtualEntry.first};
            const uint32_t mixedIndices[]{0, virtualEntry.second};
            const auto mixedWait = reinterpret_cast<VkSemaphore>(nextVbSemaphore++);
            VkResult mixedResults[2]{};
            const VkPresentInfoKHR mixedInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = supportedChain, .waitSemaphoreCount = 1,
                .pWaitSemaphores = &mixedWait, .swapchainCount = 2,
                .pSwapchains = mixedSwapchains, .pImageIndices = mixedIndices,
                .pResults = mixedResults};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            presentHarness.setAcquiredImageIndex(1);
            assert(virtualPresent(presentQueue, &mixedInfo) == VK_SUCCESS);
            assert(mixedResults[0] == VK_SUCCESS
                && mixedResults[1] == VK_SUCCESS);
            assert(modes[0] == VK_PRESENT_MODE_FIFO_KHR
                && modes[1] == VK_PRESENT_MODE_FIFO_KHR);
        }

        // VB-13/14/15: NULL fences are substituted internally; any non-NULL
        // application fence and unsupported pNext reject before queue work.
        {
            const VkFence nullFences[]{VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkSwapchainPresentFenceInfoKHR nullFenceInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .swapchainCount = 2, .pFences = nullFences};
            static_cast<void>(runVirtualBatch(
                2, 1, &nullFenceInfo, true, VK_SUCCESS));
            assert(nullFences[0] == VK_NULL_HANDLE
                && nullFences[1] == VK_NULL_HANDLE);

            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            const VkFence forbidden[]{applicationFence, VK_NULL_HANDLE};
            const VkSwapchainPresentFenceInfoKHR forbiddenInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .swapchainCount = 2, .pFences = forbidden};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = &forbiddenInfo, .swapchainCount = 2,
                .pSwapchains = swapchains, .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            assert(virtualPresent(presentQueue, &info) == VK_ERROR_UNKNOWN);
            assert(presentHarness.frontSubmitCount() == submits
                && presentHarness.observedQueuePresentCalls() == presents);

            const VkBaseInStructure unsupported{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = nullptr};
            VkPresentInfoKHR unsupportedInfo = info;
            unsupportedInfo.pNext = &unsupported;
            assert(virtualPresent(presentQueue, &unsupportedInfo)
                == VK_ERROR_UNKNOWN);
            assert(presentHarness.frontSubmitCount() == submits
                && presentHarness.observedQueuePresentCalls() == presents);
        }

        // VB-16..18: transaction allocation, runtime state, and real Adaptive
        // builder failures all remain before the bridge.
        {
            auto a = makeVirtual();
            auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            VkResult results[2]{};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            failGlobalAllocation.store(true, std::memory_order_relaxed);
            assert(virtualPresent(presentQueue, &info)
                == VK_ERROR_OUT_OF_HOST_MEMORY);
            failGlobalAllocation.store(false, std::memory_order_relaxed);
            assert(presentHarness.frontSubmitCount() == submits
                && presentHarness.observedQueuePresentCalls() == presents);

            a = makeVirtual(false);
            b = makeVirtual();
            const VkSwapchainKHR stateSwapchains[]{a.first, b.first};
            const uint32_t stateIndices[]{0, b.second};
            VkPresentInfoKHR stateInfo = info;
            stateInfo.pSwapchains = stateSwapchains;
            stateInfo.pImageIndices = stateIndices;
            const auto stateSubmits = presentHarness.frontSubmitCount();
            const auto statePresents = presentHarness.observedQueuePresentCalls();
            assert(virtualPresent(presentQueue, &stateInfo)
                == VK_ERROR_DEVICE_LOST);
            assert(presentHarness.frontSubmitCount() == stateSubmits
                && presentHarness.observedQueuePresentCalls() == statePresents);

            a = makeVirtual();
            b = makeVirtual();
            const VkSwapchainKHR builderSwapchains[]{a.first, b.first};
            const uint32_t builderIndices[]{a.second, b.second};
            VkPresentInfoKHR builderInfo = info;
            builderInfo.pSwapchains = builderSwapchains;
            builderInfo.pImageIndices = builderIndices;
            const auto builderSubmits = presentHarness.frontSubmitCount();
            const auto builderPresents = presentHarness.observedQueuePresentCalls();
            lsfgvk::layer::test::setAdaptiveBuilderFailurePoint(
                BuilderFailure::AcquireBacking);
            assert(virtualPresent(presentQueue, &builderInfo)
                == VK_ERROR_OUT_OF_HOST_MEMORY);
            lsfgvk::layer::test::setAdaptiveBuilderFailurePoint(
                BuilderFailure::None);
            assert(presentHarness.frontSubmitCount() == builderSubmits
                && presentHarness.observedQueuePresentCalls() == builderPresents);
        }

        // VB-19..21: exact bridge results occur before every producer/final.
        for (const auto bridgeResult : {VK_ERROR_OUT_OF_HOST_MEMORY,
                VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_DEVICE_LOST}) {
            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.setQueueSubmitResult(bridgeResult);
            assert(virtualPresent(presentQueue, &info) == bridgeResult);
            presentHarness.setQueueSubmitResult(VK_SUCCESS);
            assert(presentHarness.frontSubmitCount() == submits + 1);
            assert(presentHarness.observedQueuePresentCalls() == presents);
        }
        {
            // D2 reservation failure is injected after all transaction storage
            // exists but before the bridge; no queue operation is observable.
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{0, virtualEntry.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            lsfgvk::layer::test::setD2BatchReservationFailureForTesting(true);
            assert(virtualPresent(presentQueue, &info)
                == VK_ERROR_OUT_OF_HOST_MEMORY);
            lsfgvk::layer::test::setD2BatchReservationFailureForTesting(false);
            assert(presentHarness.frontSubmitCount() == submits
                && presentHarness.observedQueuePresentCalls() == presents);
        }

        // VB-22: bridge and P0 are accepted; P1 fails. Recovery is typed and
        // no final QueuePresentKHR is issued.
        {
            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            const auto submits = presentHarness.frontSubmitCount();
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(submits + 3);
            presentHarness.setQueueSubmitFailureOnCall(
                static_cast<uint32_t>(submits + 3),
                VK_ERROR_OUT_OF_HOST_MEMORY);
            assert(virtualPresent(presentQueue, &info) == VK_ERROR_DEVICE_LOST);
            presentHarness.clearQueueSubmitFailureOnCall();
            assert(presentHarness.frontSubmitCount() == submits + 3);
            assert(presentHarness.observedQueuePresentCalls() == presents);
        }

        // VB-23..32 plus VB-33: final result classes and differing defined
        // per-entry results are delivered through the one final call.
        for (const auto finalResult : {VK_SUCCESS, VK_SUBOPTIMAL_KHR,
                VK_ERROR_OUT_OF_DATE_KHR, VK_ERROR_SURFACE_LOST_KHR,
                VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT,
                VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                VK_ERROR_DEVICE_LOST, VK_ERROR_INITIALIZATION_FAILED}) {
            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            VkResult results[]{VK_NOT_READY, VK_NOT_READY};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            const auto presents = presentHarness.observedQueuePresentCalls();
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 3);
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.setQueuePresentResult(finalResult);
            const auto result = virtualPresent(presentQueue, &info);
            presentHarness.setQueuePresentResult(VK_SUCCESS);
            assert(presentHarness.observedQueuePresentCalls() == presents + 1);
            const auto expected = (finalResult == VK_ERROR_OUT_OF_HOST_MEMORY
                    || finalResult == VK_ERROR_OUT_OF_DEVICE_MEMORY
                    || finalResult == VK_ERROR_INITIALIZATION_FAILED)
                ? VK_ERROR_DEVICE_LOST : finalResult;
            assert(result == expected);
            assert(results[0] == expected && results[1] == expected);
        }
        {
            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            VkResult results[2]{};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 3);
            presentHarness.setQueuePresentPerResults(
                {VK_SUCCESS, VK_SUBOPTIMAL_KHR});
            assert(virtualPresent(presentQueue, &info) == VK_SUBOPTIMAL_KHR);
            assert(results[0] == VK_SUCCESS
                && results[1] == VK_SUBOPTIMAL_KHR);
        }
        // D2 result authority: every aggregate class reaches only its own D2
        // reservation, with pResults used only when Vulkan defines it.
        for (const auto finalResult : {VK_SUCCESS, VK_SUBOPTIMAL_KHR,
                VK_ERROR_OUT_OF_DATE_KHR, VK_ERROR_SURFACE_LOST_KHR,
                VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT,
                VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                VK_ERROR_DEVICE_LOST, VK_ERROR_INITIALIZATION_FAILED}) {
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{0, virtualEntry.second};
            VkResult results[]{VK_NOT_READY, VK_NOT_READY};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.setQueuePresentResult(finalResult);
            const auto result = virtualPresent(presentQueue, &info);
            presentHarness.setQueuePresentResult(VK_SUCCESS);
            const auto expected = (finalResult == VK_ERROR_OUT_OF_HOST_MEMORY
                    || finalResult == VK_ERROR_OUT_OF_DEVICE_MEMORY
                    || finalResult == VK_ERROR_INITIALIZATION_FAILED)
                ? VK_ERROR_DEVICE_LOST : finalResult;
            assert(result == expected && results[0] == expected
                && results[1] == expected);
            const auto d2After = lsfgvk::layer::test::installedD2StateForTesting(
                d2, 0);
            assert(d2After);
            const auto classification = classifyPresentResult(finalResult);
            if (classification == PresentResultClass::Normal
                    || classification == PresentResultClass::EnqueuedRejection)
                assert(!d2After->appAcquired
                    && d2After->applicationGeneration == 1);
            else if (classification == PresentResultClass::PreEnqueueFailure)
                assert(d2After->appAcquired
                    && !d2After->batchPresentPrepared
                    && d2After->applicationGeneration == 0);
            else
                assert(d2After->appAcquired && d2After->batchPresentPrepared
                    && d2After->applicationGeneration == 0);
        }
        {
            // Defined mixed per-entry results do not cross-finalize D2 state.
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{0, virtualEntry.second};
            VkResult results[2]{};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices, .pResults = results};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.setQueuePresentPerResults(
                {VK_ERROR_OUT_OF_DATE_KHR, VK_SUCCESS});
            assert(virtualPresent(presentQueue, &info) == VK_ERROR_OUT_OF_DATE_KHR);
            const auto d2After = lsfgvk::layer::test::installedD2StateForTesting(
                d2, 0);
            assert(d2After && !d2After->appAcquired
                && !d2After->batchPresentPrepared
                && d2After->applicationGeneration == 1);
        }
        {
            // An indeterminate final result retains the old D2 generation
            // through destroy/recreate; handle reuse installs a distinct state.
            assert(lsfgvk::layer::test::
                replaceInstalledRetirementReactorWithFreshForTesting(
                    presentDevice));
            const auto d2 = reinterpret_cast<VkSwapchainKHR>(nextVbHandle++);
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto virtualEntry = makeVirtual();
            const VkSwapchainKHR swapchains[]{d2, virtualEntry.first};
            const uint32_t indices[]{0, virtualEntry.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 2);
            presentHarness.setAcquiredImageIndex(1);
            presentHarness.setQueuePresentResult(VK_ERROR_DEVICE_LOST);
            assert(virtualPresent(presentQueue, &info) == VK_ERROR_DEVICE_LOST);
            presentHarness.setQueuePresentResult(VK_SUCCESS);
            const auto destroyD2 = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
                lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                    "vkDestroySwapchainKHR"));
            assert(destroyD2);
            destroyD2(presentDevice, d2, nullptr);
            assert(lsfgvk::layer::test::lastInstalledD2StateAliveForTesting());
            assert(lsfgvk::layer::test::addInstalledD2SwapchainForTesting(d2));
            const auto replacement =
                lsfgvk::layer::test::installedD2StateForTesting(d2, 0);
            assert(replacement && replacement->appAcquired
                && !replacement->batchPresentPrepared
                && replacement->applicationGeneration == 0);
        }

        // VB-47: keep producer/WSI reactor FDs pending, destroy one virtual
        // runtime, verify exact pending-operation adoption, then recreate the
        // same application handle without touching the detached authority.
        {
            const auto a = makeVirtual();
            const auto b = makeVirtual();
            const VkSwapchainKHR swapchains[]{a.first, b.first};
            const uint32_t indices[]{a.second, b.second};
            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .swapchainCount = 2, .pSwapchains = swapchains,
                .pImageIndices = indices};
            presentHarness.reserveFrontSubmitCapacity(
                presentHarness.frontSubmitCount() + 3);
            presentHarness.setFenceFdInitiallySignaled(false);
            assert(virtualPresent(presentQueue, &info) == VK_SUCCESS);
            const auto deferredBefore =
                lsfgvk::layer::test::deferredPendingCount(presentDevice);
            const auto destroyForVb47 = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
                lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                    "vkDestroySwapchainKHR"));
            assert(destroyForVb47);
            destroyForVb47(presentDevice, a.first, nullptr);
            assert(lsfgvk::layer::test::deferredPendingCount(presentDevice)
                >= deferredBefore + 1);
            assert(lsfgvk::layer::test::addInstalledVirtualSwapchainForTesting(
                a.first));
            presentHarness.setFenceFdInitiallySignaled(true);
        }

        const auto destroySwapchain = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkDestroySwapchainKHR"));
        assert(destroySwapchain);
        AllocatorProbe allocatorProbe{};
        allocatorProbe.callingThread = std::this_thread::get_id();
        const VkAllocationCallbacks createTimeAllocator{
            .pUserData = nullptr,
            .pfnAllocation = probeAllocate,
            .pfnReallocation = probeReallocate,
            .pfnFree = probeFree};
        const VkAllocationCallbacks destroyTimeAllocator{
            .pUserData = &allocatorProbe,
            .pfnAllocation = probeAllocate,
            .pfnReallocation = probeReallocate,
            .pfnFree = probeFree};
        assert(&createTimeAllocator != &destroyTimeAllocator);
        assert(lsfgvk::layer::test::seedPendingRetirement(presentSwapchain));
        const auto idleBeforeDestroy = presentHarness.deviceIdleCalls;
        const auto destroysBeforeVirtualDestroy =
            presentHarness.observedDestroySwapchainCalls();
        allocatorProbe.insidePublicDestroy = true;
        destroySwapchain(presentDevice, presentSwapchain, &destroyTimeAllocator);
        allocatorProbe.insidePublicDestroy = false;
        assert(presentHarness.deviceIdleCalls == idleBeforeDestroy + 1);
        assert(presentHarness.observedDestroySwapchainCalls()
            == destroysBeforeVirtualDestroy + 1);
        assert(presentHarness.observedDestroyAllocator() == &destroyTimeAllocator);
        assert(presentHarness.observedDestroyThread() == allocatorProbe.callingThread);
        assert(allocatorProbe.callbacks == 1);
        assert(allocatorProbe.callbackInsideCommand);
        assert(allocatorProbe.callbackOnCallingThread);
        assert(allocatorProbe.callbackWhileValid);
        assert(lsfgvk::layer::test::deferredPendingCount(presentDevice) >= 1);
        assert(lsfgvk::layer::test::seededPendingBackingAlive());
        assert(lsfgvk::layer::test::seededPendingDestroyCount() == 0);

        allocatorProbe.valid = false;
        lsfgvk::layer::test::notifyDeferredRetirement(presentDevice);
        assert(allocatorProbe.callbacks == 1);
        assert(lsfgvk::layer::test::deferredPendingCount(presentDevice) >= 1);
        assert(lsfgvk::layer::test::seededPendingBackingAlive());
        assert(lsfgvk::layer::test::seededPendingDestroyCount() == 0);
        lsfgvk::layer::test::completeSeededPendingRetirement();
        lsfgvk::layer::test::notifyDeferredRetirement(presentDevice);
        assert(!lsfgvk::layer::test::seededPendingBackingAlive());
        assert(lsfgvk::layer::test::seededPendingDestroyCount() == 1);
        lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();

        // M8: a WaitIdle error is irrelevant to the already-adopted present
        // authority; it remains owned until exact completion.
        auto waitIdleErrorVulkan = presentHarness.createQueuePresentVulkan();
        assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
            .device = presentDevice,
            .queue = presentQueue,
            .queueFamily = presentHarness.endpoint().queueFamilyIndex,
            .swapchain = presentSwapchain,
            .vulkan = waitIdleErrorVulkan.get(),
            .backend = backend.get(),
            .devicePair = presentHarness.sameDevicePair(),
            .swapchainInfo = virtualInfo,
            .deviceLifetimeIdentity = 103,
            .virtualized = true}));
        waitIdleErrorVulkan.reset();
        assert(lsfgvk::layer::test::seedPendingRetirement(presentSwapchain));
        presentHarness.setDeviceWaitIdleResult(VK_ERROR_OUT_OF_HOST_MEMORY);
        const auto destroyWithIdleError = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
            lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                "vkDestroySwapchainKHR"));
        assert(destroyWithIdleError);
        const auto destroysBeforeIdleError =
            presentHarness.observedDestroySwapchainCalls();
        destroyWithIdleError(presentDevice, presentSwapchain, nullptr);
        assert(presentHarness.observedDestroySwapchainCalls()
            == destroysBeforeIdleError + 1);
        assert(lsfgvk::layer::test::deferredPendingCount(presentDevice) >= 1);
        assert(lsfgvk::layer::test::seededPendingBackingAlive());
        assert(lsfgvk::layer::test::seededPendingDestroyCount() == 0);
        lsfgvk::layer::test::completeSeededPendingRetirement();
        lsfgvk::layer::test::notifyDeferredRetirement(presentDevice);
        assert(!lsfgvk::layer::test::seededPendingBackingAlive());
        assert(lsfgvk::layer::test::seededPendingDestroyCount() == 1);
        presentHarness.setDeviceWaitIdleResult(VK_SUCCESS);
        lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();

        const auto runOriginCase = [&](VkResult internalFailure,
                VkResult hiddenAcquireResult, VkResult downstreamPresentResult,
                VkResult expected, uint64_t identity, bool extOnly = false,
                VkResult fenceExportResult = VK_SUCCESS) {
            auto caseVulkan = presentHarness.createQueuePresentVulkan(extOnly);
            auto caseSwapchainInfo = virtualInfo;
            caseSwapchainInfo.releaseBackend = extOnly
                ? SwapchainReleaseBackend::Ext : SwapchainReleaseBackend::Khr;
            assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
                .device = presentDevice,
                .queue = presentQueue,
                .queueFamily = presentHarness.endpoint().queueFamilyIndex,
                .swapchain = presentSwapchain,
                .vulkan = caseVulkan.get(),
                .backend = backend.get(),
                .devicePair = presentHarness.sameDevicePair(),
                .swapchainInfo = caseSwapchainInfo,
                .deviceLifetimeIdentity = identity,
                .virtualized = true}));
            caseVulkan.reset();
            lsfgvk::layer::test::setQueuePresentWorkerInternalFailure(
                internalFailure);
            presentHarness.setAcquireNextImageResult(hiddenAcquireResult);
            presentHarness.setQueuePresentResult(downstreamPresentResult);
            presentHarness.setFenceFdExportResult(fenceExportResult);
            if (lsfgvk::layer::classifyPresentResult(downstreamPresentResult)
                    == lsfgvk::layer::PresentResultClass::PreEnqueueFailure)
                presentHarness.setFenceWaitResultForTesting(VK_SUCCESS);
            const auto caseAcquire = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
                lsfgvk::layer::test::queuePresentAcquireEntrypoint());
            const auto casePresent = reinterpret_cast<PFN_vkQueuePresentKHR>(
                lsfgvk::layer::test::queuePresentEntrypoint());
            assert(caseAcquire && casePresent);
            VkFence caseApplicationFence{VK_NULL_HANDLE};
            VkSwapchainPresentFenceInfoKHR caseFenceInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .swapchainCount = 1,
                .pFences = &caseApplicationFence};
            if (fenceExportResult != VK_SUCCESS) {
                const auto caseCreateFence = reinterpret_cast<PFN_vkCreateFence>(
                    lsfgvk::layer::test::queuePresentDeviceEntrypoint(
                        "vkCreateFence"));
                assert(caseCreateFence);
                const VkFenceCreateInfo createInfo{
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                assert(caseCreateFence(presentDevice, &createInfo, nullptr,
                    &caseApplicationFence) == VK_SUCCESS);
            }
            if (identity == 104) {
                const auto missingSwapchain = reinterpret_cast<VkSwapchainKHR>(
                    uintptr_t{0xDEAD104});
                const uint32_t missingIndex{};
                VkResult missingPerSwapchain = VK_SUCCESS;
                const VkPresentInfoKHR missingInfo{
                    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                    .swapchainCount = 1,
                    .pSwapchains = &missingSwapchain,
                    .pImageIndices = &missingIndex,
                    .pResults = &missingPerSwapchain};
                assert(casePresent(presentQueue, &missingInfo)
                    == VK_ERROR_DEVICE_LOST);
                assert(missingPerSwapchain == VK_ERROR_DEVICE_LOST);
                assert(missingPerSwapchain != VK_ERROR_INITIALIZATION_FAILED);
            }
            uint32_t caseIndex{};
            const auto ready = reinterpret_cast<VkSemaphore>(
                uintptr_t{0x8600 + identity});
            assert(caseAcquire(presentDevice, presentSwapchain, 0, ready,
                VK_NULL_HANDLE, &caseIndex) == VK_SUCCESS);
            VkResult perSwapchain = VK_SUCCESS;
            const VkPresentInfoKHR caseInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = caseApplicationFence != VK_NULL_HANDLE
                    ? &caseFenceInfo : nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &ready,
                .swapchainCount = 1,
                .pSwapchains = &presentSwapchain,
                .pImageIndices = &caseIndex,
                .pResults = &perSwapchain};
            const auto releasesBefore = presentHarness.observedReleaseCalls();
            const auto aggregate = casePresent(presentQueue, &caseInfo);
            if (aggregate != expected || perSwapchain != expected)
                std::cerr << "origin case " << identity << " aggregate="
                    << aggregate << " perSwapchain=" << perSwapchain
                    << " expected=" << expected << '\n';
            assert(aggregate == expected);
            assert(perSwapchain == expected);
            if (fenceExportResult != VK_SUCCESS) {
                assert(lsfgvk::layer::test::productionPresentedLeaseInstalled());
                assert(presentHarness.observedReleaseCalls() == releasesBefore);
            }
            if (lsfgvk::layer::classifyPresentResult(downstreamPresentResult)
                    == lsfgvk::layer::PresentResultClass::PreEnqueueFailure) {
                assert(presentHarness.observedReleaseCalls() == releasesBefore + 1);
            }
            presentHarness.setAcquireNextImageResult(VK_SUCCESS);
            presentHarness.setQueuePresentResult(VK_SUCCESS);
            presentHarness.setFenceWaitResultForTesting(VK_TIMEOUT);
            presentHarness.setFenceFdExportResult(VK_SUCCESS);
            lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();
        };

        // C7/C8: generic internal Vulkan exceptions after C1 are firewalled.
        runOriginCase(VK_ERROR_INITIALIZATION_FAILED, VK_SUCCESS, VK_SUCCESS,
            VK_ERROR_DEVICE_LOST, 104);
        runOriginCase(VK_ERROR_OUT_OF_HOST_MEMORY, VK_SUCCESS, VK_SUCCESS,
            VK_ERROR_DEVICE_LOST, 105);
        // C9: identical enum, different operation origin: hidden acquire is internal.
        runOriginCase(VK_SUCCESS, VK_ERROR_OUT_OF_DATE_KHR, VK_SUCCESS,
            VK_ERROR_DEVICE_LOST, 106);
        // C10/C11: only the exact final QueuePresentKHR keeps downstream origin.
        runOriginCase(VK_SUCCESS, VK_SUCCESS, VK_ERROR_OUT_OF_DATE_KHR,
            VK_ERROR_OUT_OF_DATE_KHR, 107);
        runOriginCase(VK_SUCCESS, VK_SUCCESS, VK_SUBOPTIMAL_KHR,
            VK_SUBOPTIMAL_KHR, 108);
        runOriginCase(VK_SUCCESS, VK_SUCCESS, VK_ERROR_OUT_OF_HOST_MEMORY,
            VK_ERROR_DEVICE_LOST, 111);
        runOriginCase(VK_SUCCESS, VK_SUCCESS, VK_ERROR_OUT_OF_DEVICE_MEMORY,
            VK_ERROR_DEVICE_LOST, 112, true);
        // A1X-PREP FF-13: after a successful logical present, producer SYNC_FD
        // export failure retains the committed WSI lease/backing and never
        // substitutes ReleaseSwapchainImages.
        runOriginCase(VK_SUCCESS, VK_SUCCESS, VK_SUCCESS,
            VK_ERROR_DEVICE_LOST, 113, false, VK_ERROR_OUT_OF_HOST_MEMORY);

        // C12: a legacy transformation helper failure is internal, even though
        // it is represented by ls::vulkan_error.
        auto legacyFailureVulkan = presentHarness.createQueuePresentVulkan();
        SwapchainInfo legacyFailureInfo = virtualInfo;
        legacyFailureInfo.virtualized = false;
        assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
            .device = presentDevice,
            .queue = presentQueue,
            .queueFamily = presentHarness.endpoint().queueFamilyIndex,
            .swapchain = presentSwapchain,
            .vulkan = legacyFailureVulkan.get(),
            .backend = backend.get(),
            .devicePair = presentHarness.sameDevicePair(),
            .swapchainInfo = legacyFailureInfo,
            .deviceLifetimeIdentity = 109,
            .virtualized = false,
            .frameGenerationMultiplier = 2}));
        legacyFailureVulkan.reset();
        presentHarness.setQueueSubmitResult(VK_ERROR_INITIALIZATION_FAILED);
        const auto legacyFailurePresent = reinterpret_cast<PFN_vkQueuePresentKHR>(
            lsfgvk::layer::test::queuePresentEntrypoint());
        const VkSemaphore legacyWait =
            reinterpret_cast<VkSemaphore>(uintptr_t{0x8700});
        const uint32_t legacyIndex{};
        VkResult legacyPerSwapchain = VK_SUCCESS;
        const VkPresentInfoKHR legacyPresentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &legacyWait,
            .swapchainCount = 1,
            .pSwapchains = &presentSwapchain,
            .pImageIndices = &legacyIndex,
            .pResults = &legacyPerSwapchain};
        assert(legacyFailurePresent(presentQueue, &legacyPresentInfo)
            == VK_ERROR_DEVICE_LOST);
        assert(legacyPerSwapchain == VK_ERROR_DEVICE_LOST);
        assert(legacyPerSwapchain != VK_ERROR_INITIALIZATION_FAILED);
        presentHarness.setQueueSubmitResult(VK_SUCCESS);
        lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();

        // 4B1C closure: recovery dispatch follows the recorded EXT backend and
        // never dereferences a missing KHR function pointer.
        auto extOnlyVulkan = presentHarness.createQueuePresentVulkan(true);
        SwapchainInfo extOnlyInfo = legacyFailureInfo;
        extOnlyInfo.releaseBackend = SwapchainReleaseBackend::Ext;
        assert(lsfgvk::layer::test::installQueuePresentEntrypointHarness({
            .device = presentDevice,
            .queue = presentQueue,
            .queueFamily = presentHarness.endpoint().queueFamilyIndex,
            .swapchain = presentSwapchain,
            .vulkan = extOnlyVulkan.get(),
            .backend = backend.get(),
            .devicePair = presentHarness.sameDevicePair(),
            .swapchainInfo = extOnlyInfo,
            .deviceLifetimeIdentity = 110,
            .virtualized = false,
            .frameGenerationMultiplier = 1}));
        extOnlyVulkan.reset();
        const auto khrBeforeExtRecovery = presentHarness.observedKhrReleaseCalls();
        const auto extBeforeExtRecovery = presentHarness.observedExtReleaseCalls();
        assert(lsfgvk::layer::test::releaseD2PhysicalImageForTesting(
            presentDevice, presentSwapchain, 0) == VK_SUCCESS);
        assert(presentHarness.observedKhrReleaseCalls() == khrBeforeExtRecovery
            && presentHarness.observedExtReleaseCalls()
                == extBeforeExtRecovery + 1);
        lsfgvk::layer::test::uninstallQueuePresentEntrypointHarness();
    }

    // Tracking failure must not mutate a successful application create.
    capture.returnedFence = reinterpret_cast<VkFence>(uintptr_t{0x7300});
    const auto destroysBeforeTrackingFailure = capture.destroyCalls;
    assert(lsfgvk::layer::test::installDeviceEntrypointHarness(
        {device, downstream, 0}));
    const auto createUntracked = reinterpret_cast<PFN_vkCreateFence>(
        lsfgvk::layer::test::deviceEntrypoint("vkCreateFence"));
    VkFence untracked{};
    assert(createUntracked(device, &createInfo, nullptr, &untracked) == VK_SUCCESS
        && untracked == capture.returnedFence
        && capture.destroyCalls == destroysBeforeTrackingFailure);
    lsfgvk::layer::test::uninstallDeviceEntrypointHarness();
    return 0;
}
