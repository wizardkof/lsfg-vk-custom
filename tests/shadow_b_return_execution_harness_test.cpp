#include "shadow_b_return_execution_harness.hpp"
#include "d3b3_production_core_owner.hpp"
#include "d3b3_production_seams.hpp"
#include "d3b3_normal_present_adapter.hpp"
#include "d3b3_per_swapchain_runtime.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <cassert>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using lsfgvk::test::ShadowBReturnExecutionHarness;
using lsfgvk::test::ShadowHarnessFailurePoint;
using lsfgvk::test::ShadowAHarnessFailurePoint;

namespace lsfgvk::backend {
struct RuntimeOperationAuthorityTestAccess {
    static void setAReturnEpoch(ReturnedGeneratedOperation& operation, uint64_t epoch) {
        operation.aReturnSubmission.operationEpoch = epoch;
    }
    static void setPayloadEpoch(ReturnedGeneratedOperation& operation, uint64_t epoch) {
        operation.payload.operationEpoch = epoch;
    }
    static void setReturnedReady(ReturnedGeneratedOperation& operation, VkSemaphore value) {
        operation.returnedForGraphics = value;
    }
    static void setReturnedConsumed(ReturnedGeneratedOperation& operation, bool value) {
        operation.returnedConsumed = value;
    }
};
}

namespace vk {
struct RuntimeImageEndpointTestAccess {
    static RuntimeImageEndpoint make(RuntimeExchangeEndpoint endpoint,
            VkImage image, VkExtent2D extent, VkFormat format) {
        RuntimeImageEndpoint result;
        endpoint.FreeCommandBuffers = nullptr;
        endpoint.DestroyCommandPool = nullptr;
        endpoint.DestroyImage = nullptr;
        result.endpoint = std::move(endpoint);
        result.imageHandle = image;
        result.extent = extent;
        result.format = format;
        result.commandBuffers = {
            reinterpret_cast<VkCommandBuffer>(uintptr_t{0xA6A500})};
        return result;
    }
};
}

namespace {

void assertFdClosed(int fd) {
    errno = 0;
    assert(::fcntl(fd, F_GETFD) == -1);
    assert(errno == EBADF);
}

void assertFirewall(const ShadowBReturnExecutionHarness& harness) {
    assert(harness.productionBReturnCalls == 0);
    assert(harness.observed.calls == 0);
    assert(harness.aPhaseCalls == 0);
    assert(harness.returnedForGraphicsCalls == 0);
    assert(harness.terminalCalls == 0);
    assert(harness.fenceWaitCalls == 0);
    assert(harness.deviceIdleCalls == 0);
    assert(!harness.hasStagedPending());
}

void expectExerciseFailure(ShadowBReturnExecutionHarness& harness) {
    bool failed{};
    try {
        harness.exerciseResourceAndCommandPath();
    } catch (const std::exception&) {
        failed = true;
    }
    assert(failed);
    assertFirewall(harness);
}

void testQueueSubmitObserverInIsolation() {
    ShadowBReturnExecutionHarness harness;
    auto endpoint = harness.endpoint();
    const VkSemaphore wait = harness.generationReady;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkCommandBuffer command = harness.returnCommand;
    const VkSemaphore signal = harness.exportSignal;
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
        1, &wait, &stage, 1, &command, 1, &signal};

    // Direct observer self-test only: no production B-return function is linked
    // into this test target or called here.
    assert(endpoint.QueueSubmit(harness.queue, 1, &submit, harness.returnFence) == VK_SUCCESS);
    assert(harness.observed.calls == 1 && harness.observed.queue == harness.queue);
    assert(harness.observed.submitCount == 1);
    assert(harness.observed.waitCount == 1 && harness.observed.wait == wait);
    assert(harness.observed.waitStage == stage);
    assert(harness.observed.commandCount == 1 && harness.observed.command == command);
    assert(harness.observed.signalCount == 1 && harness.observed.signal == signal);
    assert(harness.observed.fence == harness.returnFence);

    harness.submitResult = VK_ERROR_INITIALIZATION_FAILED;
    assert(endpoint.QueueSubmit(harness.queue, 1, &submit, harness.returnFence)
        == VK_ERROR_INITIALIZATION_FAILED);
    harness.submitResult = VK_ERROR_DEVICE_LOST;
    assert(endpoint.QueueSubmit(harness.queue, 1, &submit, harness.returnFence)
        == VK_ERROR_DEVICE_LOST);
    assert(harness.productionBReturnCalls == 0);
    assert(harness.aPhaseCalls == 0);
}

void testCompleteResourceAndCommandSurface() {
    ShadowBReturnExecutionHarness harness;
    assert(harness.productionResourcePfnInventoryComplete());

    const auto generation = harness.generationEndpoint();
    const auto render = harness.renderEndpoint();
    assert(generation.bufferDevice.device == harness.device);
    assert(generation.physicalDevice == harness.generationPhysicalDevice);
    assert(generation.queue == harness.queue);
    assert(generation.queueFamilyIndex == ShadowBReturnExecutionHarness::generationQueueFamily);
    assert(render.bufferDevice.device == harness.renderDevice);
    assert(render.physicalDevice == harness.renderPhysicalDevice);
    assert(render.queue == harness.renderQueue);
    assert(render.queueFamilyIndex == ShadowBReturnExecutionHarness::renderQueueFamily);

    // B0C extends the same consumer endpoint with only the production A-return
    // import/readback surface. These PFNs remain deterministic test dispatch.
    assert(render.GetImageMemoryRequirements2 != nullptr);
    assert(render.AllocateMemory != nullptr);
    assert(render.BindImageMemory != nullptr);
    assert(render.CreateCommandPool != nullptr);
    assert(render.QueueSubmit != nullptr);
    assert(render.GetFenceStatus != nullptr);

    harness.exerciseResourceAndCommandPath();

    assert(harness.imageCreates.size() == 2);
    const auto& consumerCreate = harness.imageCreates[0];
    assert(consumerCreate.device == harness.renderDevice);
    assert(consumerCreate.image == harness.consumerProbeImage);
    assert(!consumerCreate.explicitModifier);
    assert(consumerCreate.externalHandleTypes
        == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assert(consumerCreate.format == harness.returnFormat);
    assert(consumerCreate.extent.width == harness.returnExtent.width);
    assert(consumerCreate.extent.height == harness.returnExtent.height);

    const auto& returnCreate = harness.imageCreates[1];
    assert(returnCreate.device == harness.device);
    assert(returnCreate.image == harness.returnImage);
    assert(returnCreate.explicitModifier);
    assert(returnCreate.modifierPlaneCount == 1);
    assert(returnCreate.plane.offset == harness.returnPlane.offset);
    assert(returnCreate.plane.rowPitch == harness.returnPlane.rowPitch);
    assert(returnCreate.externalHandleTypes
        == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assert(returnCreate.usage
        == (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    assert(returnCreate.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT);

    assert(harness.formatPropertyQueries == 4);
    assert(harness.imageFormatPropertyQueries == 2);
    assert(harness.modifierQueries == 2);
    assert(harness.subresourceLayoutQueries == 2);
    assert((harness.destroyedImages
        == std::vector<VkImage>{harness.consumerProbeImage, harness.returnImage}));

    assert(harness.memoryRequirements.calls == 1);
    assert(harness.memoryRequirements.image == harness.returnImage);
    assert(harness.memoryRequirements.size
        == ShadowBReturnExecutionHarness::returnAllocationSize);
    assert(harness.memoryRequirements.alignment
        == ShadowBReturnExecutionHarness::returnAllocationAlignment);
    assert(harness.memoryRequirements.memoryTypeBits
        == ShadowBReturnExecutionHarness::returnMemoryTypeBits);
    assert(harness.memoryAllocation.calls == 1);
    assert(harness.memoryAllocation.size
        == ShadowBReturnExecutionHarness::returnAllocationSize);
    assert(harness.memoryAllocation.memoryTypeIndex == 0);
    assert(harness.memoryAllocation.exportHandleTypes
        == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assert(!harness.memoryAllocation.dedicated);
    assert(harness.memoryAllocation.memory == harness.returnMemory);
    assert(harness.imageBind.calls == 1);
    assert(harness.imageBind.image == harness.returnImage);
    assert(harness.imageBind.memory == harness.returnMemory);
    assert(harness.imageBind.offset == 0);
    assert(harness.freedMemories == 1);
    assert(harness.memoryFdExports == 1);
    assert(harness.lastExportedMemory == harness.returnMemory);
    assert(harness.lastMemoryExportHandleType
        == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    assertFdClosed(harness.lastMemoryExportFd);

    assert(harness.commandPoolObservation.createCalls == 1);
    assert(harness.commandPoolObservation.destroyCalls == 1);
    assert(harness.commandPoolObservation.pool == harness.returnCommandPool);
    assert(harness.commandPoolObservation.queueFamilyIndex
        == ShadowBReturnExecutionHarness::generationQueueFamily);
    assert(harness.commandAllocation.calls == 1);
    assert(harness.commandAllocation.pool == harness.returnCommandPool);
    assert(harness.commandAllocation.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    assert(harness.commandAllocation.count == 1);
    assert(harness.commandAllocation.command == harness.returnCommand);
    assert(harness.commandAllocation.freeCalls == 1);
    assert(harness.commandAllocation.freedCount == 1);
    assert(harness.commandBeginEnd.beginCalls == 1);
    assert(harness.commandBeginEnd.endCalls == 1);
    assert(harness.commandBeginEnd.beginCommand == harness.returnCommand);
    assert(harness.commandBeginEnd.endCommand == harness.returnCommand);
    assert(harness.commandBeginEnd.beginFlags == 0);

    assert(harness.barriers.size() == 2);
    const auto& toDestination = harness.barriers[0];
    assert(toDestination.command == harness.returnCommand);
    assert(toDestination.sourceStage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    assert(toDestination.destinationStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(toDestination.memoryBarrierCount == 0);
    assert(toDestination.bufferBarrierCount == 0);
    assert(toDestination.imageBarrierCount == 1);
    assert(toDestination.imageBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    assert(toDestination.imageBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(toDestination.imageBarrier.srcAccessMask == 0);
    assert(toDestination.imageBarrier.dstAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(toDestination.imageBarrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    assert(toDestination.imageBarrier.dstQueueFamilyIndex
        == ShadowBReturnExecutionHarness::generationQueueFamily);
    assert(toDestination.imageBarrier.image == harness.returnImage);

    const auto& release = harness.barriers[1];
    assert(release.command == harness.returnCommand);
    assert(release.sourceStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(release.destinationStage == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    assert(release.imageBarrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(release.imageBarrier.dstAccessMask == 0);
    assert(release.imageBarrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(release.imageBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(release.imageBarrier.srcQueueFamilyIndex
        == ShadowBReturnExecutionHarness::generationQueueFamily);
    assert(release.imageBarrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT);
    assert(release.imageBarrier.image == harness.returnImage);

    assert(harness.imageCopy.calls == 1);
    assert(harness.imageCopy.command == harness.returnCommand);
    assert(harness.imageCopy.source == harness.generatedSourceImage);
    assert(harness.imageCopy.sourceLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(harness.imageCopy.destination == harness.returnImage);
    assert(harness.imageCopy.destinationLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(harness.imageCopy.regionCount == 1);
    assert(harness.imageCopy.region.extent.width == harness.returnExtent.width);
    assert(harness.imageCopy.region.extent.height == harness.returnExtent.height);
    assert(harness.imageCopy.region.extent.depth == 1);
    assert(harness.imageCopy.region.srcSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
    assert(harness.imageCopy.region.srcSubresource.layerCount == 1);
    assert(harness.imageCopy.region.dstSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
    assert(harness.imageCopy.region.dstSubresource.layerCount == 1);
    assert((harness.commandTrace == std::vector<std::string>{
        "BEGIN_CB", "BARRIER", "COPY_IMAGE", "BARRIER", "END_CB"}));

    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    assert(generation.CreateFence(harness.device, &fenceInfo, nullptr, &fence)
        == VK_SUCCESS);
    assert(fence == harness.returnFence);
    generation.DestroyFence(harness.device, fence, nullptr);
    assert(harness.fenceCreateCalls == 1 && harness.fenceDestroyCalls == 1);

    {
        auto semaphore = vk::createExportableSyncFdSemaphore(generation.semaphoreDevice);
        assert(semaphore.handle() == harness.exportSignal);
    }
    assert(harness.semaphoreCreateCalls == 1);
    assert(harness.semaphoreDestroyCalls == 1);

    assertFirewall(harness);
}

void testFailureCleanup() {
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::CONSUMER_IMAGE_CREATE;
        expectExerciseFailure(harness);
        assert(harness.imageCreates.size() == 1);
        assert(harness.destroyedImages.empty());
        assert(harness.memoryAllocation.calls == 0);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::RETURN_IMAGE_CREATE;
        expectExerciseFailure(harness);
        assert(harness.imageCreates.size() == 2);
        assert((harness.destroyedImages
            == std::vector<VkImage>{harness.consumerProbeImage}));
        assert(harness.memoryAllocation.calls == 0);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::MEMORY_ALLOCATION;
        expectExerciseFailure(harness);
        assert(harness.memoryAllocation.calls == 1);
        assert(harness.memoryAllocation.result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        assert((harness.destroyedImages
            == std::vector<VkImage>{harness.consumerProbeImage, harness.returnImage}));
        assert(harness.freedMemories == 0);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::IMAGE_BIND;
        expectExerciseFailure(harness);
        assert(harness.imageBind.calls == 1);
        assert(harness.imageBind.result == VK_ERROR_INVALID_EXTERNAL_HANDLE);
        assert((harness.destroyedImages
            == std::vector<VkImage>{harness.consumerProbeImage, harness.returnImage}));
        assert(harness.freedMemories == 1);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::COMMAND_POOL_CREATE;
        expectExerciseFailure(harness);
        assert(harness.commandPoolObservation.createCalls == 1);
        assert(harness.commandPoolObservation.destroyCalls == 0);
        assert(harness.commandAllocation.calls == 0);
        assert(harness.freedMemories == 1);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::COMMAND_BUFFER_ALLOCATION;
        expectExerciseFailure(harness);
        assert(harness.commandPoolObservation.destroyCalls == 1);
        assert(harness.commandAllocation.calls == 1);
        assert(harness.commandAllocation.freeCalls == 0);
        assert(harness.freedMemories == 1);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER;
        expectExerciseFailure(harness);
        assert(harness.commandBeginEnd.beginCalls == 1);
        assert(harness.commandBeginEnd.endCalls == 0);
        assert(harness.commandAllocation.freeCalls == 1);
        assert(harness.commandPoolObservation.destroyCalls == 1);
        assert(harness.freedMemories == 1);
    }
    {
        ShadowBReturnExecutionHarness harness;
        harness.failurePoint = ShadowHarnessFailurePoint::END_COMMAND_BUFFER;
        expectExerciseFailure(harness);
        assert(harness.commandBeginEnd.beginCalls == 1);
        assert(harness.commandBeginEnd.endCalls == 1);
        assert(harness.barriers.size() == 2);
        assert(harness.imageCopy.calls == 1);
        assert(harness.commandAllocation.freeCalls == 1);
        assert(harness.commandPoolObservation.destroyCalls == 1);
        assert(harness.freedMemories == 1);
    }
}

void testSyncFdPayloadMoveRegression() {
    ShadowBReturnExecutionHarness harness;

    int moveConstructedFd{-1};
    {
        auto payload = harness.exportTestPayload();
        assert(payload.valid() && !payload.sentinel());
        moveConstructedFd = payload.nativeFd();
        auto moved = std::move(payload);
        assert(!payload.valid());
        assert(payload.nativeFd() == -1);
        assert(moved.valid());
        assert(moved.nativeFd() == moveConstructedFd);
        assert(::fcntl(moveConstructedFd, F_GETFD) != -1);
    }
    assertFdClosed(moveConstructedFd);

    int replacedFd{-1};
    int assignedFd{-1};
    {
        auto destination = harness.exportTestPayload();
        replacedFd = destination.nativeFd();
        auto source = harness.exportTestPayload();
        assignedFd = source.nativeFd();
        destination = std::move(source);
        assert(!source.valid());
        assert(source.nativeFd() == -1);
        assert(destination.valid());
        assert(destination.nativeFd() == assignedFd);
        assertFdClosed(replacedFd);
        assert(::fcntl(assignedFd, F_GETFD) != -1);
    }
    assertFdClosed(assignedFd);

    vk::SyncFdPayload invalid;
    auto movedInvalid = std::move(invalid);
    assert(!invalid.valid() && !movedInvalid.valid());
    vk::SyncFdPayload assignedInvalid;
    assignedInvalid = std::move(movedInvalid);
    assert(!movedInvalid.valid() && !assignedInvalid.valid());

    int chainedFd{-1};
    {
        auto source = harness.exportTestPayload();
        chainedFd = source.nativeFd();
        auto middle = std::move(source);
        auto destination = std::move(middle);
        assert(!source.valid() && !middle.valid());
        assert(destination.valid());
        assert(destination.nativeFd() == chainedFd);
        assert(::fcntl(chainedFd, F_GETFD) != -1);
    }
    assertFdClosed(chainedFd);

    harness.exportResult = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    bool exportFailed{};
    try {
        static_cast<void>(harness.exportTestPayload());
    } catch (const vk::ExternalSemaphoreError&) {
        exportFailed = true;
    }
    assert(exportFailed);
    assert(harness.exportCalls == 5);
    assert(harness.lastExportedSemaphore == harness.exportSignal);
    assertFirewall(harness);
}

void testIndependentFenceRetirement() {
    ShadowBReturnExecutionHarness harness;
    assert(!harness.fenceRetired);
    assert(harness.fenceWaitResult == VK_TIMEOUT);
    harness.retireFence();
    assert(harness.fenceRetired);
    assert(harness.fenceWaitResult == VK_SUCCESS);
    assertFirewall(harness);
}

void testRealB0B3PendingStaging() {
    ShadowBReturnExecutionHarness harness;

    // C1B3A3-26: an empty capability is not promoted into fixture identities.
    lsfgvk::backend::RuntimeGenerateDiagnosticPending missing;
    bool missingRejected{};
    try {
        harness.stageProductionPending(std::move(missing), {}, harness.generationEndpoint());
    } catch (const std::invalid_argument&) {
        missingRejected = true;
    }
    assert(missingRejected && !harness.hasStagedPending());

    auto backend = harness.createB0B3Instance();
    constexpr VkExtent2D productionExtent{64, 64};
    auto& session = backend->openRuntimeGenerateSession(
        productionExtent, VK_FORMAT_B8G8R8A8_UNORM, 0, 1.0F, false,
        lsfgvk::backend::RuntimeGenerateMode::SerialReusable);

    // The session constructor has one diagnostic initialization submit/wait.
    // It is not part of B0B3 and is kept separate from the two shadow submits.
    assert(harness.frontSubmitCount() == 1);
    assert(harness.frontFenceWaitCount() == 1);
    const auto hostWaitsBeforeB0B3 = harness.frontFenceWaitCount();

    constexpr uint64_t frameB = static_cast<uint64_t>('B');
    constexpr uint64_t frameC = static_cast<uint64_t>('C');
    const auto transportImage = reinterpret_cast<VkImage>(uintptr_t{0xC100});
    auto inputPayload = harness.exportTestPayload();
    assert(inputPayload.valid());
    const auto ingest = backend->submitShadowTemporalIngest(session, transportImage,
        std::move(inputPayload), lsfgvk::backend::TemporalSourceSlot::Slot0, frameC);
    assert(!inputPayload.valid());
    assert(ingest.submitAccepted);
    assert(ingest.frameId == frameC);
    assert(ingest.slot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(ingest.epoch != 0);
    assert(ingest.shadowSubmitCount == 1);
    assert(ingest.shadowGenerateSubmitCount == 0);
    assert(harness.frontImportedFdCount() == 1);

    const lsfgvk::backend::RuntimeTemporalPairIdentity pair{
        .olderSlot = lsfgvk::backend::TemporalSourceSlot::Slot1,
        .newerSlot = lsfgvk::backend::TemporalSourceSlot::Slot0,
        .olderFrameId = frameB,
        .newerFrameId = frameC,
        .generationId = ingest.epoch
    };
    const auto generated = backend->submitShadowPrepassGenerate(session, pair);
    assert(generated.submitAccepted);
    assert(!generated.executionRetired);
    assert(!generated.generationReadyReusable);
    assert(generated.totalShadowSubmitCount == 2);
    assert(generated.generationId == ingest.epoch);
    assert(generated.olderFrameId == frameB && generated.newerFrameId == frameC);
    assert(generated.olderSlot == lsfgvk::backend::TemporalSourceSlot::Slot1);
    assert(generated.newerSlot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(generated.signalSemaphore != VK_NULL_HANDLE);
    assert(generated.generatedImage != VK_NULL_HANDLE);
    assert(generated.fence != VK_NULL_HANDLE);
    assert(harness.frontSubmitCount() == 3);
    assert(harness.frontFenceWaitCount() == hostWaitsBeforeB0B3);

    const auto& ingestSubmit = harness.frontSubmit(1);
    const auto& generateSubmit = harness.frontSubmit(2);
    assert(ingestSubmit.queue == harness.queue);
    assert(ingestSubmit.waitCount == 1);
    assert(ingestSubmit.waitStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(ingestSubmit.signalCount == 1);
    assert(generateSubmit.queue == harness.queue);
    assert(generateSubmit.waitCount == 1);
    assert(generateSubmit.wait == ingest.signalSemaphore);
    assert(generateSubmit.waitStage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    assert(generateSubmit.signalCount == 1);
    assert(generateSubmit.signal == generated.signalSemaphore);
    assert(generateSubmit.fence == generated.fence);

    const auto productionEndpoint = backend->runtimeExchangeEndpoint();
    assert(productionEndpoint.bufferDevice.device == harness.device);
    assert(productionEndpoint.queue == harness.queue);
    assert(productionEndpoint.queueFamilyIndex
        == ShadowBReturnExecutionHarness::generationQueueFamily);

    const auto producerSnapshot = backend->inspectShadowPrepassGenerate(session);
    auto actualPending = backend->takeShadowPrepassGeneratePending(session);
    assert(actualPending.has_value() && actualPending->valid());
    assert(actualPending->temporalPair().olderFrameId == frameB);
    assert(actualPending->temporalPair().newerFrameId == frameC);
    assert(actualPending->identity() == ingest.epoch);
    assert(actualPending->readinessSemaphore() == generated.signalSemaphore);
    assert(actualPending->imageHandle() == generated.generatedImage);
    assert(actualPending->extentValue().width == productionExtent.width);
    assert(actualPending->extentValue().height == productionExtent.height);
    assert(actualPending->formatValue() == VK_FORMAT_R8G8B8A8_UNORM);
    assert(actualPending->queueFamily() == productionEndpoint.queueFamilyIndex);
    assert(actualPending->sourceReadRetirementFence() == generated.fence);
    assert(!actualPending->transportConsumedForTesting());
    assert(!actualPending->retirementAuthorityIssuedForTesting());
    assert(!actualPending->operationRetiredForTesting());

    bool sourceSeamDoubleTakeRejected{};
    try {
        static_cast<void>(backend->takeShadowPrepassGeneratePending(session));
    } catch (const std::logic_error&) {
        sourceSeamDoubleTakeRejected = true;
    }
    assert(sourceSeamDoubleTakeRejected);

    harness.stageProductionPending(
        std::move(*actualPending), producerSnapshot, productionEndpoint);
    assert(!actualPending->valid());
    assert(harness.hasStagedPending());

    bool movedSourceConsumeRejected{};
    try {
        actualPending->consumeTransport();
    } catch (const std::logic_error&) {
        movedSourceConsumeRejected = true;
    }
    assert(movedSourceConsumeRejected);

    const auto& staged = harness.stagedView();
    assert(staged.pair.olderFrameId == frameB);
    assert(staged.pair.newerFrameId == frameC);
    assert(staged.pair.olderSlot == lsfgvk::backend::TemporalSourceSlot::Slot1);
    assert(staged.pair.newerSlot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(staged.generation == ingest.epoch);
    assert(staged.generationReady == generated.signalSemaphore);
    assert(staged.futureWaitSemaphore == staged.generationReady);
    assert(staged.generatedImage == generated.generatedImage);
    assert(staged.futureCopySource == staged.generatedImage);
    assert(staged.extent.width == productionExtent.width);
    assert(staged.extent.height == productionExtent.height);
    assert(staged.format == VK_FORMAT_R8G8B8A8_UNORM);
    assert(staged.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(staged.device == productionEndpoint.bufferDevice.device);
    assert(staged.queue == productionEndpoint.queue);
    assert(staged.futureSubmitQueue == staged.queue);
    assert(staged.queueFamily == productionEndpoint.queueFamilyIndex);
    assert(staged.futureCommandPoolFamily == staged.queueFamily);
    assert(staged.sourceReadRetirementFence == generated.fence);
    assert(staged.sourceReadAuthorityPresent);
    assert(staged.outputAuthorityLive);
    assert(!staged.transportConsumed);
    assert(!staged.retirementAuthorityIssued);
    assert(!staged.operationRetired);
    assert(staged.generationReadySignalOutstanding);
    assert(!staged.generationReadyWaitSubmitted);
    assert(!staged.generationReadyWaitRetired);
    assert(!staged.generationReadyReusable);
    assert(staged.futureReturnBackingImage == harness.returnImage);
    assert(staged.futureReturnCommandPool == harness.returnCommandPool);
    assert(staged.futureReturnCommand == harness.returnCommand);
    assert(staged.futureExportSignal == harness.exportSignal);
    assert(staged.futureShadowRealFence == harness.returnFence);
    assert(staged.futureShadowRealFence
        == lsfgvk::layer::returnSubmissionFence(
            lsfgvk::layer::ReturnSubmissionFencePolicy::SHADOW_REAL,
            harness.returnFence));

    harness.validateStagedReturnConfiguration(
        productionEndpoint, staged.extent, staged.format);
    auto mismatchedEndpoint = productionEndpoint;
    ++mismatchedEndpoint.queueFamilyIndex;
    bool familyMismatchRejected{};
    try {
        harness.validateStagedReturnConfiguration(
            mismatchedEndpoint, staged.extent, staged.format);
    } catch (const std::invalid_argument&) {
        familyMismatchRejected = true;
    }
    assert(familyMismatchRejected);
    auto wrongSessionEndpoint = productionEndpoint;
    wrongSessionEndpoint.queue = reinterpret_cast<VkQueue>(uintptr_t{0xDEAD});
    bool sessionMismatchRejected{};
    try {
        harness.validateStagedReturnConfiguration(
            wrongSessionEndpoint, staged.extent, staged.format);
    } catch (const std::invalid_argument&) {
        sessionMismatchRejected = true;
    }
    assert(sessionMismatchRejected);
    bool extentMismatchRejected{};
    try {
        harness.validateStagedReturnConfiguration(
            productionEndpoint, {staged.extent.width + 1, staged.extent.height},
            staged.format);
    } catch (const std::invalid_argument&) {
        extentMismatchRejected = true;
    }
    assert(extentMismatchRejected);
    bool formatMismatchRejected{};
    try {
        harness.validateStagedReturnConfiguration(
            productionEndpoint, staged.extent, VK_FORMAT_R16G16B16A16_SFLOAT);
    } catch (const std::invalid_argument&) {
        formatMismatchRejected = true;
    }
    assert(formatMismatchRejected);

    bool doubleStageRejected{};
    try {
        harness.stageProductionPending(
            std::move(*actualPending), producerSnapshot, productionEndpoint);
    } catch (const std::logic_error&) {
        doubleStageRejected = true;
    }
    assert(doubleStageRejected);

    // A3 hard stop: staging did not dispatch, export, wait, import, or advance A.
    assert(harness.frontSubmitCount() == 3);
    assert(harness.frontFenceWaitCount() == hostWaitsBeforeB0B3);
    assert(harness.observed.calls == 0);
    assert(harness.productionBReturnCalls == 0);
    assert(harness.productionBReturnExportCalls == 0);
    assert(harness.aPhaseCalls == 0);
    assert(harness.aImportCalls == 0);
    assert(harness.aQueueSubmitCalls == 0);
    assert(harness.foreignReadbackPendingCalls == 0);
    assert(harness.returnedForGraphicsCalls == 0);
    assert(harness.terminalCalls == 0);
    assert(harness.fenceWaitCalls == 0);
    assert(harness.deviceIdleCalls == 0);

    {
        auto soleOwner = harness.takeStaged();
        assert(soleOwner.valid());
        assert(!harness.hasStagedPending());
        assert(soleOwner.identity() == ingest.epoch);
        assert(soleOwner.readinessSemaphore() == generated.signalSemaphore);
        bool doubleTakeRejected{};
        try {
            static_cast<void>(harness.takeStaged());
        } catch (const std::logic_error&) {
            doubleTakeRejected = true;
        }
        assert(doubleTakeRejected);
    }

    assertFirewall(harness);
}

void testProductionWarmupNoSignalAndBCTransition() {
    ShadowBReturnExecutionHarness harness;
    auto backend = harness.createB0B3Instance();
    auto& session = backend->openRuntimeGenerateSession(
        {64, 64}, VK_FORMAT_B8G8R8A8_UNORM, 0, 1.0F, false,
        lsfgvk::backend::RuntimeGenerateMode::SerialReusable);
    const auto hostWaitsBeforeWarmup = harness.frontFenceWaitCount();
    const auto transport = reinterpret_cast<VkImage>(uintptr_t{0xC100});

    const auto warmup = [&](uint64_t frame, auto slot) {
        auto payload = harness.exportTestPayload();
        const auto submitted = backend->submitRuntimeIngest(session, transport,
            std::move(payload), slot, frame,
            lsfgvk::backend::RuntimeIngestIntent::WARMUP_TEMPORAL);
        assert(submitted.submitAccepted && submitted.signalSemaphore == VK_NULL_HANDLE);
        assert(submitted.ingestReadyReusable && !submitted.frameTransportReusable);
        const auto& observed = harness.frontSubmit(harness.frontSubmitCount() - 1);
        assert(observed.waitCount == 1
            && observed.waitStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
        assert(observed.signalCount == 0);
        assert(backend->tryRetireRuntimeIngest(session)
            == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED);
        const auto retired = backend->inspectRuntimeIngest(session);
        assert(retired.frameTransportReusable && retired.ingestReadyReusable);
    };

    warmup(static_cast<uint64_t>('A'),
        lsfgvk::backend::TemporalSourceSlot::Slot0);
    warmup(static_cast<uint64_t>('B'),
        lsfgvk::backend::TemporalSourceSlot::Slot1);

    auto payload = harness.exportTestPayload();
    const auto c = backend->submitRuntimeIngest(session, transport, std::move(payload),
        lsfgvk::backend::TemporalSourceSlot::Slot0, static_cast<uint64_t>('C'),
        lsfgvk::backend::RuntimeIngestIntent::GENERATE_SOURCE);
    assert(c.signalSemaphore != VK_NULL_HANDLE && !c.ingestReadyReusable);
    const auto generated = backend->submitRuntimePrepassGenerate(session, {
        .olderSlot = lsfgvk::backend::TemporalSourceSlot::Slot1,
        .newerSlot = lsfgvk::backend::TemporalSourceSlot::Slot0,
        .olderFrameId = static_cast<uint64_t>('B'),
        .newerFrameId = static_cast<uint64_t>('C'),
        .generationId = c.epoch});
    const auto& generateSubmit = harness.frontSubmit(harness.frontSubmitCount() - 1);
    assert(generateSubmit.wait == c.signalSemaphore);
    assert(generateSubmit.waitStage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    assert(generated.olderFrameId == static_cast<uint64_t>('B')
        && generated.newerFrameId == static_cast<uint64_t>('C'));
    assert(harness.frontFenceWaitCount() == hostWaitsBeforeWarmup);
}

struct RealShadowPairFixture {
    ShadowBReturnExecutionHarness harness;
    std::unique_ptr<lsfgvk::backend::Instance> backend;
    lsfgvk::backend::RuntimeGenerateSession* backendSession{};
    lsfgvk::backend::RuntimeShadowGenerateSnapshot generated{};
    uint32_t hostWaitsBeforeGraph{};
    std::optional<std::pair<uint64_t, lsfgvk::backend::TemporalSourceSlot>> preparedIngest;
    uint32_t productionWarmupIngests{};
    uint32_t productionGenerateSourceIngests{};

    explicit RealShadowPairFixture(bool prepareInitialPair = true) {
        backend = harness.createB0B3Instance();
        backendSession = &backend->openRuntimeGenerateSession(
            {64, 64}, VK_FORMAT_B8G8R8A8_UNORM, 0, 1.0F, false,
            lsfgvk::backend::RuntimeGenerateMode::SerialReusable);
        assert(harness.frontSubmitCount() == 1);
        assert(harness.frontFenceWaitCount() == 1);
        hostWaitsBeforeGraph = harness.frontFenceWaitCount();

        if (prepareInitialPair)
            preparePair(static_cast<uint64_t>('B'), static_cast<uint64_t>('C'),
                lsfgvk::backend::TemporalSourceSlot::Slot1,
                lsfgvk::backend::TemporalSourceSlot::Slot0);
    }

    void preparePair(uint64_t olderFrameId, uint64_t newerFrameId,
            lsfgvk::backend::TemporalSourceSlot olderSlot,
            lsfgvk::backend::TemporalSourceSlot newerSlot) {
        ingestCurrent(newerFrameId, newerSlot);
        generateCurrent(olderFrameId, olderSlot, newerFrameId, newerSlot);
    }

    void ingestCurrent(uint64_t newerFrameId,
            lsfgvk::backend::TemporalSourceSlot newerSlot) {
        auto payload = harness.exportTestPayload();
        const auto ingest = backend->submitRuntimeIngest(*backendSession,
            reinterpret_cast<VkImage>(uintptr_t{0xC100}), std::move(payload),
            newerSlot, newerFrameId,
            lsfgvk::backend::RuntimeIngestIntent::GENERATE_SOURCE);
        assert(ingest.submitAccepted && !payload.valid());
        ++productionGenerateSourceIngests;
        preparedIngest = std::pair<uint64_t, lsfgvk::backend::TemporalSourceSlot>{
            ingest.epoch, newerSlot};
    }

    void ingestWarmup(uint64_t frameId, lsfgvk::backend::TemporalSourceSlot slot) {
        auto payload = harness.exportTestPayload();
        const auto ingest = backend->submitRuntimeIngest(*backendSession,
            reinterpret_cast<VkImage>(uintptr_t{0xC100}), std::move(payload),
            slot, frameId, lsfgvk::backend::RuntimeIngestIntent::WARMUP_TEMPORAL);
        assert(ingest.submitAccepted && ingest.signalSemaphore == VK_NULL_HANDLE
            && ingest.ingestReadyReusable);
        ++productionWarmupIngests;
    }

    void generateCurrent(uint64_t olderFrameId,
            lsfgvk::backend::TemporalSourceSlot olderSlot,
            uint64_t newerFrameId,
            lsfgvk::backend::TemporalSourceSlot newerSlot) {
        assert(preparedIngest.has_value());
        const lsfgvk::backend::RuntimeTemporalPairIdentity pair{
            .olderSlot = olderSlot,
            .newerSlot = newerSlot,
            .olderFrameId = olderFrameId,
            .newerFrameId = newerFrameId,
            .generationId = preparedIngest->first};
        generated = backend->submitRuntimePrepassGenerate(*backendSession, pair);
        assert(generated.submitAccepted && generated.generationReadySignalOutstanding);
        assert(!generated.generationReadyWaitSubmitted
            && !generated.generationReadyWaitRetired
            && !generated.generationReadyReusable);
        const auto producer = backend->inspectRuntimePrepassGenerate(*backendSession);
        auto pending = backend->takeRuntimePrepassGeneratePending(*backendSession);
        assert(pending && pending->valid());
        harness.stageProductionPending(
            std::move(*pending), producer, backend->runtimeExchangeEndpoint());
        assert(!pending->valid());
        preparedIngest.reset();
    }

    [[nodiscard]] lsfgvk::backend::RuntimeGenerateDiagnosticPending take() {
        return harness.takeStagedForExecution();
    }

    void retireSourceReads() {
        harness.retireSourceFence();
        assert(backend->tryRetireRuntimePrepassGenerate(*backendSession)
            == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
        assert(backend->inspectRuntimePrepassGenerate(*backendSession).executionRetired);
    }
};

void testCompleteOnePairNonTerminalProductionGraph() {
    RealShadowPairFixture fixture;
    auto& harness = fixture.harness;
    const auto staged = harness.stagedView();
    auto pending = fixture.take();
    assert(!harness.hasStagedPending() && pending.valid());

    lsfgvk::layer::GeneratedOutputReturnSession returnSession(
        lsfgvk::layer::ProductionReturnExecution{}, harness.devicePair(),
        harness.endpoint(), harness.renderEndpoint(), true);

    auto bPending = harness.submitProductionBReturn(
        returnSession, std::move(pending), *fixture.backend, *fixture.backendSession);
    assert(!pending.valid() && bPending.valid());
    assert(harness.productionBReturnCalls == 1);
    assert(returnSession.gpuChainedStateForTesting()
        == lsfgvk::layer::GpuChainedReturnState::RETURN_B_SUBMITTED);
    assert(bPending.identity().olderFrameId == static_cast<uint64_t>('B'));
    assert(bPending.identity().newerFrameId == static_cast<uint64_t>('C'));
    assert(bPending.identity().olderSlot == lsfgvk::backend::TemporalSourceSlot::Slot1);
    assert(bPending.identity().newerSlot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(bPending.identity().generationId == staged.generation);
    assert(bPending.payload().valid());
    assert(bPending.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);

    assert(harness.observed.calls == 1);
    assert(harness.observed.queue == staged.queue);
    assert(harness.observed.submitCount == 1);
    assert(harness.observed.waitCount == 1);
    assert(harness.observed.wait == staged.generationReady);
    assert(harness.observed.waitStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(harness.observed.commandCount == 1);
    assert(harness.observed.command == harness.returnCommand);
    assert(harness.observed.signalCount == 1);
    assert(harness.observed.signal == harness.exportSignal);
    assert(harness.observed.fence == harness.returnFence);
    assert(harness.observed.fence == bPending.bReturnAuthority().fenceHandle());
    assert(harness.productionBReturnExportCalls == 1);
    assert(harness.lastExportedSemaphore == harness.observed.signal);

    assert(harness.commandTrace == (std::vector<std::string>{
        "BEGIN_CB", "BARRIER", "COPY_IMAGE", "BARRIER", "END_CB"}));
    assert(harness.barriers.size() == 2);
    assert(harness.imageCopy.calls == 1);
    assert(harness.imageCopy.source == staged.generatedImage);
    assert(harness.imageCopy.destination == harness.returnImage);
    assert(harness.imageCopy.region.extent.width == staged.extent.width);
    assert(harness.imageCopy.region.extent.height == staged.extent.height);
    assert(harness.barriers.front().imageBarrier.image == harness.returnImage);
    assert(harness.barriers.front().imageBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    assert(harness.barriers.front().imageBarrier.newLayout
        == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(harness.barriers.back().imageBarrier.srcQueueFamilyIndex
        == ShadowBReturnExecutionHarness::generationQueueFamily);
    assert(harness.barriers.back().imageBarrier.dstQueueFamilyIndex
        == VK_QUEUE_FAMILY_FOREIGN_EXT);

    auto afterB = fixture.backend->inspectShadowPrepassGenerate(*fixture.backendSession);
    assert(afterB.bReturnSubmitCount == 1 && afterB.aReturnSubmitCount == 0);
    assert(afterB.generationReadyWaitSubmitted);
    assert(!afterB.generationReadyWaitRetired && !afterB.generationReadyReusable);
    assert(afterB.generatedOutputLive && !afterB.generatedOutputRetired);
    assert(harness.frontFenceWaitCount() == fixture.hostWaitsBeforeGraph);
    assert(harness.sourceFenceWaitCalls == 0 && harness.fenceWaitCalls == 0);
    assert(harness.deviceIdleCalls == 0);

    auto operation = harness.completeProductionAReturn(
        returnSession, std::move(bPending), *fixture.backend, *fixture.backendSession);
    assert(!bPending.valid() && operation.valid());
    assert(harness.aPhaseCalls == 1 && harness.aImportCalls == 1);
    assert(harness.aObserved.calls == 1 && harness.aQueueSubmitCalls == 1);
    assert(harness.aObserved.queue == harness.renderQueue);
    assert(harness.aObserved.waitCount == 1);
    assert(harness.aObserved.wait == harness.aImportedWait);
    assert(harness.aObserved.wait == operation.payloadAuthority().semaphoreHandle());
    assert(harness.aObserved.waitStage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(harness.aObserved.commandCount == 1);
    assert(harness.aObserved.command == harness.aCommand);
    assert(harness.aObserved.signalCount == 1);
    assert(harness.aObserved.signal == harness.returnedForGraphics);
    assert(harness.aObserved.signal == operation.returnedReady());
    assert(harness.aObserved.fence == harness.aReturnFence);
    assert(harness.aObserved.fence == operation.aReturnAuthority().fenceHandle());
    assert(operation.aReturnPending().retirementFence() == harness.aObserved.fence);
    assert(operation.aReturnPending().importedWaitSemaphore() == harness.aObserved.wait);
    assert(operation.aReturnPending().returnedSignalSemaphore() == harness.aObserved.signal);
    assert(operation.aReturnPending().submissionAccepted());
    assert(operation.aReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
    assert(operation.payloadAuthority().state()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_SUBMITTED);
    assert(operation.returnedForGraphicsOutstanding());
    assert(operation.generatedOutputLive());

    assert(harness.aSemaphoreImport.calls == 1);
    assert(harness.aSemaphoreImport.semaphore == harness.aImportedWait);
    assert(harness.aSemaphoreImport.flags == VK_SEMAPHORE_IMPORT_TEMPORARY_BIT);
    assert(harness.aSemaphoreImport.handleType
        == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);
    const std::vector<std::string> expectedACommandTrace{
        "BEGIN_CB", "BARRIER", "COPY_IMAGE_TO_BUFFER", "BARRIER", "BARRIER", "END_CB"};
    assert(harness.aCommandTrace == expectedACommandTrace);
    assert(harness.aImageToBuffer.calls == 1);
    assert(harness.aImageToBuffer.source == harness.aImportedImage);
    assert(harness.aImageToBuffer.destination == harness.aStagingBuffer);
    assert(harness.aImageToBuffer.region.imageExtent.width == staged.extent.width);
    assert(harness.aImageToBuffer.region.imageExtent.height == staged.extent.height);
    assert(harness.aBarriers.size() == 3);
    assert(harness.aBarriers[0].imageBarrier.srcQueueFamilyIndex
        == VK_QUEUE_FAMILY_FOREIGN_EXT);
    assert(harness.aBarriers[0].imageBarrier.dstQueueFamilyIndex
        == ShadowBReturnExecutionHarness::renderQueueFamily);
    assert(harness.aBarriers[1].bufferBarrier.srcAccessMask
        == VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(harness.aBarriers[1].bufferBarrier.dstAccessMask == VK_ACCESS_HOST_READ_BIT);
    assert(harness.aBarriers[2].imageBarrier.srcQueueFamilyIndex
        == ShadowBReturnExecutionHarness::renderQueueFamily);
    assert(harness.aBarriers[2].imageBarrier.dstQueueFamilyIndex
        == ShadowBReturnExecutionHarness::terminalQueueFamily);
    const auto returnedView = operation.aReturnPending().imageView();
    assert(returnedView.valid() && returnedView.image() == harness.aImportedImage);
    assert(returnedView.format() == staged.format);
    assert(returnedView.extent().width == staged.extent.width);
    assert(returnedView.extent().height == staged.extent.height);
    assert(returnedView.layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(returnedView.handoffPendingAcquire());
    assert(returnedView.sourceQueueFamily()
        == ShadowBReturnExecutionHarness::renderQueueFamily);
    assert(returnedView.destinationQueueFamily()
        == ShadowBReturnExecutionHarness::terminalQueueFamily);

    const auto identity = operation.identity();
    assert(identity.olderFrameId == static_cast<uint64_t>('B'));
    assert(identity.newerFrameId == static_cast<uint64_t>('C'));
    assert(identity.olderSlot == lsfgvk::backend::TemporalSourceSlot::Slot1);
    assert(identity.newerSlot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(identity.generationId == staged.generation);
    auto soleOwner = std::move(operation);
    assert(!operation.valid() && !operation.returnedForGraphicsOutstanding());
    assert(soleOwner.valid() && soleOwner.returnedForGraphicsOutstanding());

    bool prematureReturnedReuseRejected{};
    try { soleOwner.rejectReturnedForGraphicsReuse(); }
    catch (const std::logic_error&) { prematureReturnedReuseRejected = true; }
    assert(prematureReturnedReuseRejected);
    bool prematureOutputRetirementRejected{};
    try { soleOwner.rejectGeneratedOutputRetirement(); }
    catch (const std::logic_error&) { prematureOutputRetirementRejected = true; }
    assert(prematureOutputRetirementRejected);

    bool staleBRetirementRejected{};
    try { soleOwner.bReturnAuthority().retire(identity.generationId + 1); }
    catch (const std::logic_error&) { staleBRetirementRejected = true; }
    assert(staleBRetirementRejected);
    bool staleARetirementRejected{};
    try { soleOwner.aReturnAuthority().retire(identity.generationId + 1); }
    catch (const std::logic_error&) { staleARetirementRejected = true; }
    assert(staleARetirementRejected);

    fixture.retireSourceReads();
    auto afterSource = fixture.backend->inspectShadowPrepassGenerate(*fixture.backendSession);
    assert(afterSource.executionRetired);
    assert(afterSource.generationReadyWaitSubmitted);
    assert(!afterSource.generationReadyWaitRetired
        && !afterSource.generationReadyReusable);
    assert(soleOwner.generatedOutputLive());

    harness.retireFence();
    returnSession.retireShadowBReturnForTesting(
        soleOwner, *fixture.backend, *fixture.backendSession);
    auto afterBFence = fixture.backend->inspectShadowPrepassGenerate(*fixture.backendSession);
    assert(soleOwner.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::RETIRED);
    assert(afterBFence.generationReadyWaitRetired);
    assert(!afterBFence.generationReadyReusable);
    assert(soleOwner.generatedOutputLive());

    bool doubleBRetirementRejected{};
    try { soleOwner.bReturnAuthority().retire(identity.generationId); }
    catch (const std::logic_error&) { doubleBRetirementRejected = true; }
    assert(doubleBRetirementRejected);
    returnSession.releaseShadowGeneratedOutputForTesting(
        soleOwner, *fixture.backend, *fixture.backendSession);
    const auto afterOutput = fixture.backend->inspectShadowPrepassGenerate(
        *fixture.backendSession);
    assert(!soleOwner.generatedOutputLive() && soleOwner.generatedOutputRetired());
    assert(afterOutput.generatedOutputRetired && !afterOutput.generatedOutputLive);
    assert(afterOutput.generationReadyReusable);

    harness.retireAFence();
    returnSession.retireShadowAReturnForTesting(soleOwner);
    assert(soleOwner.aReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::RETIRED);
    assert(soleOwner.payloadAuthority().state()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_RETIRED);
    assert(soleOwner.payloadAuthority().canReimportPayload());
    assert(soleOwner.returnedForGraphicsOutstanding());
    assert(harness.aFenceWaitCalls == 1);

    bool duplicateACompletionRejected{};
    try { static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(
        soleOwner.aReturnPending())); }
    catch (const std::logic_error&) { duplicateACompletionRejected = true; }
    assert(duplicateACompletionRejected);
    bool doubleARetirementRejected{};
    try { soleOwner.aReturnAuthority().retire(identity.generationId); }
    catch (const std::logic_error&) { doubleARetirementRejected = true; }
    assert(doubleARetirementRejected);
    bool postFenceReturnedReuseRejected{};
    try { soleOwner.rejectReturnedForGraphicsReuse(); }
    catch (const std::logic_error&) { postFenceReturnedReuseRejected = true; }
    assert(postFenceReturnedReuseRejected);

    const auto finalSnapshot = fixture.backend->inspectShadowPrepassGenerate(
        *fixture.backendSession);
    assert(finalSnapshot.totalShadowSubmitCount == 2);
    assert(finalSnapshot.bReturnSubmitCount == 1);
    assert(finalSnapshot.aReturnSubmitCount == 1);
    assert(harness.frontSubmitCount() == 3);
    assert(harness.observed.calls == 1 && harness.aObserved.calls == 1);
    assert(harness.foreignReadbackPendingCalls == 1);
    assert(harness.returnedForGraphicsCalls == 1);
    assert(harness.terminalCalls == 0);
    assert(harness.deviceIdleCalls == 0);
}

std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession>
makeReturnSession(RealShadowPairFixture& fixture) {
    return std::make_unique<lsfgvk::layer::GeneratedOutputReturnSession>(
        lsfgvk::layer::ProductionReturnExecution{}, fixture.harness.devicePair(),
        fixture.harness.endpoint(), fixture.harness.renderEndpoint(), true);
}

void testProductionNonblockingGenerateRetirement() {
    RealShadowPairFixture fixture;
    fixture.harness.backendFenceStatusResult = VK_NOT_READY;
    assert(fixture.backend->tryRetireRuntimePrepassGenerate(*fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::NOT_READY);
    auto pendingSnapshot = fixture.backend->inspectRuntimePrepassGenerate(
        *fixture.backendSession);
    assert(!pendingSnapshot.executionRetired && !pendingSnapshot.ingestReadyReusable);
    assert(pendingSnapshot.generationReadySignalOutstanding);

    fixture.harness.backendFenceStatusResult = VK_SUCCESS;
    assert(fixture.backend->tryRetireRuntimePrepassGenerate(*fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(fixture.backend->tryRetireRuntimePrepassGenerate(*fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    const auto retired = fixture.backend->inspectRuntimePrepassGenerate(
        *fixture.backendSession);
    assert(retired.executionRetired && retired.ingestReadyReusable);
    assert(retired.generationReadySignalOutstanding && !retired.generationReadyReusable);
    assert(fixture.harness.frontFenceWaitCount() == fixture.hostWaitsBeforeGraph);
}

void testProductionGenerateRetirementDeviceLost() {
    RealShadowPairFixture fixture;
    fixture.harness.backendFenceStatusResult = VK_ERROR_DEVICE_LOST;
    assert(fixture.backend->tryRetireRuntimePrepassGenerate(*fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::DEVICE_LOST);
    const auto failed = fixture.backend->inspectRuntimePrepassGenerate(
        *fixture.backendSession);
    assert(failed.failed && failed.deviceLost && !failed.executionRetired);
}

void testProductionBReturnAndOutputRetirement() {
    RealShadowPairFixture fixture;
    fixture.harness.backendFenceStatusResult = VK_SUCCESS;
    assert(fixture.backend->tryRetireRuntimePrepassGenerate(*fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    auto pending = fixture.take();
    auto returnSession = makeReturnSession(fixture);
    auto bPending = fixture.harness.submitProductionBReturn(
        *returnSession, std::move(pending), *fixture.backend, *fixture.backendSession);
    auto operation = fixture.harness.completeProductionAReturn(
        *returnSession, std::move(bPending), *fixture.backend, *fixture.backendSession);

    assert(returnSession->releaseProductionGeneratedOutput(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::NOT_READY);
    assert(returnSession->tryRetireProductionBReturn(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::NOT_READY);
    assert(operation.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
    assert(operation.generatedOutputLive());

    fixture.harness.retireFence();
    assert(returnSession->tryRetireProductionBReturn(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(returnSession->tryRetireProductionBReturn(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(operation.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::RETIRED);
    assert(operation.generatedOutputLive());
    assert(!operation.aReturnRetired());

    assert(returnSession->releaseProductionGeneratedOutput(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(returnSession->releaseProductionGeneratedOutput(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(operation.generatedOutputRetired());
    const auto released = fixture.backend->inspectRuntimePrepassGenerate(
        *fixture.backendSession);
    assert(released.generationReadyWaitRetired && released.generationReadyReusable);
    assert(!operation.aReturnRetired());
    assert(fixture.harness.aFenceWaitCalls == 0);
}

void testProductionCoreOwnerLifecycle() {
    lsfgvk::test::ShadowBReturnExecutionHarness harness;
    auto backend = harness.createB0B3Instance();
    const auto pair = harness.devicePair();
    const auto submitsBeforeConstruction = harness.frontSubmitCount();
    lsfgvk::layer::D3B3ProductionCoreOwner owner({
        .backend = backend.get(),
        .devicePair = &pair,
        .exchangeChannel = harness.exchangeChannelOwnerToken(),
        .generationEndpoint = harness.endpoint(),
        .renderEndpoint = harness.renderEndpoint(),
        .extent = {64, 64},
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .modifier = 0,
        .flow = 1.0F,
        .performanceMode = false,
        .sourceImage = [](lsfgvk::backend::TemporalSourceSlot) {
            return reinterpret_cast<VkImage>(uintptr_t{0xC100});
        }});
    assert(owner.structurallyReady());
    assert(harness.frontSubmitCount() == submitsBeforeConstruction);

    auto frameTransportA = vk::RuntimeImageEndpointTestAccess::make(
        harness.frameTransportAEndpoint(),
        reinterpret_cast<VkImage>(uintptr_t{0xA6A501}), {64, 64},
        VK_FORMAT_R8G8B8A8_UNORM);
    const auto makeTransport = [&](uint64_t generation) {
        const auto lifetime = std::make_shared<const uint8_t>(0);
        const vk::RuntimeFrameTransportSource source{
            .image = reinterpret_cast<VkImage>(uintptr_t{0xC100}),
            .currentLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {64, 64},
            .generation = generation,
            .lifetime = lifetime};
        auto submitted = vk::RuntimeImageEndpoint::trySubmitFrameTransportA(
            frameTransportA, source);
        assert(submitted.status
            == vk::RuntimeFrameTransportSubmitStatus::SUBMITTED);
        assert(submitted.submission.valid());
        return std::move(submitted.submission);
    };

    const auto warmup = [&](uint64_t frame, auto slot) {
        auto submitted = owner.submitWarmup(frame, slot, makeTransport(frame));
        assert(submitted.submitAccepted && submitted.signalSemaphore == VK_NULL_HANDLE
            && submitted.ingestReadyReusable);
        assert(owner.tryRetireWarmup()
            == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED);
        const auto observed = owner.temporalObservationSource(slot);
        assert(observed.image != VK_NULL_HANDLE
            && observed.layout == VK_IMAGE_LAYOUT_GENERAL
            && observed.format == VK_FORMAT_R8G8B8A8_UNORM
            && observed.extent.width == 64 && observed.extent.height == 64
            && observed.queueFamilyIndex == 3
            && !observed.lifetime.expired());
    };
    warmup('A', lsfgvk::backend::TemporalSourceSlot::Slot0);
    warmup('B', lsfgvk::backend::TemporalSourceSlot::Slot1);

    const auto c = owner.submitGenerateSource('C',
        lsfgvk::backend::TemporalSourceSlot::Slot0, makeTransport('C'));
    assert(c.submitAccepted && c.signalSemaphore != VK_NULL_HANDLE);
    assert(!owner.temporalObservationSource(
        lsfgvk::backend::TemporalSourceSlot::Slot0).image);
    const auto generated = owner.submitGenerate({
        .olderSlot = lsfgvk::backend::TemporalSourceSlot::Slot1,
        .newerSlot = lsfgvk::backend::TemporalSourceSlot::Slot0,
        .olderFrameId = 'B', .newerFrameId = 'C', .generationId = c.epoch});
    assert(generated.submitAccepted && generated.generationReadySignalOutstanding);
    harness.backendFenceStatusResult = VK_SUCCESS;
    assert(owner.tryRetireGenerate()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(frameTransportA.frameTransportStateValue()
        == vk::RuntimeFrameTransportState::REUSABLE);
    const auto observedC = owner.temporalObservationSource(
        lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(observedC.image != VK_NULL_HANDLE
        && observedC.layout == VK_IMAGE_LAYOUT_GENERAL
        && observedC.format == VK_FORMAT_R8G8B8A8_UNORM
        && observedC.extent.width == 64 && observedC.extent.height == 64
        && observedC.queueFamilyIndex == 3
        && !observedC.lifetime.expired());
    const auto observedGenerated = owner.generatedObservationSource();
    assert(observedGenerated.image != VK_NULL_HANDLE
        && observedGenerated.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && observedGenerated.format == VK_FORMAT_R8G8B8A8_UNORM
        && observedGenerated.extent.width == 64
        && observedGenerated.extent.height == 64
        && observedGenerated.queueFamilyIndex == 3
        && !observedGenerated.lifetime.expired());

    harness.armProductionReturnForOwner();
    owner.submitBReturn();
    const auto pendingB = owner.tryRetireBReturn();
    assert(pendingB == lsfgvk::backend::RuntimeRetirementStatus::NOT_READY);
    harness.retireFence();
    assert(owner.tryRetireBReturn()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    owner.submitAReturn(harness.productionHandoff());
    assert(owner.releaseGeneratedOutput()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(!owner.tryRetireAReturn());
    harness.retireAFence();
    assert(owner.tryRetireAReturn());
    const auto final = owner.generateState();
    assert(final.generationReadyWaitRetired && final.generationReadyReusable);
}

void testProductionCoreOwnerMissingDependencies() {
    bool rejected{};
    try {
        static_cast<void>(lsfgvk::layer::D3B3ProductionCoreOwner({}));
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void retireRecoveredBReturn(RealShadowPairFixture& fixture,
        lsfgvk::layer::GeneratedOutputReturnSession& returnSession,
        lsfgvk::layer::RuntimeGeneratedBReturnPending& recovered) {
    fixture.retireSourceReads();
    fixture.harness.retireFence();
    returnSession.retireAcceptedShadowBReturnFailureForTesting(
        recovered, *fixture.backend, *fixture.backendSession);
    assert(recovered.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::RETIRED);
    returnSession.releaseAcceptedShadowGeneratedOutputForTesting(
        recovered, *fixture.backend, *fixture.backendSession);
    const auto snapshot = fixture.backend->inspectShadowPrepassGenerate(
        *fixture.backendSession);
    assert(snapshot.generationReadyReusable);
    assert(snapshot.generatedOutputRetired);
}

void testBReturnSubmitFailures() {
    for (const auto failure : {VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_DEVICE_LOST}) {
        RealShadowPairFixture fixture;
        auto returnSession = makeReturnSession(fixture);
        auto pending = fixture.take();
        fixture.harness.submitResult = failure;
        bool rejected{};
        try {
            static_cast<void>(fixture.harness.submitProductionBReturn(
                *returnSession, std::move(pending),
                *fixture.backend, *fixture.backendSession));
        } catch (const ls::vulkan_error& error) {
            rejected = error.error() == failure;
        }
        assert(rejected);
        assert(pending.valid() && pending.transportConsumedForTesting());
        bool retryRejected{};
        try { pending.consumeTransport(); }
        catch (const std::logic_error&) { retryRejected = true; }
        assert(retryRejected);
        assert(fixture.harness.productionBReturnCalls == 1);
        assert(fixture.harness.observed.calls == 1);
        assert(fixture.harness.productionBReturnExportCalls == 0);
        assert(fixture.harness.aObserved.calls == 0);
        assert(returnSession->gpuChainedStateForTesting()
            == lsfgvk::layer::GpuChainedReturnState::FAILED);
        const auto snapshot = fixture.backend->inspectShadowPrepassGenerate(
            *fixture.backendSession);
        assert(snapshot.failed);
        assert(snapshot.deviceLost == (failure == VK_ERROR_DEVICE_LOST));
        assert(snapshot.bReturnSubmitCount == 0 && snapshot.aReturnSubmitCount == 0);
        assert(!snapshot.generationReadyWaitSubmitted);
        assert(!snapshot.generationReadyWaitRetired);
        assert(!snapshot.generationReadyReusable);
        bool noAcceptedAuthority{};
        try { static_cast<void>(returnSession->takeAcceptedShadowFailureForTesting()); }
        catch (const std::logic_error&) { noAcceptedAuthority = true; }
        assert(noAcceptedAuthority);
        assert(fixture.harness.deviceIdleCalls == 0);
        assert(fixture.harness.terminalCalls == 0);
    }
}

void testBReturnExportFailureAfterAcceptedSubmit() {
    RealShadowPairFixture fixture;
    auto returnSession = makeReturnSession(fixture);
    auto pending = fixture.take();
    fixture.harness.exportResult = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    bool failed{};
    try {
        static_cast<void>(fixture.harness.submitProductionBReturn(
            *returnSession, std::move(pending),
            *fixture.backend, *fixture.backendSession));
    } catch (const vk::ExternalSemaphoreError& error) {
        failed = error.result() == VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    assert(failed && !pending.valid());
    assert(fixture.harness.observed.calls == 1);
    assert(fixture.harness.productionBReturnExportCalls == 1);
    assert(fixture.harness.aObserved.calls == 0);
    auto recoveredOptional = returnSession->takeAcceptedShadowFailureForTesting();
    assert(recoveredOptional.has_value());
    auto recovered = std::move(*recoveredOptional);
    assert(!recovered.valid());
    assert(recovered.acceptedSubmissionAuthorityForTesting());
    assert(!recovered.payloadOwnedForTesting());
    assert(recovered.bReturnAuthority().fenceHandle() == fixture.harness.returnFence);
    assert(recovered.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
    const auto submitted = fixture.backend->inspectShadowPrepassGenerate(
        *fixture.backendSession);
    assert(submitted.bReturnSubmitCount == 1 && submitted.aReturnSubmitCount == 0);
    assert(submitted.generationReadyWaitSubmitted);
    assert(!submitted.generationReadyWaitRetired && !submitted.generationReadyReusable);
    retireRecoveredBReturn(fixture, *returnSession, recovered);
    assert(fixture.harness.deviceIdleCalls == 0);
    assert(fixture.harness.terminalCalls == 0);
}

void testBReturnFenceRetirementFailure() {
    RealShadowPairFixture fixture;
    auto returnSession = makeReturnSession(fixture);
    auto pending = fixture.take();
    auto bPending = fixture.harness.submitProductionBReturn(
        *returnSession, std::move(pending), *fixture.backend, *fixture.backendSession);
    fixture.retireSourceReads();
    fixture.harness.fenceWaitResult = VK_ERROR_DEVICE_LOST;
    bool failed{};
    try {
        returnSession->retireAcceptedShadowBReturnFailureForTesting(
            bPending, *fixture.backend, *fixture.backendSession);
    } catch (const ls::vulkan_error& error) {
        failed = error.error() == VK_ERROR_DEVICE_LOST;
    }
    assert(failed);
    assert(bPending.bReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::FAILED);
    const auto snapshot = fixture.backend->inspectShadowPrepassGenerate(
        *fixture.backendSession);
    assert(snapshot.failed && snapshot.deviceLost);
    assert(!snapshot.generationReadyReusable && !snapshot.generatedOutputRetired);
    assert(fixture.harness.aObserved.calls == 0);
    assert(fixture.harness.terminalCalls == 0);
}

struct AFailureCase {
    ShadowAHarnessFailurePoint point{ShadowAHarnessFailurePoint::NONE};
    VkResult submitResult{VK_SUCCESS};
    uint32_t expectedQueueSubmits{};
    uint32_t expectedSemaphoreImports{};
};

void testAReturnFailureMatrix() {
    const std::vector<AFailureCase> cases{
        {ShadowAHarnessFailurePoint::MEMORY_IMPORT, VK_SUCCESS, 0, 0},
        {ShadowAHarnessFailurePoint::SEMAPHORE_CREATE, VK_SUCCESS, 0, 0},
        {ShadowAHarnessFailurePoint::SEMAPHORE_IMPORT, VK_SUCCESS, 0, 1},
        {ShadowAHarnessFailurePoint::BEGIN_COMMAND_BUFFER, VK_SUCCESS, 0, 1},
        {ShadowAHarnessFailurePoint::FENCE_CREATE, VK_SUCCESS, 0, 1},
        {ShadowAHarnessFailurePoint::NONE, VK_ERROR_INITIALIZATION_FAILED, 1, 1},
        {ShadowAHarnessFailurePoint::NONE, VK_ERROR_DEVICE_LOST, 1, 1}};
    for (const auto& failure : cases) {
        RealShadowPairFixture fixture;
        auto returnSession = makeReturnSession(fixture);
        auto pending = fixture.take();
        auto bPending = fixture.harness.submitProductionBReturn(
            *returnSession, std::move(pending),
            *fixture.backend, *fixture.backendSession);
        fixture.harness.aFailurePoint = failure.point;
        fixture.harness.aSubmitResult = failure.submitResult;
        bool failed{};
        try {
            static_cast<void>(fixture.harness.completeProductionAReturn(
                *returnSession, std::move(bPending),
                *fixture.backend, *fixture.backendSession));
        } catch (const std::exception&) {
            failed = true;
        }
        assert(failed && !bPending.valid());
        assert(fixture.harness.aObserved.calls == failure.expectedQueueSubmits);
        assert(fixture.harness.aSemaphoreImport.calls
            == failure.expectedSemaphoreImports);
        assert(fixture.harness.foreignReadbackPendingCalls == 0);
        assert(fixture.harness.returnedForGraphicsCalls == 0);
        auto recoveredOptional = returnSession->takeAcceptedShadowFailureForTesting();
        assert(recoveredOptional.has_value());
        auto recovered = std::move(*recoveredOptional);
        assert(recovered.acceptedSubmissionAuthorityForTesting());
        assert(recovered.bReturnAuthority().state()
            == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
        const auto snapshot = fixture.backend->inspectShadowPrepassGenerate(
            *fixture.backendSession);
        assert(snapshot.bReturnSubmitCount == 1);
        assert(snapshot.aReturnSubmitCount == 0);
        assert(snapshot.generationReadyWaitSubmitted);
        retireRecoveredBReturn(fixture, *returnSession, recovered);
        for (size_t i = 0; i < fixture.harness.aFreedMemories.size(); ++i)
            for (size_t j = i + 1; j < fixture.harness.aFreedMemories.size(); ++j)
                assert(fixture.harness.aFreedMemories[i]
                    != fixture.harness.aFreedMemories[j]);
        assert(fixture.harness.terminalCalls == 0);
    }
}

void testAReturnFenceRetirementFailure() {
    RealShadowPairFixture fixture;
    auto returnSession = makeReturnSession(fixture);
    auto pending = fixture.take();
    auto bPending = fixture.harness.submitProductionBReturn(
        *returnSession, std::move(pending), *fixture.backend, *fixture.backendSession);
    auto operation = fixture.harness.completeProductionAReturn(
        *returnSession, std::move(bPending), *fixture.backend, *fixture.backendSession);
    fixture.retireSourceReads();
    fixture.harness.retireFence();
    returnSession->retireShadowBReturnForTesting(
        operation, *fixture.backend, *fixture.backendSession);
    returnSession->releaseShadowGeneratedOutputForTesting(
        operation, *fixture.backend, *fixture.backendSession);
    fixture.harness.aFenceWaitResult = VK_ERROR_DEVICE_LOST;
    bool failed{};
    try { returnSession->retireShadowAReturnForTesting(operation); }
    catch (const ls::vulkan_error& error) {
        failed = error.error() == VK_ERROR_DEVICE_LOST;
    }
    assert(failed);
    assert(operation.aReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::FAILED);
    assert(operation.payloadAuthority().state()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::FAILED);
    assert(operation.returnedForGraphicsOutstanding());
    assert(!operation.aReturnPending().valid());
    operation = {};
    assert(fixture.harness.deviceIdleCalls == 1);
    assert(fixture.harness.terminalCalls == 0);
}

template<class T> T pairHandle(uintptr_t value) {
    return reinterpret_cast<T>(value);
}

lsfgvk::backend::ReturnedGeneratedOperation makeSubmittedB0COperation(
        RealShadowPairFixture& fixture,
        lsfgvk::layer::GeneratedOutputReturnSession& returnSession) {
    const auto aWaitCallsBefore = fixture.harness.aFenceWaitCalls;
    const auto aStatusCallsBefore = fixture.harness.aFenceStatusCalls;
    auto pending = fixture.take();
    auto bPending = fixture.harness.submitProductionBReturn(
        returnSession, std::move(pending), *fixture.backend, *fixture.backendSession);
    auto operation = fixture.harness.completeProductionAReturn(
        returnSession, std::move(bPending), *fixture.backend, *fixture.backendSession);

    fixture.retireSourceReads();
    fixture.harness.retireFence();
    assert(returnSession.tryRetireProductionBReturn(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(returnSession.releaseProductionGeneratedOutput(
        operation, *fixture.backend, *fixture.backendSession)
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    assert(operation.valid() && operation.terminalReady());
    assert(operation.returnedForGraphicsOutstanding());
    assert(operation.aReturnAuthority().state()
        == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
    assert(operation.payloadAuthority().state()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_SUBMITTED);
    assert(!operation.aReturnPending().completed());
    assert(!operation.aReturnPending().retirementObserved());
    assert(fixture.harness.aFenceWaitCalls == aWaitCallsBefore);
    assert(fixture.harness.aFenceStatusCalls == aStatusCallsBefore);
    return operation;
}

enum class PairTerminalFailure {
    NONE, PREFLIGHT, ACQUIRE_GENERATED, ACQUIRE_ORIGINAL, RECORD, SUBMIT,
    DEVICE_LOST, GENERATED_PRESENT, ORIGINAL_PRESENT, GRAPHICS_FENCE,
    GENERATED_PRESENT_FENCE
};

struct PairTerminalMock {
    PairTerminalFailure failure{PairTerminalFailure::NONE};
    std::vector<std::string> events;
    std::string pairLabel;
    uint32_t acquireCalls{};
    uint32_t releaseCalls{};
    uint32_t releasedImages{};
    uint32_t terminalSubmitCalls{};
    VkSemaphore terminalWaitSemaphore{};
    VkImage terminalGeneratedSource{};
    std::function<void()> beforeGraphicsFenceRetirement;
    bool nonblocking{};
    VkResult graphicsFenceStatus{VK_SUCCESS};
    uint32_t graphicsFencePolls{};
    VkResult generatedPresentFenceStatus{VK_SUCCESS};
    VkResult originalPresentFenceStatus{VK_SUCCESS};
    uint32_t generatedPresentFencePolls{};
    uint32_t originalPresentFencePolls{};
    uint32_t bridgeSubmits{};
    std::vector<VkSemaphore> bridgedApplicationWaits;

    lsfgvk::layer::D3B2InsertionPath build(
            lsfgvk::layer::D3B2Source generated,
            lsfgvk::layer::D3B2Source original,
            VkSemaphore returnedForGraphics) {
        lsfgvk::layer::D3B2InsertionPath path;
        path.generated = generated;
        path.original = original;
        path.returnedForGraphics = returnedForGraphics;
        terminalWaitSemaphore = returnedForGraphics;
        terminalGeneratedSource = generated.image;
        path.acquireGenerated = pairHandle<VkSemaphore>(0xD301);
        path.acquireOriginal = pairHandle<VkSemaphore>(0xD302);
        path.generatedPresentReady = pairHandle<VkSemaphore>(0xD303);
        path.originalPresentReady = pairHandle<VkSemaphore>(0xD304);
        path.graphicsFence = pairHandle<VkFence>(0xD305);
        path.commandPoolFamily = 11;
        path.submitQueueFamily = 11;
        path.submitQueueFlags = VK_QUEUE_GRAPHICS_BIT;
        path.singleSwapchain = true;
        path.fifo = true;
        path.hiddenBlitDestinationSupported = true;
        path.maintenanceReleaseCapable = failure != PairTerminalFailure::PREFLIGHT;
        path.presentFences = {
            pairHandle<VkFence>(0xD306), pairHandle<VkFence>(0xD307)};
        path.returnedForGraphicsWaitStage =
            lsfgvk::layer::d3b2FirstTerminalConsumerStage();
        const auto acquireGenerated = path.acquireGenerated;
        const auto acquireOriginal = path.acquireOriginal;
        const auto generatedPresentReady = path.generatedPresentReady;
        const auto originalPresentReady = path.originalPresentReady;
        const auto generatedPresentFence = path.presentFences.generated;
        const auto originalPresentFence = path.presentFences.original;
        const auto label = pairLabel;
        const auto generatedSourceImage = generated.image;
        const auto originalSourceImage = original.image;
        const bool generatedOwnershipAcquire = generated.ownershipAcquireRequired;
        path.acquire = [&, acquireGenerated, acquireOriginal](VkSemaphore semaphore) {
            ++acquireCalls;
            const bool generatedAcquire = semaphore == acquireGenerated;
            events.emplace_back(generatedAcquire ? "ACQUIRE_G" : "ACQUIRE_O");
            if (generatedAcquire && failure == PairTerminalFailure::ACQUIRE_GENERATED)
                throw std::runtime_error("G acquire");
            if (!generatedAcquire && failure == PairTerminalFailure::ACQUIRE_ORIGINAL)
                throw std::runtime_error("O acquire");
            return lsfgvk::layer::D3B2HiddenImage{
                pairHandle<VkImage>(0xD310 + acquireCalls), acquireCalls, {8, 8}};
        };
        path.record = [&, generatedSourceImage, originalSourceImage,
                generatedOwnershipAcquire](const auto& hiddenGenerated,
                const auto& hiddenOriginal) {
            events.emplace_back("TERMINAL_RECORD_BEGIN");
            assert(hiddenGenerated.image != VK_NULL_HANDLE);
            assert(hiddenOriginal.image != VK_NULL_HANDLE);
            assert(generatedSourceImage != VK_NULL_HANDLE);
            assert(originalSourceImage != VK_NULL_HANDLE);
            assert(generatedOwnershipAcquire);
            events.emplace_back("BLIT_GENERATED_G_TO_HIDDEN");
            events.emplace_back("BLIT_ORIGINAL_C_TO_HIDDEN");
            if (failure == PairTerminalFailure::RECORD)
                throw std::runtime_error("record");
            events.emplace_back("TERMINAL_RECORD_END");
        };
        path.submitWithWaitStage = [&](VkPipelineStageFlags stage) {
            assert(stage == VK_PIPELINE_STAGE_TRANSFER_BIT);
            ++terminalSubmitCalls;
            events.emplace_back("TERMINAL_SUBMIT_WAIT_RETURNED_FOR_GRAPHICS");
            if (failure == PairTerminalFailure::DEVICE_LOST)
                return VK_ERROR_DEVICE_LOST;
            if (failure == PairTerminalFailure::SUBMIT)
                return VK_ERROR_INITIALIZATION_FAILED;
            return VK_SUCCESS;
        };
        path.present = [&](const auto&, VkSemaphore) { return VK_SUCCESS; };
        path.presentWithFence = [&, label, generatedPresentReady, originalPresentReady,
                generatedPresentFence, originalPresentFence](const auto&, VkSemaphore ready,
                VkFence fence) {
            const bool generatedPresent = ready == generatedPresentReady;
            assert((generatedPresent && fence == generatedPresentFence)
                || (!generatedPresent && ready == originalPresentReady
                    && fence == originalPresentFence));
            events.emplace_back(generatedPresent
                ? (label.empty() ? "PRESENT_GENERATED" : "PRESENT_GENERATED_" + label)
                : (label.empty() ? "PRESENT_ORIGINAL" : "PRESENT_ORIGINAL_" + label));
            if (generatedPresent && failure == PairTerminalFailure::GENERATED_PRESENT)
                return VK_ERROR_OUT_OF_DATE_KHR;
            if (!generatedPresent && failure == PairTerminalFailure::ORIGINAL_PRESENT)
                return VK_ERROR_OUT_OF_DATE_KHR;
            return VK_SUCCESS;
        };
        path.waitGraphicsFence = [&] {
            if (beforeGraphicsFenceRetirement)
                beforeGraphicsFenceRetirement();
            events.emplace_back("TERMINAL_GRAPHICS_FENCE_WAIT");
            return failure != PairTerminalFailure::GRAPHICS_FENCE;
        };
        if (nonblocking) {
            path.tryRetireGraphicsFence = [&] {
                ++graphicsFencePolls;
                events.emplace_back("TERMINAL_GRAPHICS_FENCE_POLL");
                return graphicsFenceStatus;
            };
        }
        path.generatedIntegrity = [] { return true; };
        path.originalIdentity = [&]( ) { return true; };
        path.retire = [] {};
        path.emitMarker = [] {};
        path.releaseAcquiredImages = [&](const std::vector<uint32_t>& indices) {
            ++releaseCalls;
            releasedImages += static_cast<uint32_t>(indices.size());
            events.emplace_back("RELEASE_HIDDEN_ACQUISIONS");
            return VK_SUCCESS;
        };
        return path;
    }
};

constexpr uintptr_t pairAuthorityScope = 0xD3B30001;

lsfgvk::layer::D3B3OriginalReadyAuthority makeOriginalReadyAuthority(
        PairTerminalMock& mock, uint64_t epoch, VkSemaphore originalReady,
        uintptr_t scope = pairAuthorityScope) {
    const std::vector<VkSemaphore> waits{
        pairHandle<VkSemaphore>(0xD400 + epoch * 2),
        pairHandle<VkSemaphore>(0xD401 + epoch * 2)};
    lsfgvk::layer::ApplicationPresentWaitAuthority input(waits, epoch, scope);
    auto result = lsfgvk::layer::bridgeApplicationPresentWaits(input,
        pairHandle<VkQueue>(0xD399), {originalReady}, originalReady,
        [&](VkQueue, const VkSubmitInfo& submit) {
            ++mock.bridgeSubmits;
            assert(submit.waitSemaphoreCount == waits.size());
            assert(submit.signalSemaphoreCount == 1);
            mock.bridgedApplicationWaits.insert(mock.bridgedApplicationWaits.end(),
                submit.pWaitSemaphores,
                submit.pWaitSemaphores + submit.waitSemaphoreCount);
            return VK_SUCCESS;
        });
    assert(result.result == VK_SUCCESS && result.originalReady);
    assert(input.state() == lsfgvk::layer::ApplicationPresentWaitState::BRIDGED);
    return std::move(*result.originalReady);
}

lsfgvk::layer::D3B3PairOperation makePairOperation(
        RealShadowPairFixture& fixture, PairTerminalMock& terminalMock,
        std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession>& session,
        const std::function<void(lsfgvk::backend::ReturnedGeneratedOperation&)>& mutate = {},
        uintptr_t originalScope = pairAuthorityScope,
        uintptr_t readyScope = pairAuthorityScope,
        int64_t readyEpochOffset = 0) {
    session = makeReturnSession(fixture);
    auto returned = makeSubmittedB0COperation(fixture, *session);
    if (mutate)
        mutate(returned);
    const auto pairIdentity = returned.identity();
    auto originalLifetime = std::make_shared<const uint8_t>(0);
    lsfgvk::layer::D3B3OriginalSourceAuthority original(
        {.image = pairHandle<VkImage>(0xC001),
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .extent = {8, 8}, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
         .queueFamily = 11, .virtualImageIndex = 1,
         .frameId = pairIdentity.newerFrameId,
         .authorityScope = originalScope},
        std::move(originalLifetime));
    const auto originalReady = pairHandle<VkSemaphore>(0xD300);
    lsfgvk::layer::D3B3PairTerminalDispatch dispatch{
        .originalReadyAuthority = makeOriginalReadyAuthority(
            terminalMock, static_cast<uint64_t>(
                static_cast<int64_t>(pairIdentity.generationId) + readyEpochOffset),
            originalReady, readyScope),
        .build = [&](auto generated, auto originalSource, VkSemaphore ready) {
            return terminalMock.build(generated, originalSource, ready);
        },
        .retireGeneratedPresentFence = [&] {
            terminalMock.events.emplace_back("PRESENT_FENCE_GENERATED_RETIRED");
            if (terminalMock.failure == PairTerminalFailure::GENERATED_PRESENT_FENCE)
                throw std::runtime_error("generated present fence");
        },
        .retireOriginalPresentFence = [&] {
            terminalMock.events.emplace_back("PRESENT_FENCE_ORIGINAL_RETIRED");
        },
        .tryRetireGeneratedPresentFence = [&] {
            ++terminalMock.generatedPresentFencePolls;
            if (terminalMock.generatedPresentFenceStatus == VK_NOT_READY)
                return lsfgvk::layer::D3B2RetirementResult::NOT_READY;
            return terminalMock.generatedPresentFenceStatus == VK_SUCCESS
                ? lsfgvk::layer::D3B2RetirementResult::RETIRED
                : terminalMock.generatedPresentFenceStatus == VK_ERROR_DEVICE_LOST
                    ? lsfgvk::layer::D3B2RetirementResult::DEVICE_LOST
                    : lsfgvk::layer::D3B2RetirementResult::FAILED;
        },
        .tryRetireOriginalPresentFence = [&] {
            ++terminalMock.originalPresentFencePolls;
            if (terminalMock.originalPresentFenceStatus == VK_NOT_READY)
                return lsfgvk::layer::D3B2RetirementResult::NOT_READY;
            return terminalMock.originalPresentFenceStatus == VK_SUCCESS
                ? lsfgvk::layer::D3B2RetirementResult::RETIRED
                : terminalMock.originalPresentFenceStatus == VK_ERROR_DEVICE_LOST
                    ? lsfgvk::layer::D3B2RetirementResult::DEVICE_LOST
                    : lsfgvk::layer::D3B2RetirementResult::FAILED;
        }};
    auto pair = lsfgvk::layer::D3B3PairOperation(
        std::move(returned), std::move(original), std::move(dispatch));
    assert(!returned.valid() && !returned.aReturnPending().valid());
    assert(pair.returnedOperation().terminalReady());
    assert(pair.returnedOperation().aReturnPending().retirementFence()
        == fixture.harness.aReturnFence);
    assert(pair.returnedOperation().payloadAuthority().semaphoreHandle()
        == fixture.harness.aImportedWait);
    return pair;
}

lsfgvk::layer::D3B3PairTerminalDispatch makeFiniteTerminalDispatch(
        PairTerminalMock& terminalMock,
        const lsfgvk::backend::RuntimeTemporalPairIdentity& identity) {
    const std::string label{
        static_cast<char>(identity.olderFrameId), static_cast<char>(identity.newerFrameId)};
    terminalMock.pairLabel = label;
    const auto originalReady = pairHandle<VkSemaphore>(0xD300);
    return {
        .originalReadyAuthority = makeOriginalReadyAuthority(
            terminalMock, identity.generationId, originalReady),
        .build = [&terminalMock, label](auto generated, auto original, VkSemaphore returned) {
            terminalMock.pairLabel = label;
            return terminalMock.build(generated, original, returned);
        },
        .retireGeneratedPresentFence = [&terminalMock, label] {
            terminalMock.events.emplace_back("PRESENT_FENCE_GENERATED_" + label);
        },
        .retireOriginalPresentFence = [&terminalMock, label] {
            terminalMock.events.emplace_back("PRESENT_FENCE_ORIGINAL_" + label);
        },
        .tryRetireGeneratedPresentFence = [&terminalMock, label] {
            ++terminalMock.generatedPresentFencePolls;
            terminalMock.events.emplace_back("PRESENT_FENCE_GENERATED_POLL_" + label);
            if (terminalMock.generatedPresentFenceStatus == VK_NOT_READY)
                return lsfgvk::layer::D3B2RetirementResult::NOT_READY;
            return terminalMock.generatedPresentFenceStatus == VK_ERROR_DEVICE_LOST
                ? lsfgvk::layer::D3B2RetirementResult::DEVICE_LOST
                : terminalMock.generatedPresentFenceStatus == VK_SUCCESS
                    ? lsfgvk::layer::D3B2RetirementResult::RETIRED
                    : lsfgvk::layer::D3B2RetirementResult::FAILED;
        },
        .tryRetireOriginalPresentFence = [&terminalMock, label] {
            ++terminalMock.originalPresentFencePolls;
            terminalMock.events.emplace_back("PRESENT_FENCE_ORIGINAL_POLL_" + label);
            if (terminalMock.originalPresentFenceStatus == VK_NOT_READY)
                return lsfgvk::layer::D3B2RetirementResult::NOT_READY;
            return terminalMock.originalPresentFenceStatus == VK_ERROR_DEVICE_LOST
                ? lsfgvk::layer::D3B2RetirementResult::DEVICE_LOST
                : terminalMock.originalPresentFenceStatus == VK_SUCCESS
                    ? lsfgvk::layer::D3B2RetirementResult::RETIRED
                    : lsfgvk::layer::D3B2RetirementResult::FAILED;
        }};
}

struct FiniteRealD3B3Run {
    RealShadowPairFixture fixture{false};
    PairTerminalMock terminal;
    std::vector<std::string> events;

    lsfgvk::layer::D3B3FiniteProductionOperations operations() {
        return {
            .ingest = [&](uint64_t frame, lsfgvk::backend::TemporalSourceSlot slot, bool seed) {
                if (frame < static_cast<uint64_t>('C'))
                    fixture.ingestWarmup(frame, slot);
                else
                    fixture.ingestCurrent(frame, slot);
                events.emplace_back(seed ? "REAL_INGEST_SEED" : "REAL_INGEST");
                return true;
            },
            .retireWarmupIngest = [&] {
                return fixture.backend->tryRetireRuntimeIngest(*fixture.backendSession)
                        == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED
                    ? lsfgvk::layer::D3B3RetirementStatus::RETIRED
                    : lsfgvk::layer::D3B3RetirementStatus::TIMEOUT;
            },
            .generate = [&](uint64_t older, uint64_t newer,
                    lsfgvk::backend::TemporalSourceSlot olderSlot,
                    lsfgvk::backend::TemporalSourceSlot newerSlot,
                    uint64_t generation) {
                fixture.generateCurrent(older, olderSlot, newer, newerSlot);
                assert(fixture.generated.generationId == generation);
                events.emplace_back("REAL_GENERATE");
                return true;
            },
            .presentWarmupOriginal = [&](uint64_t frame) {
                events.emplace_back("WARMUP_PRESENT_" + std::string(1, static_cast<char>(frame)));
                return true;
            },
            .returnGenerated = [&](const auto& identity) {
                auto returnSession = makeReturnSession(fixture);
                auto operation = makeSubmittedB0COperation(fixture, *returnSession);
                const auto actual = operation.identity();
                assert(actual.olderSlot == identity.olderSlot
                    && actual.newerSlot == identity.newerSlot
                    && actual.olderFrameId == identity.olderFrameId
                    && actual.newerFrameId == identity.newerFrameId
                    && actual.generationId == identity.generationId);
                events.emplace_back("REAL_B_RETURN");
                events.emplace_back("REAL_A_RETURN");
                return operation;
            },
            .makeOriginal = [](const auto& identity) {
                const auto lifetime = std::make_shared<const uint8_t>(0);
                return lsfgvk::layer::D3B3OriginalSourceAuthority(
                    {.image = pairHandle<VkImage>(0xC000 + identity.newerFrameId),
                     .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {8, 8},
                     .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     .queueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
                     .virtualImageIndex = static_cast<uint32_t>(identity.newerFrameId),
                     .frameId = identity.newerFrameId,
                     .authorityScope = pairAuthorityScope},
                    lifetime);
            },
            .makeTerminal = [&](const auto& identity) {
                terminal.beforeGraphicsFenceRetirement = [&] {
                    fixture.harness.retireAFence();
                };
                return makeFiniteTerminalDispatch(terminal, identity);
            },
            .stopWorkerAndJoin = [&] {
                events.emplace_back("STOP_JOIN");
                return true;
            },
            .record = [&](const std::string& event) { events.push_back(event); },
            .emitMarker = [&] { events.emplace_back("FINITE_MARKER"); }};
    }
};

void testD3B3FiniteABCDEFProductionState() {
    FiniteRealD3B3Run run;
    lsfgvk::layer::D3B3ProductionState state;
    state.configureFinite(run.operations());

    assert(state.presentFinite(static_cast<uint64_t>('A')));
    assert(state.presentFinite(static_cast<uint64_t>('B')));
    assert(state.presentFinite(static_cast<uint64_t>('C')));
    assert(state.presentFinite(static_cast<uint64_t>('D')));
    assert(state.presentFinite(static_cast<uint64_t>('E')));
    assert(state.finiteStopped());
    assert(state.finiteState() == lsfgvk::layer::D3B3FiniteProductionState::FINITE_STOPPED);

    const auto counters = state.finiteCounters();
    assert(counters.applicationFrames == 5);
    assert(counters.warmupOriginals == 2);
    assert(counters.generate == 3);
    assert(counters.bReturns == 3 && counters.aReturns == 3);
    assert(counters.pairOperations == 3 && counters.terminalSubmits == 3);
    assert(counters.generatedPresents == 3 && counters.originalPresents == 3);
    assert(counters.internalPresents == 6 && counters.gateTimeouts == 0);
    assert(state.pairCount() == 3 && state.eligibleFrameCount() == 5);
    assert(state.temporalSlots()[0].frameId == static_cast<uint64_t>('E'));
    assert(state.temporalSlots()[1].state
        == lsfgvk::layer::D3B3TemporalState::OVERWRITEABLE);
    assert(state.temporalSlots()[0].state
        == lsfgvk::layer::D3B3TemporalState::RETAINED_HISTORY);
    assert(state.pending().state == lsfgvk::layer::D3B3PendingState::EMPTY);
    assert(!state.acceptsPendingEpoch(1));

    const std::vector<std::string> expectedHighLevel{
        "APP_A", "INGEST", "APP_B", "INGEST", "APP_C", "INGEST", "GENERATE",
        "B_RETURN", "A_RETURN", "PAIR_READY", "TERMINAL_PREFLIGHT", "TERMINAL_SUBMIT",
        "APP_D", "INGEST", "GENERATE", "B_RETURN", "A_RETURN", "PAIR_READY",
        "TERMINAL_PREFLIGHT", "TERMINAL_SUBMIT", "APP_E", "INGEST", "GENERATE",
        "B_RETURN", "A_RETURN", "PAIR_READY", "TERMINAL_PREFLIGHT", "TERMINAL_SUBMIT",
        "STOP_JOIN", "DG2X_P4C_D3B3_FINITE_A_E_SEQUENCE_PASS"};
    std::vector<std::string> observedHighLevel;
    for (const auto& event : run.events) {
        if (std::ranges::find(expectedHighLevel, event) != expectedHighLevel.end())
            observedHighLevel.push_back(event);
    }
    assert(observedHighLevel == expectedHighLevel);

    const std::vector<std::string> expectedPresents{
        "PRESENT_GENERATED_BC", "PRESENT_ORIGINAL_BC",
        "PRESENT_GENERATED_CD", "PRESENT_ORIGINAL_CD",
        "PRESENT_GENERATED_DE", "PRESENT_ORIGINAL_DE"};
    std::vector<std::string> presents;
    for (const auto& event : run.terminal.events)
        if (event.starts_with("PRESENT_GENERATED_") || event.starts_with("PRESENT_ORIGINAL_"))
            presents.push_back(event);
    assert(presents == expectedPresents);
    assert(run.fixture.harness.productionBReturnCalls == 3);
    assert(run.fixture.productionWarmupIngests == 2);
    assert(run.fixture.productionGenerateSourceIngests == 3);
    assert(run.fixture.harness.aPhaseCalls == 3);
    assert(run.fixture.harness.aQueueSubmitCalls == 3);
    assert(run.fixture.harness.aFenceWaitCalls == 0);
    assert(run.fixture.harness.aFenceStatusCalls == 3);
    assert(run.fixture.harness.deviceIdleCalls == 0);
    assert(run.fixture.harness.frontFenceWaitCount() == run.fixture.hostWaitsBeforeGraph);
    assert(run.terminal.terminalSubmitCalls == 3);
    assert(run.terminal.bridgeSubmits == 3
        && run.terminal.bridgedApplicationWaits.size() == 6);
    for (size_t i = 0; i < run.terminal.bridgedApplicationWaits.size(); ++i)
        for (size_t j = i + 1; j < run.terminal.bridgedApplicationWaits.size(); ++j)
            assert(run.terminal.bridgedApplicationWaits[i]
                != run.terminal.bridgedApplicationWaits[j]);
    assert(run.terminal.terminalWaitSemaphore == run.fixture.harness.returnedForGraphics);
    assert(run.terminal.terminalGeneratedSource == run.fixture.harness.aImportedImage);
    assert(run.terminal.acquireCalls == 6);
    assert(std::ranges::count(run.terminal.events, "BLIT_GENERATED_G_TO_HIDDEN") == 3);
    assert(std::ranges::count(run.terminal.events, "BLIT_ORIGINAL_C_TO_HIDDEN") == 3);
}

void testD3B3FiniteABCDEFThroughProductionAdapter() {
    FiniteRealD3B3Run run;
    constexpr uint64_t swapchainGeneration = 11;
    auto runtime = std::make_unique<lsfgvk::layer::D3B3ProductionRuntimeSession>(
        swapchainGeneration, run.operations());
    lsfgvk::layer::D3B3NormalPresentAdapter adapter({
        .swapchain = pairHandle<VkSwapchainKHR>(0xA601),
        .sourceImage = pairHandle<VkImage>(0xA602),
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {8, 8},
        .presentQueue = pairHandle<VkQueue>(0xA603),
        .presentQueueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
        .originalReady = pairHandle<VkSemaphore>(0xA604),
        .swapchainGeneration = swapchainGeneration,
        .runtimeDevicePairReady = true,
        .exchangeChannelReady = true,
        .terminalReady = true},
        std::move(runtime));

    assert(adapter.structurallyReady() && adapter.constructOperations());
    for (const auto frame : {'A', 'B', 'C', 'D', 'E'})
        assert(adapter.processFrame(static_cast<uint64_t>(frame))
            == lsfgvk::layer::D3B3NormalAdapterResult::READY);

    const auto* controller = adapter.controller();
    assert(controller && controller->finiteStopped());
    const auto counters = controller->finiteCounters();
    assert(counters.applicationFrames == 5 && counters.generate == 3);
    assert(counters.bReturns == 3 && counters.aReturns == 3);
    assert(counters.pairOperations == 3 && counters.terminalSubmits == 3);
    assert(counters.internalPresents == 6 && controller->pairCount() == 3);
    assert(adapter.ownership() == lsfgvk::layer::D3B3PresentOwnership::D3B3_ROUTE);
    assert(adapter.frameSerial() == 6);
    assert(run.fixture.harness.productionBReturnCalls == 3);
    assert(run.fixture.productionWarmupIngests == 2);
    assert(run.fixture.productionGenerateSourceIngests == 3);
    assert(run.fixture.harness.aPhaseCalls == 3);
    assert(run.fixture.harness.aFenceWaitCalls == 0);
    assert(run.terminal.terminalSubmitCalls == 3);
    assert(run.terminal.acquireCalls == 6);
}

void testD3B3ProductionWarmupBlockedRetryDoesNotConsumeB() {
    FiniteRealD3B3Run run;
    constexpr uint64_t swapchainGeneration = 12;
    auto runtime = std::make_unique<lsfgvk::layer::D3B3ProductionRuntimeSession>(
        swapchainGeneration, run.operations());
    lsfgvk::layer::D3B3NormalPresentAdapter adapter({
        .swapchain = pairHandle<VkSwapchainKHR>(0xA611),
        .sourceImage = pairHandle<VkImage>(0xA612),
        .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {8, 8},
        .presentQueue = pairHandle<VkQueue>(0xA613),
        .presentQueueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
        .originalReady = pairHandle<VkSemaphore>(0xA614),
        .swapchainGeneration = swapchainGeneration,
        .runtimeDevicePairReady = true, .exchangeChannelReady = true,
        .terminalReady = true}, std::move(runtime));
    assert(adapter.structurallyReady() && adapter.constructOperations());
    assert(adapter.processFrame(static_cast<uint64_t>('A'))
        == lsfgvk::layer::D3B3NormalAdapterResult::READY);
    run.fixture.harness.backendFenceStatusResult = VK_NOT_READY;
    assert(adapter.processFrame(static_cast<uint64_t>('B'))
        == lsfgvk::layer::D3B3NormalAdapterResult::TEMPORARILY_BLOCKED);
    assert(adapter.controller()->finiteCounters().applicationFrames == 1);
    assert(run.fixture.productionWarmupIngests == 1);
    run.fixture.harness.backendFenceStatusResult = VK_SUCCESS;
    assert(adapter.processFrame(static_cast<uint64_t>('B'))
        == lsfgvk::layer::D3B3NormalAdapterResult::READY);
    assert(adapter.controller()->finiteCounters().applicationFrames == 2);
    assert(run.fixture.productionWarmupIngests == 2);
}

void testD3B3AsyncFiniteCompositionA5FAndRetry() {
    FiniteRealD3B3Run run;
    run.terminal.nonblocking = true;
    run.terminal.graphicsFenceStatus = VK_NOT_READY;
    std::optional<lsfgvk::backend::ReturnedGeneratedOperation> returned;
    std::vector<std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession>> sessions;
    uint32_t bPolls{};
    bool bReady{};

    lsfgvk::layer::D3B3AsyncFiniteOperations operations{
        .submitIngest = [&](uint64_t frame, auto slot, bool seed) {
            if (frame < static_cast<uint64_t>('C'))
                run.fixture.ingestWarmup(frame, slot);
            else
                run.fixture.ingestCurrent(frame, slot);
            run.events.emplace_back(seed ? "ASYNC_INGEST_SEED" : "ASYNC_INGEST");
            return frame < static_cast<uint64_t>('C')
                ? frame : run.fixture.preparedIngest->first;
        },
        .tryRetireIngest = [&] {
            return run.fixture.backend->tryRetireRuntimeIngest(*run.fixture.backendSession)
                    == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED
                ? lsfgvk::layer::D3B3RetirementStatus::RETIRED
                : lsfgvk::layer::D3B3RetirementStatus::TIMEOUT;
        },
        .presentWarmupOriginal = [&](uint64_t frame) {
            run.events.emplace_back("WARMUP_PRESENT_" + std::string(1, char(frame)));
            return true;
        },
        .submitGenerate = [&](auto identity) {
            run.fixture.generateCurrent(identity.olderFrameId, identity.olderSlot,
                identity.newerFrameId, identity.newerSlot);
            run.events.emplace_back("ASYNC_GENERATE_" +
                std::string{char(identity.olderFrameId), char(identity.newerFrameId)});
            return true;
        },
        .tryRetireGenerate = [] {
            return lsfgvk::backend::RuntimeRetirementStatus::RETIRED;
        },
        .submitBReturn = [&] {
            bReady = false;
            run.events.emplace_back("ASYNC_B_RETURN_SUBMIT");
        },
        .tryRetireBReturn = [&] {
            ++bPolls;
            return bReady ? lsfgvk::backend::RuntimeRetirementStatus::RETIRED
                          : lsfgvk::backend::RuntimeRetirementStatus::NOT_READY;
        },
        .submitAReturn = [&](auto identity) {
            auto session = makeReturnSession(run.fixture);
            returned.emplace(makeSubmittedB0COperation(run.fixture, *session));
            assert(returned->identity().generationId == identity.generationId);
            sessions.push_back(std::move(session));
            run.events.emplace_back("ASYNC_A_RETURN_SUBMIT");
        },
        .releaseGeneratedOutput = [&] {
            assert(returned && returned->generatedOutputRetired());
            return lsfgvk::backend::RuntimeRetirementStatus::RETIRED;
        },
        .takeReturnedOperation = [&] {
            assert(returned && returned->terminalReady());
            auto value = std::move(*returned);
            returned.reset();
            return value;
        },
        .makeOriginal = [](const auto& identity) {
            return lsfgvk::layer::D3B3OriginalSourceAuthority(
                {.image = pairHandle<VkImage>(0xE000 + identity.newerFrameId),
                 .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {8, 8},
                 .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 .queueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
                 .virtualImageIndex = uint32_t(identity.newerFrameId),
                 .frameId = identity.newerFrameId,
                 .authorityScope = pairAuthorityScope},
                std::make_shared<const uint8_t>(0));
        },
        .makeTerminal = [&](const auto& identity) {
            return makeFiniteTerminalDispatch(run.terminal, identity);
        }};
    lsfgvk::layer::D3B3AsyncFiniteComposition composition(std::move(operations));

    for (const char frame : {'A', 'B'}) {
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::ACCEPTED);
    }

    for (const char frame : {'C', 'D', 'E'}) {
        const auto submitsBefore = run.terminal.terminalSubmitCalls;
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        for (int retry = 0; retry < 3; ++retry)
            assert(composition.processFrame(uint64_t(frame))
                == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        assert(run.terminal.terminalSubmitCalls == submitsBefore);
        bReady = true;
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        const auto* pair = composition.activePair();
        assert(pair && pair->aReturnState()
            == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
        assert(!pair->aReturnRetired());
        assert(pair->state() == lsfgvk::layer::D3B3PairState::PRESENTS_SUBMITTED);
        assert(run.terminal.terminalSubmitCalls == submitsBefore + 1);
        assert(run.fixture.harness.aFenceWaitCalls == 0);
        for (int retry = 0; retry < 3; ++retry)
            assert(composition.processFrame(uint64_t(frame))
                == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        assert(run.terminal.terminalSubmitCalls == submitsBefore + 1);
        run.terminal.graphicsFenceStatus = VK_SUCCESS;
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        run.fixture.harness.retireAFence();
        const auto result = composition.processFrame(uint64_t(frame));
        assert(result == (frame == 'E' ? lsfgvk::layer::D3B3AsyncProgress::FINISHED
                                      : lsfgvk::layer::D3B3AsyncProgress::ACCEPTED));
        run.terminal.graphicsFenceStatus = VK_NOT_READY;
    }
    const auto counters = composition.counters();
    assert(counters.frames == 5 && counters.generateSubmits == 3);
    assert(counters.bReturnSubmits == 3 && counters.aReturnSubmits == 3);
    assert(counters.pairConstructions == 3 && counters.terminalSubmits == 3);
    assert(counters.pairRetirements == 3);
    assert(run.terminal.terminalSubmitCalls == 3);
    assert(run.fixture.harness.aFenceWaitCalls == 0);
    assert(bPolls >= 12 && run.terminal.graphicsFencePolls >= 12);
    const std::vector<std::string> expectedPresents{
        "PRESENT_GENERATED_BC", "PRESENT_ORIGINAL_BC",
        "PRESENT_GENERATED_CD", "PRESENT_ORIGINAL_CD",
        "PRESENT_GENERATED_DE", "PRESENT_ORIGINAL_DE"};
    std::vector<std::string> presents;
    for (const auto& event : run.terminal.events)
        if (event.starts_with("PRESENT_GENERATED_")
                || event.starts_with("PRESENT_ORIGINAL_"))
            presents.push_back(event);
    assert(presents == expectedPresents);
}

void testD3B3FactoryRunsRealCoreAsyncFiniteAE() {
    FiniteRealD3B3Run run;
    run.terminal.nonblocking = true;
    run.terminal.graphicsFenceStatus = VK_NOT_READY;
    const auto pair = run.fixture.harness.devicePair();
    const auto generationEndpoint = run.fixture.harness.endpoint();
    const auto renderEndpoint = run.fixture.harness.renderEndpoint();
    lsfgvk::layer::D3B3PerSwapchainRuntimeDescriptor descriptor;
    descriptor.present = {
        .swapchain = pairHandle<VkSwapchainKHR>(0xFA63),
        .sourceImage = pairHandle<VkImage>(0xFA64),
        .format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {64, 64},
        .presentQueue = pairHandle<VkQueue>(0xFA65),
        .presentQueueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
        .originalReady = pairHandle<VkSemaphore>(0xFA66),
        .swapchainGeneration = 1, .runtimeDevicePairReady = true,
        .exchangeChannelReady = true, .terminalReady = true};
    descriptor.core = {
        .backend = run.fixture.backend.get(), .devicePair = &pair,
        .exchangeChannel = run.fixture.harness.exchangeChannelOwnerToken(),
        .generationEndpoint = generationEndpoint, .renderEndpoint = renderEndpoint,
        .extent = {64, 64}, .format = VK_FORMAT_B8G8R8A8_UNORM,
        .flow = 1.0F, .performanceMode = false,
        .sourceImage = [](auto) { return pairHandle<VkImage>(0xFA67); }};

    const lsfgvk::layer::D3B3PerSwapchainRuntimeAssembly assembly{
        .bindFiniteOperations = [&](auto&, const auto&) { return run.operations(); },
        .bindRetirementReady = [](auto&) { return [] { return true; }; },
        .bindAsyncFiniteOperations = [&](lsfgvk::layer::D3B3ProductionCoreOwner& core,
                const auto&) {
            return lsfgvk::layer::D3B3AsyncFiniteOperations{
                .submitIngest = [&](uint64_t frame, auto slot, bool seed) {
                    auto payload = run.fixture.harness.exportTestPayload();
                    const auto snapshot = frame < uint64_t('C')
                        ? core.submitWarmup(frame, slot, std::move(payload))
                        : core.submitGenerateSource(frame, slot, std::move(payload));
                    assert(snapshot.submitAccepted);
                    run.events.emplace_back(seed ? "OWNER_INGEST_SEED" : "OWNER_INGEST");
                    return snapshot.epoch;
                },
                .tryRetireIngest = [&] {
                    const auto status = core.tryRetireWarmup();
                    return status == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED
                        ? lsfgvk::layer::D3B3RetirementStatus::RETIRED
                        : lsfgvk::layer::D3B3RetirementStatus::TIMEOUT;
                },
                .presentWarmupOriginal = [](uint64_t) { return true; },
                .submitGenerate = [&](auto identity) {
                    return core.submitGenerate(identity).submitAccepted;
                },
                .tryRetireGenerate = [&] { return core.tryRetireGenerate(); },
                .submitBReturn = [&] {
                    run.fixture.harness.armProductionReturnForOwner();
                    core.submitBReturn();
                },
                .tryRetireBReturn = [&] { return core.tryRetireBReturn(); },
                .submitAReturn = [&](auto) {
                    core.submitAReturn(run.fixture.harness.productionHandoff());
                },
                .releaseGeneratedOutput = [&] { return core.releaseGeneratedOutput(); },
                .takeReturnedOperation = [&] { return core.takeReturnedOperation(); },
                .makeOriginal = [](const auto& identity) {
                    return lsfgvk::layer::D3B3OriginalSourceAuthority(
                        {.image = pairHandle<VkImage>(0xFB00 + identity.newerFrameId),
                         .format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {64, 64},
                         .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         .queueFamily = ShadowBReturnExecutionHarness::terminalQueueFamily,
                         .virtualImageIndex = uint32_t(identity.newerFrameId),
                         .frameId = identity.newerFrameId,
                         .authorityScope = pairAuthorityScope},
                        std::make_shared<const uint8_t>(0));
                },
                .makeTerminal = [&](const auto& identity) {
                    return makeFiniteTerminalDispatch(run.terminal, identity);
                }};
        }};
    lsfgvk::layer::D3B3PerSwapchainRuntimeFactory factory;
    auto owner = factory.create(std::move(descriptor), assembly);
    auto& composition = owner->asyncComposition();
    for (const char frame : {'A', 'B'}) {
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        run.fixture.harness.backendFenceStatusResult = VK_SUCCESS;
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::ACCEPTED);
    }
    for (const char frame : {'C', 'D', 'E'}) {
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        run.fixture.harness.retireSourceFence();
        run.fixture.harness.retireFence();
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        assert(composition.activePair()
            && composition.activePair()->aReturnState()
                == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
        assert(run.fixture.harness.aFenceWaitCalls == 0);
        run.terminal.graphicsFenceStatus = VK_SUCCESS;
        assert(composition.processFrame(uint64_t(frame))
            == lsfgvk::layer::D3B3AsyncProgress::PENDING);
        run.fixture.harness.retireAFence();
        const auto result = composition.processFrame(uint64_t(frame));
        assert(result == (frame == 'E' ? lsfgvk::layer::D3B3AsyncProgress::FINISHED
                                      : lsfgvk::layer::D3B3AsyncProgress::ACCEPTED));
        run.terminal.graphicsFenceStatus = VK_NOT_READY;
    }
    const auto counters = composition.counters();
    assert(counters.frames == 5 && counters.generateSubmits == 3);
    assert(counters.bReturnSubmits == 3 && counters.aReturnSubmits == 3);
    assert(counters.pairConstructions == 3 && counters.terminalSubmits == 3);
    assert(counters.pairRetirements == 3 && run.terminal.terminalSubmitCalls == 3);
    assert(run.terminal.bridgeSubmits == 3
        && run.terminal.bridgedApplicationWaits.size() == 6);
    assert(run.fixture.harness.aFenceWaitCalls == 0);
    assert(owner->retirementReady() && factory.destroy(owner->swapchain()));
}

void testD3B3ApplicationWaitBridgeProvenance() {
    static_assert(!std::is_copy_constructible_v<
        lsfgvk::layer::ApplicationPresentWaitAuthority>);
    static_assert(!std::is_copy_constructible_v<
        lsfgvk::layer::D3B3OriginalReadyAuthority>);
    const auto wait0 = pairHandle<VkSemaphore>(0xAB01);
    const auto wait1 = pairHandle<VkSemaphore>(0xAB02);
    const auto ready = pairHandle<VkSemaphore>(0xAB03);
    const auto queue = pairHandle<VkQueue>(0xAB04);
    uint32_t submits{};
    lsfgvk::layer::ApplicationPresentWaitAuthority input(
        {wait0, wait1}, uint64_t{'C'}, 0xABC1);
    auto bridged = lsfgvk::layer::bridgeApplicationPresentWaits(input, queue,
        {ready}, ready, [&](VkQueue actualQueue, const VkSubmitInfo& submit) {
            ++submits;
            assert(actualQueue == queue && submit.waitSemaphoreCount == 2);
            assert(submit.pWaitSemaphores[0] == wait0
                && submit.pWaitSemaphores[1] == wait1);
            assert(submit.signalSemaphoreCount == 1
                && submit.pSignalSemaphores[0] == ready);
            return VK_SUCCESS;
        });
    assert(submits == 1 && bridged.result == VK_SUCCESS && bridged.originalReady);
    assert(input.state() == lsfgvk::layer::ApplicationPresentWaitState::BRIDGED);
    assert(bridged.originalReady->epoch() == uint64_t{'C'}
        && bridged.originalReady->scope() == 0xABC1);
    auto movedReady = std::move(*bridged.originalReady);
    assert(movedReady.valid() && !bridged.originalReady->valid());
    auto retry = lsfgvk::layer::bridgeApplicationPresentWaits(input, queue,
        {ready}, ready, [&](VkQueue, const VkSubmitInfo&) {
            ++submits;
            return VK_SUCCESS;
        });
    assert(retry.result == VK_ERROR_INITIALIZATION_FAILED
        && !retry.originalReady && submits == 1);

    lsfgvk::layer::ApplicationPresentWaitAuthority rejected(
        {wait0, wait1}, uint64_t{'D'}, 0xABC1);
    auto failure = lsfgvk::layer::bridgeApplicationPresentWaits(rejected, queue,
        {ready}, ready, [&](VkQueue, const VkSubmitInfo&) {
            ++submits;
            return VK_ERROR_DEVICE_LOST;
        });
    assert(failure.result == VK_ERROR_DEVICE_LOST && !failure.originalReady);
    assert(rejected.valid()
        && rejected.state() == lsfgvk::layer::ApplicationPresentWaitState::AVAILABLE);
    assert(submits == 2);

    lsfgvk::layer::ApplicationPresentWaitAuthority swapchain1(
        {pairHandle<VkSemaphore>(0xAC01)}, uint64_t{'E'}, 0xABC1);
    lsfgvk::layer::ApplicationPresentWaitAuthority swapchain2(
        {pairHandle<VkSemaphore>(0xAC02)}, uint64_t{'E'}, 0xABC2);
    const auto ready1 = pairHandle<VkSemaphore>(0xAC03);
    const auto ready2 = pairHandle<VkSemaphore>(0xAC04);
    auto bridge1 = lsfgvk::layer::bridgeApplicationPresentWaits(swapchain1, queue,
        {ready1}, ready1, [](VkQueue, const VkSubmitInfo&) { return VK_SUCCESS; });
    auto bridge2 = lsfgvk::layer::bridgeApplicationPresentWaits(swapchain2, queue,
        {ready2}, ready2, [](VkQueue, const VkSubmitInfo&) { return VK_SUCCESS; });
    assert(bridge1.originalReady && bridge2.originalReady);
    assert(bridge1.originalReady->scope() != bridge2.originalReady->scope());
    assert(bridge1.originalReady->semaphore() != bridge2.originalReady->semaphore());
    bridge1.originalReady->waitSubmitted();
    assert(bridge1.originalReady->state()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_SUBMITTED);
    assert(bridge2.originalReady->state()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED);

    // Pair rejects both stale epochs and a readiness authority from another
    // swapchain scope before any terminal submit can occur.
    for (const auto mismatch : {0, 1}) {
        RealShadowPairFixture fixture;
        PairTerminalMock mock;
        std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
        bool rejectedPair{};
        try {
            static_cast<void>(makePairOperation(fixture, mock, session, {},
                pairAuthorityScope, mismatch == 0 ? pairAuthorityScope : 0xD3B30002,
                mismatch == 0 ? 1 : 0));
        } catch (const std::invalid_argument&) { rejectedPair = true; }
        assert(rejectedPair && mock.terminalSubmitCalls == 0);
    }
}

void testD3B3PairOperationTerminalRetirementContract() {
    static_assert(!std::is_copy_constructible_v<lsfgvk::layer::D3B3PairOperation>);
    static_assert(std::is_move_constructible_v<lsfgvk::layer::D3B3PairOperation>);
    static_assert(!std::is_copy_constructible_v<lsfgvk::layer::D3B3OriginalSourceAuthority>);

    RealShadowPairFixture fixture;
    PairTerminalMock mock;
    std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
    auto pair = makePairOperation(fixture, mock, session);
    const auto identity = pair.identity();
    assert(identity.olderFrameId == static_cast<uint64_t>('B'));
    assert(identity.newerFrameId == static_cast<uint64_t>('C'));
    assert(identity.olderSlot == lsfgvk::backend::TemporalSourceSlot::Slot1);
    assert(identity.newerSlot == lsfgvk::backend::TemporalSourceSlot::Slot0);
    assert(identity.generationId == fixture.generated.generationId);
    assert(pair.state() == lsfgvk::layer::D3B3PairState::NON_TERMINAL_READY);
    assert(pair.returnedForGraphicsState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED);
    assert(pair.returnedOperation().returnedForGraphicsOutstanding());
    assert(pair.returnedOperation().terminalReady());
    assert(pair.returnedImageAlive());
    assert(pair.aReturnState() == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
    assert(pair.aReturnPayloadState()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_SUBMITTED);
    assert(!pair.aReturnRetired());
    assert(pair.originalState() == lsfgvk::layer::D3B3OriginalSourceState::AVAILABLE);
    const auto aWaitsBeforeTerminal = fixture.harness.aFenceWaitCalls;
    const auto aStatusBeforeTerminal = fixture.harness.aFenceStatusCalls;

    pair.preflight();
    assert(pair.state() == lsfgvk::layer::D3B3PairState::TERMINAL_PREFLIGHTED);
    assert(mock.acquireCalls == 0 && fixture.harness.returnedForGraphicsCalls == 1);
    assert(mock.terminalWaitSemaphore == fixture.harness.aObserved.signal);
    assert(mock.terminalWaitSemaphore == pair.returnedOperation().returnedReady());
    assert(mock.terminalGeneratedSource
        == pair.returnedOperation().aReturnPending().imageView().image());
    assert(mock.terminalGeneratedSource == fixture.harness.aImportedImage);
    bool prematureReuseRejected{};
    try { pair.rejectReturnedForGraphicsReuse(); }
    catch (const std::logic_error&) { prematureReuseRejected = true; }
    assert(prematureReuseRejected);

    pair.submitTerminal();
    assert(mock.terminalSubmitCalls == 1);
    assert(fixture.harness.aFenceWaitCalls == aWaitsBeforeTerminal);
    assert(fixture.harness.aFenceStatusCalls == aStatusBeforeTerminal);
    assert(pair.state() == lsfgvk::layer::D3B3PairState::GRAPHICS_RETIRED);
    assert(pair.originalReadyState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_RETIRED);
    assert(pair.returnedForGraphicsState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_RETIRED);
    assert(pair.returnedForGraphicsReusable());
    assert(pair.originalState() == lsfgvk::layer::D3B3OriginalSourceState::TERMINAL_RETIRED);
    assert(pair.returnedImageAlive());
    assert(pair.terminalCommandBufferReusable());
    assert(mock.releaseCalls == 0 && mock.releasedImages == 0);
    assert(fixture.harness.frontFenceWaitCount() == fixture.hostWaitsBeforeGraph);

    pair.retirePresentWaits();
    assert(pair.state() == lsfgvk::layer::D3B3PairState::PRESENTS_RETIRED);
    assert(pair.generatedPresentRetired() && pair.originalPresentRetired());
    assert(pair.hiddenAcquisitionLeasesReleased());
    bool pendingAReturnBlockedPairRetirement{};
    try { pair.retirePair(); }
    catch (const std::logic_error&) { pendingAReturnBlockedPairRetirement = true; }
    assert(pendingAReturnBlockedPairRetirement);
    assert(!pair.tryRetireAReturn());
    assert(fixture.harness.aFenceStatusCalls == aStatusBeforeTerminal + 1);
    assert(fixture.harness.aFenceWaitCalls == aWaitsBeforeTerminal);
    fixture.harness.retireAFence();
    assert(pair.tryRetireAReturn());
    assert(pair.aReturnRetired());
    assert(pair.aReturnState() == lsfgvk::backend::RuntimeAuthorityState::RETIRED);
    assert(pair.aReturnPayloadState()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_RETIRED);
    assert(fixture.harness.aFenceStatusCalls == aStatusBeforeTerminal + 2);
    assert(fixture.harness.aFenceWaitCalls == aWaitsBeforeTerminal);
    pair.retirePair();
    assert(pair.state() == lsfgvk::layer::D3B3PairState::PAIR_RETIRED);
    assert(pair.pairRetired() && pair.originalState() == lsfgvk::layer::D3B3OriginalSourceState::RELEASED);
    assert(!pair.valid());
    assert(mock.events == (std::vector<std::string>{
        "ACQUIRE_G", "ACQUIRE_O", "TERMINAL_RECORD_BEGIN",
        "BLIT_GENERATED_G_TO_HIDDEN", "BLIT_ORIGINAL_C_TO_HIDDEN",
        "TERMINAL_RECORD_END", "TERMINAL_SUBMIT_WAIT_RETURNED_FOR_GRAPHICS",
        "PRESENT_GENERATED", "PRESENT_ORIGINAL", "TERMINAL_GRAPHICS_FENCE_WAIT",
        "PRESENT_FENCE_GENERATED_RETIRED", "PRESENT_FENCE_ORIGINAL_RETIRED"}));
}

void testD3B3SubmittedAReturnTerminalReadinessFirewall() {
    lsfgvk::backend::ReturnedGeneratedOperation empty;
    assert(!empty.terminalReady());

    RealShadowPairFixture fixture;
    auto session = makeReturnSession(fixture);
    auto returned = makeSubmittedB0COperation(fixture, *session);
    const auto generation = returned.identity().generationId;
    const auto ready = returned.returnedReady();
    assert(returned.terminalReady());

    lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setAReturnEpoch(
        returned, generation + 1);
    assert(!returned.terminalReady());
    lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setAReturnEpoch(
        returned, generation);
    assert(returned.terminalReady());

    lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setReturnedReady(
        returned, VK_NULL_HANDLE);
    assert(!returned.terminalReady());
    lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setReturnedReady(
        returned, ready);
    assert(returned.terminalReady());

    returned.consumeReturnedForGraphics();
    assert(!returned.terminalReady());
    lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setReturnedConsumed(
        returned, false);
    assert(returned.terminalReady());

    fixture.harness.retireAFence();
    assert(returned.tryRetireAReturn());
    assert(returned.aReturnRetired());
    assert(returned.terminalReady());
    assert(fixture.harness.aFenceWaitCalls == 0);
    assert(fixture.harness.aFenceStatusCalls == 1);
}

void testD3B3PairConstructionRejectsInvalidSubmittedState() {
    using Returned = lsfgvk::backend::ReturnedGeneratedOperation;
    const std::vector<std::function<void(Returned&)>> invalidStates{
        [](Returned& returned) { returned.failAReturn(); },
        [](Returned& returned) {
            lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setReturnedReady(
                returned, VK_NULL_HANDLE);
        },
        [](Returned& returned) { returned.consumeReturnedForGraphics(); },
        [](Returned& returned) {
            lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setAReturnEpoch(
                returned, returned.identity().generationId + 1);
        },
        [](Returned& returned) {
            lsfgvk::backend::RuntimeOperationAuthorityTestAccess::setPayloadEpoch(
                returned, returned.identity().generationId + 1);
        },
        [](Returned& returned) { returned.aReturnPending() = {}; }};

    for (const auto& mutate : invalidStates) {
        RealShadowPairFixture fixture;
        PairTerminalMock mock;
        std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
        bool rejected{};
        try { static_cast<void>(makePairOperation(fixture, mock, session, mutate)); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
        assert(mock.terminalSubmitCalls == 0);
        assert(fixture.harness.aFenceWaitCalls == 0);
        assert(fixture.harness.aFenceStatusCalls == 0);
    }
}

void testD3B3AReturnObservedBeforeTerminalFence() {
    RealShadowPairFixture fixture;
    PairTerminalMock mock;
    std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
    auto pair = makePairOperation(fixture, mock, session);
    const auto aWaitsBeforeTerminal = fixture.harness.aFenceWaitCalls;
    bool aObservedBeforeTerminalFence{};
    mock.beforeGraphicsFenceRetirement = [&] {
        assert(mock.terminalSubmitCalls == 1);
        assert(pair.state() == lsfgvk::layer::D3B3PairState::PRESENTS_SUBMITTED);
        assert(pair.returnedForGraphicsState()
            == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_SUBMITTED);
        fixture.harness.retireAFence();
        assert(pair.tryRetireAReturn());
        assert(pair.aReturnRetired());
        assert(pair.returnedForGraphicsState()
            == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_SUBMITTED);
        aObservedBeforeTerminalFence = true;
    };

    pair.preflight();
    pair.submitTerminal();
    assert(aObservedBeforeTerminalFence);
    assert(pair.state() == lsfgvk::layer::D3B3PairState::GRAPHICS_RETIRED);
    assert(pair.returnedForGraphicsState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_RETIRED);
    assert(fixture.harness.aFenceWaitCalls == aWaitsBeforeTerminal);
    assert(fixture.harness.aFenceStatusCalls == 1);
    pair.retirePresentWaits();
    pair.retirePair();
}

void testD3B3LateAReturnFailureIsConservative() {
    RealShadowPairFixture fixture;
    PairTerminalMock mock;
    std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
    auto pair = makePairOperation(fixture, mock, session);
    pair.preflight();
    pair.submitTerminal();
    fixture.harness.aFenceStatusResult = VK_ERROR_DEVICE_LOST;
    bool deviceLost{};
    try { static_cast<void>(pair.tryRetireAReturn()); }
    catch (const ls::vulkan_error& error) {
        deviceLost = error.error() == VK_ERROR_DEVICE_LOST;
    }
    assert(deviceLost && pair.state() == lsfgvk::layer::D3B3PairState::FAILED);
    assert(pair.aReturnState() == lsfgvk::backend::RuntimeAuthorityState::FAILED);
    assert(pair.aReturnPayloadState()
        == lsfgvk::backend::RuntimeTemporaryPayloadState::FAILED);
    assert(pair.returnedForGraphicsState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_RETIRED);
    assert(fixture.harness.aFenceWaitCalls == 0);
    assert(fixture.harness.aFenceStatusCalls == 1);
}

void testD3B3A5GPresentAcceptanceAndNonblockingRetirement() {
    RealShadowPairFixture fixture;
    PairTerminalMock mock;
    mock.nonblocking = true;
    mock.graphicsFenceStatus = VK_SUCCESS;
    mock.generatedPresentFenceStatus = VK_NOT_READY;
    mock.originalPresentFenceStatus = VK_NOT_READY;
    std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
    auto pair = makePairOperation(fixture, mock, session);
    pair.preflight();
    pair.submitTerminal();
    assert(pair.state() == lsfgvk::layer::D3B3PairState::PRESENTS_SUBMITTED);
    assert(pair.originalReadyState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_SUBMITTED);
    assert(pair.originalReadySemaphore() == pairHandle<VkSemaphore>(0xD300));
    assert(std::ranges::none_of(mock.bridgedApplicationWaits,
        [&](VkSemaphore wait) { return wait == pair.originalReadySemaphore(); }));
    assert(mock.bridgeSubmits == 1 && mock.bridgedApplicationWaits.size() == 2);
    assert(pair.hiddenAcquisitionLeasesReleased());
    assert(mock.terminalSubmitCalls == 1);
    assert(pair.tryRetireTerminal()
        == lsfgvk::layer::D3B2RetirementResult::RETIRED);
    assert(pair.state() == lsfgvk::layer::D3B3PairState::GRAPHICS_RETIRED);
    assert(pair.originalReadyState()
        == lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_RETIRED);
    for (int poll = 0; poll < 3; ++poll)
        assert(pair.tryRetirePresentWaits()
            == lsfgvk::layer::D3B2RetirementResult::NOT_READY);
    assert(mock.terminalSubmitCalls == 1);
    assert(mock.generatedPresentFencePolls == 3);
    assert(mock.originalPresentFencePolls == 0);
    mock.generatedPresentFenceStatus = VK_SUCCESS;
    assert(pair.tryRetirePresentWaits()
        == lsfgvk::layer::D3B2RetirementResult::NOT_READY);
    assert(pair.generatedPresentRetired() && !pair.originalPresentRetired());
    mock.originalPresentFenceStatus = VK_SUCCESS;
    assert(pair.tryRetirePresentWaits()
        == lsfgvk::layer::D3B2RetirementResult::RETIRED);
    assert(pair.state() == lsfgvk::layer::D3B3PairState::PRESENTS_RETIRED);
    assert(mock.terminalSubmitCalls == 1);
}

void testD3B3NonblockingPresentRetirementFailuresAreSticky() {
    for (const bool generated : {true, false}) {
        RealShadowPairFixture fixture;
        PairTerminalMock mock;
        mock.nonblocking = true;
        mock.graphicsFenceStatus = VK_SUCCESS;
        mock.generatedPresentFenceStatus = generated
            ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
        mock.originalPresentFenceStatus = generated
            ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
        std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
        auto pair = makePairOperation(fixture, mock, session);
        pair.preflight();
        pair.submitTerminal();
        assert(pair.tryRetireTerminal()
            == lsfgvk::layer::D3B2RetirementResult::RETIRED);
        assert(pair.tryRetirePresentWaits()
            == (generated ? lsfgvk::layer::D3B2RetirementResult::DEVICE_LOST
                          : lsfgvk::layer::D3B2RetirementResult::FAILED));
        assert(pair.state() == lsfgvk::layer::D3B3PairState::FAILED);
        assert(mock.terminalSubmitCalls == 1);
    }
}

void testD3B3PairOperationFailureBoundaries() {
    const std::vector<PairTerminalFailure> failures{
        PairTerminalFailure::PREFLIGHT, PairTerminalFailure::ACQUIRE_GENERATED,
        PairTerminalFailure::ACQUIRE_ORIGINAL, PairTerminalFailure::RECORD,
        PairTerminalFailure::SUBMIT, PairTerminalFailure::DEVICE_LOST,
        PairTerminalFailure::GENERATED_PRESENT, PairTerminalFailure::ORIGINAL_PRESENT,
        PairTerminalFailure::GRAPHICS_FENCE};
    for (const auto failure : failures) {
        RealShadowPairFixture fixture;
        PairTerminalMock mock{.failure = failure};
        std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
        auto pair = makePairOperation(fixture, mock, session);
        bool failed{};
        try {
            pair.preflight();
            pair.submitTerminal();
        } catch (const std::exception&) {
            failed = true;
        }
        assert(failed && pair.state() == lsfgvk::layer::D3B3PairState::FAILED);
        const bool submitWasAccepted = failure == PairTerminalFailure::GENERATED_PRESENT
            || failure == PairTerminalFailure::ORIGINAL_PRESENT
            || failure == PairTerminalFailure::GRAPHICS_FENCE;
        assert(pair.returnedForGraphicsState()
            == (submitWasAccepted
                ? lsfgvk::backend::RuntimeBinarySemaphoreState::WAIT_SUBMITTED
                : lsfgvk::backend::RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED));
        assert(pair.returnedOperation().returnedForGraphicsOutstanding()
            == !submitWasAccepted);
        assert(pair.aReturnState() == lsfgvk::backend::RuntimeAuthorityState::SUBMITTED);
        assert(pair.aReturnPayloadState()
            == lsfgvk::backend::RuntimeTemporaryPayloadState::WAIT_SUBMITTED);
        if (failure == PairTerminalFailure::ACQUIRE_ORIGINAL)
            assert(mock.releaseCalls == 1 && mock.releasedImages == 1);
        if (failure == PairTerminalFailure::ACQUIRE_GENERATED
                || failure == PairTerminalFailure::PREFLIGHT)
            assert(mock.releaseCalls == 0);
        if (failure == PairTerminalFailure::RECORD
                || failure == PairTerminalFailure::SUBMIT)
            assert(mock.releaseCalls == 1 || mock.releaseCalls == 0);
        if (failure == PairTerminalFailure::DEVICE_LOST)
            assert(mock.releaseCalls == 0);
        fixture.harness.retireAFence();
        assert(pair.tryRetireAReturn());
        assert(pair.state() == lsfgvk::layer::D3B3PairState::FAILED);
    }

    RealShadowPairFixture fixture;
    PairTerminalMock mock;
    std::unique_ptr<lsfgvk::layer::GeneratedOutputReturnSession> session;
    auto pair = makePairOperation(fixture, mock, session);
    pair.preflight();
    pair.submitTerminal();
    bool doubleSubmitRejected{};
    try { pair.submitTerminal(); }
    catch (const std::logic_error&) { doubleSubmitRejected = true; }
    assert(doubleSubmitRejected);
    pair.retirePresentWaits();
    assert(pair.hiddenAcquisitionLeasesReleased());
    fixture.harness.retireAFence();
    assert(pair.tryRetireAReturn());
    pair.retirePair();
}

struct HiddenWsiLeaseModel {
    explicit HiddenWsiLeaseModel(uint32_t count) : acquired(count), presentPending(count) {}

    bool acquire(uint32_t index) {
        if (index >= acquired.size() || acquired[index])
            return false;
        acquired[index] = true;
        return true;
    }

    bool present(uint32_t index, VkResult result = VK_SUCCESS) {
        if (index >= acquired.size() || !acquired[index] || result != VK_SUCCESS)
            return false;
        acquired[index] = false;
        presentPending[index] = true;
        return true;
    }

    bool release(uint32_t index) {
        if (index >= acquired.size() || !acquired[index])
            return false;
        acquired[index] = false;
        return true;
    }

    void retirePresentFence(uint32_t index) {
        assert(index < presentPending.size() && presentPending[index]);
        presentPending[index] = false;
    }

    bool acquisitionLeasesReleased() const {
        return std::ranges::none_of(acquired, [](bool value) { return value; });
    }

    bool presentPendingFor(uint32_t index) const { return presentPending[index]; }

    std::vector<bool> acquired;
    std::vector<bool> presentPending;
};

void testD3B3HiddenWsiLeaseRetirementModel() {
    // Exact R3 shape: two presented images, while future acquire returns image 2 forever.
    HiddenWsiLeaseModel r3(3);
    assert(r3.acquire(0) && r3.acquire(1));
    assert(r3.present(0) && r3.present(1));
    assert(r3.acquisitionLeasesReleased());
    assert(r3.presentPendingFor(0) && r3.presentPendingFor(1));
    assert(r3.acquire(2) && r3.release(2));
    r3.retirePresentFence(0);
    r3.retirePresentFence(1);
    assert(r3.acquisitionLeasesReleased());

    // Pair 2 owns a fresh, independently acquired pair; its legal indices need not match Pair 1.
    HiddenWsiLeaseModel pair2(3);
    assert(pair2.acquire(2) && pair2.acquire(1));
    assert(pair2.present(2) && pair2.present(1));
    assert(pair2.acquisitionLeasesReleased());

    for (const uint32_t count : {2u, 3u, 4u}) {
        HiddenWsiLeaseModel model(count);
        assert(model.acquire(0) && model.acquire(1));
        assert(model.present(0) && model.present(1));
        assert(model.acquisitionLeasesReleased());
    }

    // Present-resource retirement is independent from acquisition-lease retirement.
    HiddenWsiLeaseModel fenceIndependent(3);
    assert(fenceIndependent.acquire(0) && fenceIndependent.acquire(1));
    assert(fenceIndependent.present(0) && fenceIndependent.present(1));
    assert(fenceIndependent.acquisitionLeasesReleased());
    assert(fenceIndependent.presentPendingFor(0));
    fenceIndependent.retirePresentFence(0);
    fenceIndependent.retirePresentFence(1);

    // Failed present leaves that image application-owned until explicit release cleanup.
    HiddenWsiLeaseModel failure(3);
    assert(failure.acquire(0) && failure.acquire(1));
    assert(failure.present(0));
    assert(!failure.present(1, VK_ERROR_OUT_OF_DATE_KHR));
    assert(!failure.acquisitionLeasesReleased());
    assert(failure.release(1));
    assert(failure.acquisitionLeasesReleased());
}

}

int main() {
    static_assert(!std::is_copy_constructible_v<ShadowBReturnExecutionHarness>);
    static_assert(!std::is_copy_constructible_v<
        lsfgvk::backend::RuntimeGenerateDiagnosticPending>);
    static_assert(!std::is_copy_constructible_v<vk::SyncFdPayload>);
    static_assert(std::is_move_constructible_v<vk::SyncFdPayload>);
    static_assert(std::is_move_assignable_v<vk::SyncFdPayload>);

    testQueueSubmitObserverInIsolation();
    testCompleteResourceAndCommandSurface();
    testFailureCleanup();
    testSyncFdPayloadMoveRegression();
    testIndependentFenceRetirement();
    testRealB0B3PendingStaging();
    testProductionWarmupNoSignalAndBCTransition();
    testProductionNonblockingGenerateRetirement();
    testProductionGenerateRetirementDeviceLost();
    testProductionBReturnAndOutputRetirement();
    testProductionCoreOwnerLifecycle();
    testProductionCoreOwnerMissingDependencies();
    testCompleteOnePairNonTerminalProductionGraph();
    testBReturnSubmitFailures();
    testBReturnExportFailureAfterAcceptedSubmit();
    testBReturnFenceRetirementFailure();
    testAReturnFailureMatrix();
    testAReturnFenceRetirementFailure();
    testD3B3FiniteABCDEFProductionState();
    testD3B3FiniteABCDEFThroughProductionAdapter();
    testD3B3ProductionWarmupBlockedRetryDoesNotConsumeB();
    testD3B3AsyncFiniteCompositionA5FAndRetry();
    testD3B3FactoryRunsRealCoreAsyncFiniteAE();
    testD3B3ApplicationWaitBridgeProvenance();
    testD3B3SubmittedAReturnTerminalReadinessFirewall();
    testD3B3PairConstructionRejectsInvalidSubmittedState();
    testD3B3PairOperationTerminalRetirementContract();
    testD3B3AReturnObservedBeforeTerminalFence();
    testD3B3LateAReturnFailureIsConservative();
    testD3B3A5GPresentAcceptanceAndNonblockingRetirement();
    testD3B3NonblockingPresentRetirementFailuresAreSticky();
    testD3B3PairOperationFailureBoundaries();
    testD3B3HiddenWsiLeaseRetirementModel();
    return 0;
}
