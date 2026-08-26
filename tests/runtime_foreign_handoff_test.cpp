#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace {
template<class T> T h(uintptr_t v) { return reinterpret_cast<T>(v); }
struct BarrierBatch {
    VkPipelineStageFlags srcStage{}, dstStage{};
    std::vector<VkImageMemoryBarrier> barriers;
};
struct BlitCall {
    VkImage source{}, destination{};
    VkImageLayout sourceLayout{}, destinationLayout{};
    VkFilter filter{};
};
struct Trace {
    std::vector<std::string> events;
    std::vector<VkImageMemoryBarrier> barriers;
    std::vector<BarrierBatch> barrierBatches;
    std::vector<BlitCall> blits;
    VkPipelineStageFlags srcStage{}, dstStage{};
    VkSubmitInfo submit{};
    std::vector<VkSemaphore> waits, signals;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkFence> submitFences;
    std::vector<VkSemaphore> destroyedSemaphores;
    uint32_t submits{}, waitsOnFence{}, destroys{}, imagesDestroyed{}, memoriesFreed{},
        buffersDestroyed{}, poolsDestroyed{}, commandsFreed{}, fencesDestroyed{},
        fenceStatusChecks{}, fenceCreates{}, fenceResets{}, fdExports{}, begins{}, ends{},
        semaphoreCreates{};
    VkResult submitResult{VK_SUCCESS};
    VkResult fenceResult{VK_SUCCESS};
    VkResult waitResult{VK_SUCCESS};
    VkResult fenceStatusResult{VK_NOT_READY};
    VkResult invalidateResult{VK_SUCCESS};
    uint8_t* mapped{};
} *trace;

VkResult begin(VkCommandBuffer, const VkCommandBufferBeginInfo*) { ++trace->begins; trace->events.emplace_back("begin"); return VK_SUCCESS; }
VkResult end(VkCommandBuffer) { ++trace->ends; trace->events.emplace_back("end"); return VK_SUCCESS; }
void barrier(VkCommandBuffer, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags,
             uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*,
             uint32_t n, const VkImageMemoryBarrier* b) {
    trace->events.emplace_back("barrier"); trace->srcStage=s; trace->dstStage=d;
    BarrierBatch batch{.srcStage=s, .dstStage=d};
    for (uint32_t i=0;i<n;++i) trace->barriers.push_back(b[i]);
    for (uint32_t i=0;i<n;++i) batch.barriers.push_back(b[i]);
    trace->barrierBatches.push_back(std::move(batch));
}
void copy(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy*) { trace->events.emplace_back("copy"); }
void blit(VkCommandBuffer, VkImage source, VkImageLayout sourceLayout,
        VkImage destination, VkImageLayout destinationLayout, uint32_t,
        const VkImageBlit*, VkFilter filter) {
    trace->events.emplace_back("blit");
    trace->blits.push_back({source, destination, sourceLayout, destinationLayout, filter});
}
VkResult fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* f) { ++trace->fenceCreates; if (trace->fenceResult == VK_SUCCESS) *f=h<VkFence>(0xf1); return trace->fenceResult; }
void destroyFence(VkDevice, VkFence, const VkAllocationCallbacks*) { ++trace->fencesDestroyed; }
VkResult fenceStatus(VkDevice, VkFence) { ++trace->fenceStatusChecks; return trace->fenceStatusResult; }
VkResult wait(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) { ++trace->waitsOnFence; trace->events.emplace_back("wait"); return trace->waitResult; }
VkResult resetFences(VkDevice, uint32_t, const VkFence*) { ++trace->fenceResets; trace->events.emplace_back("reset-fence"); return VK_SUCCESS; }
VkResult submit(VkQueue, uint32_t n, const VkSubmitInfo* s, VkFence submitFence) {
    ++trace->submits; trace->submit=*s; trace->waits.assign(s->pWaitSemaphores,s->pWaitSemaphores+s->waitSemaphoreCount); trace->signals.assign(s->pSignalSemaphores,s->pSignalSemaphores+s->signalSemaphoreCount);
    trace->waitStages.assign(s->pWaitDstStageMask,
        s->pWaitDstStageMask ? s->pWaitDstStageMask+s->waitSemaphoreCount : s->pWaitDstStageMask);
    trace->submitFences.push_back(submitFence); trace->events.emplace_back("submit"); return trace->submitResult;
}
VkResult createSem(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* s) { ++trace->semaphoreCreates; *s=h<VkSemaphore>(0xe1); return VK_SUCCESS; }
void destroySem(VkDevice, VkSemaphore value, const VkAllocationCallbacks*) { ++trace->destroys; trace->destroyedSemaphores.push_back(value); }
void destroyImage(VkDevice, VkImage, const VkAllocationCallbacks*) { ++trace->imagesDestroyed; }
void destroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) { ++trace->buffersDestroyed; }
void freeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++trace->memoriesFreed; }
void destroyPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) { ++trace->poolsDestroyed; }
void freeCommands(VkDevice, VkCommandPool, uint32_t n, const VkCommandBuffer*) { trace->commandsFreed += n; }
VkResult importFd(VkDevice, const VkImportSemaphoreFdInfoKHR* i) { if(i->fd>=0) close(i->fd); return VK_SUCCESS; }
VkResult getFd(VkDevice, const VkSemaphoreGetFdInfoKHR*, int* fd) { ++trace->fdExports; int p[2]; assert(pipe(p)==0); close(p[1]); *fd=p[0]; return VK_SUCCESS; }
VkResult invalidate(VkDevice, uint32_t, const VkMappedMemoryRange*) { trace->events.emplace_back("invalidate"); return trace->invalidateResult; }
void unmap(VkDevice, VkDeviceMemory) { delete[] trace->mapped; trace->mapped=nullptr; }

vk::RuntimeExchangeEndpoint endpoint(uint32_t family) {
    VkPhysicalDeviceMemoryProperties p{}; p.memoryTypeCount=1;
    return {
        .bufferDevice={.device=h<VkDevice>(0x1), .memoryProperties=p,
            .funcs={.DestroyBuffer=destroyBuffer, .FreeMemory=freeMemory}},
        .semaphoreDevice={.device=h<VkDevice>(0x1),
            .funcs={.CreateSemaphore=createSem, .DestroySemaphore=destroySem,
                .GetSemaphoreFdKHR=getFd, .ImportSemaphoreFdKHR=importFd}},
        .queue=h<VkQueue>(0x2),
        .queueFamilyIndex=family,
        .QueueSubmit=submit,
        .CreateFence=fence,
        .DestroyFence=destroyFence,
        .GetFenceStatus=fenceStatus,
        .WaitForFences=wait,
        .ResetFences=resetFences,
        .DestroyCommandPool=destroyPool,
        .FreeCommandBuffers=freeCommands,
        .BeginCommandBuffer=begin,
        .EndCommandBuffer=end,
        .CmdPipelineBarrier=barrier,
        .UnmapMemory=unmap,
        .InvalidateMappedMemoryRanges=invalidate,
        .DestroyImage=destroyImage,
        .CmdCopyImageToBuffer=copy,
        .CmdBlitImage=blit};
}
}

namespace vk {
struct RuntimeImageEndpointTestAccess {
    static RuntimeImageEndpoint make(RuntimeExchangeEndpoint ep, uint32_t family, Trace& t) {
        ::trace=&t; ep.queueFamilyIndex=family; RuntimeImageEndpoint out;
        out.endpoint=ep; out.imageHandle=h<VkImage>(0x111); out.memory=ImportedExternalMemory(
            ep.bufferDevice.device, freeMemory, h<VkDeviceMemory>(0x555), 0, false);
        out.extent={1,1}; out.format=VK_FORMAT_R8G8B8A8_UNORM;
        out.commandPool=h<VkCommandPool>(0x222); out.commandBuffers={h<VkCommandBuffer>(0x333)};
        out.stagingBuffer=h<VkBuffer>(0x444); out.stagingMemory=h<VkDeviceMemory>(0x556); out.stagingSize=4;
        t.mapped=new uint8_t[4]{1,2,3,4}; out.stagingMapped=t.mapped; out.stagingHostCoherent=false;
        return out;
    }
};

struct RuntimeFrameTransportSubmissionTestAccess {
    static SyncFdPayload consumerSubmitted(RuntimeFrameTransportSubmission& submission) {
        auto payload=submission.releasePayload();
        assert(payload.valid());
        submission.consumerSubmitted();
        return payload;
    }
    static void consumerRetired(RuntimeFrameTransportSubmission& submission) {
        submission.consumerRetired();
    }
};
}

namespace {
vk::RuntimeFrameTransportSource frameSource(VkImageLayout layout, uint64_t generation,
        const std::shared_ptr<const uint8_t>& lifetime,
        VkExtent2D extent={1,1}, VkFormat format=VK_FORMAT_R8G8B8A8_UNORM) {
    return {
        .image=h<VkImage>(0x777),
        .currentLayout=layout,
        .format=format,
        .extent=extent,
        .generation=generation,
        .lifetime=lifetime
    };
}

void assertPresentTransportTrace(const Trace& t, VkImage exchange,
        const vk::RuntimeFrameTransportSource& source, VkSemaphore bridgeWait) {
    assert(t.events.size()==6 && t.events[0]=="begin" && t.events[1]=="barrier"
        && t.events[2]=="blit" && t.events[3]=="barrier" && t.events[4]=="end"
        && t.events[5]=="submit");
    assert(t.barrierBatches.size()==2 && t.barrierBatches[0].barriers.size()==2
        && t.barrierBatches[1].barriers.size()==2);
    const auto& sourceAcquire=t.barrierBatches[0].barriers[0];
    const auto& exchangeAcquire=t.barrierBatches[0].barriers[1];
    const auto& sourceRestore=t.barrierBatches[1].barriers[0];
    const auto& exchangeRelease=t.barrierBatches[1].barriers[1];
    assert(t.barrierBatches[0].srcStage==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        && t.barrierBatches[0].dstStage==VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(t.barrierBatches[1].srcStage==VK_PIPELINE_STAGE_TRANSFER_BIT
        && t.barrierBatches[1].dstStage==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    assert(sourceAcquire.image==source.image
        && sourceAcquire.oldLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        && sourceAcquire.newLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && sourceAcquire.srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && sourceAcquire.dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && sourceAcquire.dstAccessMask==VK_ACCESS_TRANSFER_READ_BIT);
    assert(sourceRestore.image==source.image
        && sourceRestore.oldLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && sourceRestore.newLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        && sourceRestore.srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && sourceRestore.dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && sourceRestore.srcAccessMask==VK_ACCESS_TRANSFER_READ_BIT);
    assert(exchangeAcquire.image==exchange
        && exchangeAcquire.oldLayout==VK_IMAGE_LAYOUT_UNDEFINED
        && exchangeAcquire.newLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        && exchangeAcquire.srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && exchangeAcquire.dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && exchangeAcquire.dstAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(exchangeRelease.image==exchange
        && exchangeRelease.oldLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        && exchangeRelease.newLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && exchangeRelease.srcQueueFamilyIndex==3
        && exchangeRelease.dstQueueFamilyIndex==VK_QUEUE_FAMILY_FOREIGN_EXT
        && exchangeRelease.srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(t.blits.size()==1 && t.blits[0].source==source.image
        && t.blits[0].destination==exchange
        && t.blits[0].sourceLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && t.blits[0].destinationLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        && t.blits[0].filter==VK_FILTER_NEAREST);
    assert(t.submits==1 && t.submit.waitSemaphoreCount==1
        && t.waits.size()==1 && t.waits[0]==bridgeWait
        && t.waitStages.size()==1 && t.waitStages[0]==VK_PIPELINE_STAGE_TRANSFER_BIT
        && t.submit.signalSemaphoreCount==1 && t.signals.size()==1
        && t.fdExports==1);
}

void testFrameTransportPresentAndWsiWrapper() {
    static_assert(!std::is_copy_constructible_v<vk::RuntimeFrameTransportSubmission>);
    static_assert(!std::is_copy_assignable_v<vk::RuntimeFrameTransportSubmission>);
    static_assert(std::is_nothrow_move_constructible_v<vk::RuntimeFrameTransportSubmission>);
    const auto bridge=h<VkSemaphore>(0xc9);
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto lifetime=std::make_shared<const uint8_t>(0); const auto source=frameSource(
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,41,lifetime);
        auto result=vk::RuntimeImageEndpoint::trySubmitFrameTransportA(image,source,bridge);
        assert(result.status==vk::RuntimeFrameTransportSubmitStatus::SUBMITTED
            && result.submission.valid() && result.submission.generation()==41
            && result.submission.epoch()==1
            && result.submission.state()==vk::RuntimeFrameTransportState::A_SUBMITTED
            && image.frameTransportStateValue()==vk::RuntimeFrameTransportState::A_SUBMITTED);
        assertPresentTransportTrace(t,image.image(),source,bridge);
        auto payload=vk::RuntimeFrameTransportSubmissionTestAccess::consumerSubmitted(
            result.submission);
        assert(payload.valid()
            && image.frameTransportStateValue()==vk::RuntimeFrameTransportState::B_WAIT_SUBMITTED);
        vk::RuntimeFrameTransportSubmissionTestAccess::consumerRetired(result.submission);
        assert(!result.submission.valid()
            && image.frameTransportStateValue()==vk::RuntimeFrameTransportState::REUSABLE);
    }
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto lifetime=std::make_shared<const uint8_t>(0); const auto source=frameSource(
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,42,lifetime,{1,1},
            VK_FORMAT_B8G8R8A8_UNORM);
        auto submission=vk::RuntimeImageEndpoint::submitRealFrameTransportA(
            image,source,bridge);
        assert(submission.valid() && submission.generation()==42
            && submission.state()==vk::RuntimeFrameTransportState::A_SUBMITTED);
        // The throwing WSI wrapper must preserve the exact PRESENT transition,
        // restore, bridge wait, blit, and release semantics of the generic core.
        assertPresentTransportTrace(t,image.image(),source,bridge);
        auto payload=vk::RuntimeFrameTransportSubmissionTestAccess::consumerSubmitted(submission);
        assert(payload.valid());
        vk::RuntimeFrameTransportSubmissionTestAccess::consumerRetired(submission);
    }
}

void testFrameTransportRejectsBeforeSubmit() {
    auto rejected=[](VkImageLayout layout, VkExtent2D extent, VkFormat format) {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto lifetime=std::make_shared<const uint8_t>(0);
        const auto source=frameSource(layout,1,lifetime,extent,format);
        bool failed=false;
        try { static_cast<void>(vk::RuntimeImageEndpoint::trySubmitFrameTransportA(
            image,source)); } catch (const std::exception&) { failed=true; }
        assert(failed && t.submits==0 && t.fdExports==0 && t.fenceResets==0
            && t.semaphoreCreates==0 && t.begins==0 && t.barrierBatches.empty()
            && t.blits.empty());
    };
    rejected(VK_IMAGE_LAYOUT_UNDEFINED,{1,1},VK_FORMAT_R8G8B8A8_UNORM);
    rejected(VK_IMAGE_LAYOUT_GENERAL,{1,1},VK_FORMAT_R8G8B8A8_UNORM);
    rejected(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,{2,1},VK_FORMAT_R8G8B8A8_UNORM);
    rejected(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,{1,1},VK_FORMAT_UNDEFINED);
    rejected(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,{1,1},VK_FORMAT_R16G16B16A16_SFLOAT);
}

void testFrameTransportBackpressureAndReuse() {
    Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
    auto lifetime1=std::make_shared<const uint8_t>(0); const auto firstSource=frameSource(
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,1,lifetime1);
    auto first=vk::RuntimeImageEndpoint::trySubmitFrameTransportA(image,firstSource);
    assert(first.status==vk::RuntimeFrameTransportSubmitStatus::SUBMITTED
        && first.submission.valid() && first.submission.epoch()==1
        && first.submission.state()==vk::RuntimeFrameTransportState::A_SUBMITTED);
    assert(t.barrierBatches.size()==2 && t.barrierBatches[0].barriers.size()==2
        && t.barrierBatches[1].barriers.size()==2);
    assert(t.barrierBatches[0].barriers[0].oldLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && t.barrierBatches[0].barriers[0].newLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && t.barrierBatches[1].barriers[0].oldLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && t.barrierBatches[1].barriers[0].newLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(t.barrierBatches[0].barriers[1].oldLayout==VK_IMAGE_LAYOUT_UNDEFINED
        && t.barrierBatches[0].barriers[1].srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED
        && t.barrierBatches[0].barriers[1].dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED);

    auto lifetime2=std::make_shared<const uint8_t>(0); const auto secondSource=frameSource(
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,2,lifetime2);
    const auto submits=t.submits;
    const auto fdExports=t.fdExports;
    const auto fenceResets=t.fenceResets;
    const auto begins=t.begins;
    const auto barrierBatchCount=t.barrierBatches.size();
    auto blocked=vk::RuntimeImageEndpoint::trySubmitFrameTransportA(image,secondSource);
    assert(blocked.status==vk::RuntimeFrameTransportSubmitStatus::TEMPORARILY_BLOCKED
        && t.submits==submits && t.fdExports==fdExports && t.fenceResets==fenceResets
        && t.begins==begins && t.barrierBatches.size()==barrierBatchCount);

    auto firstPayload=vk::RuntimeFrameTransportSubmissionTestAccess::consumerSubmitted(
        first.submission);
    assert(firstPayload.valid()
        && image.frameTransportStateValue()==vk::RuntimeFrameTransportState::B_WAIT_SUBMITTED);
    auto blockedUntilBRetires=vk::RuntimeImageEndpoint::trySubmitFrameTransportA(
        image,secondSource);
    assert(blockedUntilBRetires.status
            ==vk::RuntimeFrameTransportSubmitStatus::TEMPORARILY_BLOCKED
        && t.submits==submits && t.fdExports==fdExports && t.fenceResets==fenceResets
        && t.begins==begins && t.barrierBatches.size()==barrierBatchCount);

    vk::RuntimeFrameTransportSubmissionTestAccess::consumerRetired(first.submission);
    assert(image.frameTransportStateValue()==vk::RuntimeFrameTransportState::REUSABLE);
    auto second=vk::RuntimeImageEndpoint::trySubmitFrameTransportA(image,secondSource);
    assert(second.status==vk::RuntimeFrameTransportSubmitStatus::SUBMITTED
        && second.submission.valid() && second.submission.epoch()==2
        && second.submission.generation()==2 && t.submits==submits+1
        && t.fdExports==fdExports+1 && t.barrierBatches.size()==barrierBatchCount+2);
    const auto& reuseAcquire=t.barrierBatches[barrierBatchCount].barriers[1];
    assert(reuseAcquire.image==image.image()
        && reuseAcquire.oldLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && reuseAcquire.newLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        && reuseAcquire.srcQueueFamilyIndex==VK_QUEUE_FAMILY_FOREIGN_EXT
        && reuseAcquire.dstQueueFamilyIndex==3);
    auto secondPayload=vk::RuntimeFrameTransportSubmissionTestAccess::consumerSubmitted(
        second.submission);
    assert(secondPayload.valid());
    vk::RuntimeFrameTransportSubmissionTestAccess::consumerRetired(second.submission);
}
}

int main() {
    testFrameTransportPresentAndWsiWrapper();
    testFrameTransportRejectsBeforeSubmit();
    testFrameTransportBackpressureAndReuse();
    auto makePayload = [](const vk::RuntimeExchangeEndpoint& ep) {
        auto exported = vk::createExportableSyncFdSemaphore(ep.semaphoreDevice);
        return vk::exportSyncFd(ep.semaphoreDevice, exported.handle());
    };
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        const auto consumer=h<VkSemaphore>(0xc1); auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(
            std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,consumer});
        assert(pending.valid() && pending.terminalReady()); auto view=pending.imageView(); assert(view.handoffPendingAcquire() && view.queueFamily()==VK_QUEUE_FAMILY_IGNORED); assert(view.sourceQueueFamily()==3 && view.destinationQueueFamily()==7);
        assert(t.submits==1 && t.submit.waitSemaphoreCount==1 && t.submit.signalSemaphoreCount==1 && t.waits.size()==1 && t.signals[0]==consumer); assert(t.submit.pWaitDstStageMask[0]==VK_PIPELINE_STAGE_TRANSFER_BIT);
        assert(t.events.size()==7 && t.events[0]=="begin" && t.events[1]=="barrier" && t.events[2]=="copy" && t.events[3]=="barrier" && t.events[4]=="barrier" && t.events[5]=="end" && t.events[6]=="submit");
        assert(t.barriers.back().srcQueueFamilyIndex==3 && t.barriers.back().dstQueueFamilyIndex==7 && t.barriers.back().srcAccessMask==VK_ACCESS_TRANSFER_READ_BIT && t.barriers.back().dstAccessMask==0 && t.srcStage==VK_PIPELINE_STAGE_TRANSFER_BIT && t.dstStage==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        assert(t.imagesDestroyed==0 && t.memoriesFreed==0 && t.buffersDestroyed==0 && t.poolsDestroyed==0 && t.fencesDestroyed==0);
        auto moved=std::move(pending); assert(!pending.valid() && moved.valid() && view.valid());
        const auto bytes=vk::RuntimeImageEndpoint::completeForeignImageReadback(moved); assert(bytes.size()==4 && t.submits==1 && t.waitsOnFence==1);
        assert(t.imagesDestroyed==0 && t.memoriesFreed==0 && view.valid());
        bool second=false; try { static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(moved)); } catch (...) { second=true; } assert(second);
        moved={}; assert(!view.valid());
        assert(t.imagesDestroyed==1 && t.memoriesFreed==2 && t.buffersDestroyed==1 && t.poolsDestroyed==1 && t.commandsFreed==1 && t.fencesDestroyed==1);
        for (const auto value : t.destroyedSemaphores) assert(value!=consumer);
    }
    {
        Trace t{}; auto ep=endpoint(5); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,5,t);
        auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{5,h<VkSemaphore>(0xc2)});
        assert(pending.valid() && pending.terminalReady() && pending.imageView().queueFamily()==5 && !pending.imageView().handoffPendingAcquire()); assert(t.barriers.size()==1 && t.submit.signalSemaphoreCount==1);
        static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(pending));
    }
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,h<VkSemaphore>(0xc7)});
        assert(pending.terminalReady() && !pending.completed());
        assert(!vk::RuntimeImageEndpoint::tryRetireForeignImageReadback(pending));
        assert(t.fenceStatusChecks==1 && t.waitsOnFence==0 && !pending.completed()
            && !pending.retirementObserved());
        t.fenceStatusResult=VK_SUCCESS;
        assert(vk::RuntimeImageEndpoint::tryRetireForeignImageReadback(pending));
        assert(t.fenceStatusChecks==2 && t.waitsOnFence==0 && !pending.completed()
            && pending.retirementObserved());
        assert(pending.terminalReady());
        assert(vk::RuntimeImageEndpoint::tryRetireForeignImageReadback(pending));
        assert(t.fenceStatusChecks==2 && t.waitsOnFence==0);
        const auto bytes=vk::RuntimeImageEndpoint::completeForeignImageReadback(pending);
        assert(bytes.size()==4 && pending.completed() && pending.retirementObserved());
        assert(t.fenceStatusChecks==2 && t.waitsOnFence==0);
    }
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep));
        assert(pending.valid() && !pending.terminalReady());
        static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(pending));
    }
    {
        Trace t{}; t.submitResult=VK_ERROR_DEVICE_LOST; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        bool failed=false; try { static_cast<void>(vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,h<VkSemaphore>(0xc3)})); } catch (...) { failed=true; }
        assert(failed && t.submits==1);
    }
    {
        Trace t{}; t.fenceResult=VK_ERROR_OUT_OF_HOST_MEMORY; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        bool failed=false; try { static_cast<void>(vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,h<VkSemaphore>(0xc4)})); } catch (...) { failed=true; }
        assert(failed && t.submits==0);
    }
    for (const bool invalidateFailure : {false,true}) {
        Trace t{}; if (invalidateFailure) t.invalidateResult=VK_ERROR_MEMORY_MAP_FAILED; else t.waitResult=VK_ERROR_DEVICE_LOST;
        auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,h<VkSemaphore>(0xc5)});
        bool failed=false; try { static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(pending)); } catch (...) { failed=true; }
        assert(failed); const auto waits=t.waitsOnFence; bool retried=false; try { static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(pending)); } catch (...) { retried=true; }
        assert(retried && t.waitsOnFence==waits && t.submits==1);
    }
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        const auto bytes=vk::RuntimeImageEndpoint::readForeignImage(image,makePayload(ep));
        assert(bytes.size()==4 && image.image()!=VK_NULL_HANDLE && t.submits==1 && t.waitsOnFence==1 && t.submit.signalSemaphoreCount==0 && t.barriers.size()==1);
    }
    vk::RuntimeForeignImageView stale;
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,h<VkSemaphore>(0xc6)});
        stale=pending.imageView(); assert(stale.valid()); static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(pending));
    }
    assert(!stale.valid());
    {
        Trace t{}; auto ep=endpoint(3); auto reused=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        assert(reused.image()==h<VkImage>(0x111) && !stale.valid());
    }
    return 0;
}
