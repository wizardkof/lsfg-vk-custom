#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
template<class T> T h(uintptr_t v) { return reinterpret_cast<T>(v); }
struct Trace {
    std::vector<std::string> events;
    std::vector<VkImageMemoryBarrier> barriers;
    VkPipelineStageFlags srcStage{}, dstStage{};
    VkSubmitInfo submit{};
    std::vector<VkSemaphore> waits, signals;
    std::vector<VkSemaphore> destroyedSemaphores;
    uint32_t submits{}, waitsOnFence{}, destroys{}, imagesDestroyed{}, memoriesFreed{},
        buffersDestroyed{}, poolsDestroyed{}, commandsFreed{}, fencesDestroyed{};
    VkResult submitResult{VK_SUCCESS};
    VkResult fenceResult{VK_SUCCESS};
    VkResult waitResult{VK_SUCCESS};
    VkResult invalidateResult{VK_SUCCESS};
    uint8_t* mapped{};
} *trace;

VkResult begin(VkCommandBuffer, const VkCommandBufferBeginInfo*) { trace->events.emplace_back("begin"); return VK_SUCCESS; }
VkResult end(VkCommandBuffer) { trace->events.emplace_back("end"); return VK_SUCCESS; }
void barrier(VkCommandBuffer, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags,
             uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*,
             uint32_t n, const VkImageMemoryBarrier* b) {
    trace->events.emplace_back("barrier"); trace->srcStage=s; trace->dstStage=d;
    for (uint32_t i=0;i<n;++i) trace->barriers.push_back(b[i]);
}
void copy(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy*) { trace->events.emplace_back("copy"); }
VkResult fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* f) { if (trace->fenceResult == VK_SUCCESS) *f=h<VkFence>(0xf1); return trace->fenceResult; }
void destroyFence(VkDevice, VkFence, const VkAllocationCallbacks*) { ++trace->fencesDestroyed; }
VkResult wait(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) { ++trace->waitsOnFence; trace->events.emplace_back("wait"); return trace->waitResult; }
VkResult submit(VkQueue, uint32_t n, const VkSubmitInfo* s, VkFence) {
    ++trace->submits; trace->submit=*s; trace->waits.assign(s->pWaitSemaphores,s->pWaitSemaphores+s->waitSemaphoreCount); trace->signals.assign(s->pSignalSemaphores,s->pSignalSemaphores+s->signalSemaphoreCount); trace->events.emplace_back("submit"); return trace->submitResult;
}
VkResult createSem(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* s) { *s=h<VkSemaphore>(0xe1); return VK_SUCCESS; }
void destroySem(VkDevice, VkSemaphore value, const VkAllocationCallbacks*) { ++trace->destroys; trace->destroyedSemaphores.push_back(value); }
void destroyImage(VkDevice, VkImage, const VkAllocationCallbacks*) { ++trace->imagesDestroyed; }
void destroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) { ++trace->buffersDestroyed; }
void freeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++trace->memoriesFreed; }
void destroyPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) { ++trace->poolsDestroyed; }
void freeCommands(VkDevice, VkCommandPool, uint32_t n, const VkCommandBuffer*) { trace->commandsFreed += n; }
VkResult importFd(VkDevice, const VkImportSemaphoreFdInfoKHR* i) { if(i->fd>=0) close(i->fd); return VK_SUCCESS; }
VkResult getFd(VkDevice, const VkSemaphoreGetFdInfoKHR*, int* fd) { int p[2]; assert(pipe(p)==0); close(p[1]); *fd=p[0]; return VK_SUCCESS; }
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
        .WaitForFences=wait,
        .DestroyCommandPool=destroyPool,
        .FreeCommandBuffers=freeCommands,
        .BeginCommandBuffer=begin,
        .EndCommandBuffer=end,
        .CmdPipelineBarrier=barrier,
        .UnmapMemory=unmap,
        .InvalidateMappedMemoryRanges=invalidate,
        .DestroyImage=destroyImage,
        .CmdCopyImageToBuffer=copy};
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
}

int main() {
    auto makePayload = [](const vk::RuntimeExchangeEndpoint& ep) {
        auto exported = vk::createExportableSyncFdSemaphore(ep.semaphoreDevice);
        return vk::exportSyncFd(ep.semaphoreDevice, exported.handle());
    };
    {
        Trace t{}; auto ep=endpoint(3); auto image=vk::RuntimeImageEndpointTestAccess::make(ep,3,t);
        const auto consumer=h<VkSemaphore>(0xc1); auto pending=vk::RuntimeImageEndpoint::submitForeignImageReadback(
            std::move(image),makePayload(ep),vk::RuntimeForeignImageHandoffInfo{7,consumer});
        assert(pending.valid()); auto view=pending.imageView(); assert(view.handoffPendingAcquire() && view.queueFamily()==VK_QUEUE_FAMILY_IGNORED); assert(view.sourceQueueFamily()==3 && view.destinationQueueFamily()==7);
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
        assert(pending.valid() && pending.imageView().queueFamily()==5 && !pending.imageView().handoffPendingAcquire()); assert(t.barriers.size()==1 && t.submit.signalSemaphoreCount==1);
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
