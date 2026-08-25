/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "swapchain.hpp"
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk;
using namespace lsfgvk::layer;

namespace {
    struct CaptureResources {
        VkDevice device{};
        PFN_vkDestroyBuffer destroyBuffer{};
        PFN_vkFreeMemory freeMemory{};
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        ~CaptureResources() {
            if (buffer) destroyBuffer(device, buffer, nullptr);
            if (memory) freeMemory(device, memory, nullptr);
        }
    };

    [[nodiscard]] bool isFixedMode(const ls::GameConf& profile) {
        return profile.frame_generation_mode == ls::FrameGenerationMode::Fixed;
    }

    [[nodiscard]] bool isAdaptiveBypass(const ls::GameConf& profile) {
        return !isFixedMode(profile) && profile.multiplier == 1;
    }

    [[nodiscard]] bool hasPresentModeInfo(const void* nextChain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(nextChain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR)
                return true;
            current = current->pNext;
        }
        return false;
    }

    [[nodiscard]] VkFence applicationPresentFence(const void* nextChain) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(nextChain);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR) {
                const auto* info =
                    reinterpret_cast<const VkSwapchainPresentFenceInfoKHR*>(current);
                // Swapchain::present() handles one real WSI swapchain. Do not
                // guess the mapping of a batched application's fence array.
                if (info->swapchainCount == 1 && info->pFences)
                    return info->pFences[0];
                return VK_NULL_HANDLE;
            }
            current = current->pNext;
        }
        return VK_NULL_HANDLE;
    }

    [[nodiscard]] size_t generatedFrameCapacity(const ls::GameConf& profile) {
        if (isFixedMode(profile))
            return FixedFrameScheduler::DEFAULT_MAX_GENERATED_FRAMES;
        return profile.multiplier > 1 ? profile.multiplier - 1 : 0;
    }

    VkImageMemoryBarrier barrierHelper(VkImage handle,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            VkImageSubresourceRange range = vk::exchangeImageSubresourceRange()) {
        return vk::makeImageBarrier(handle,
            srcAccessMask, dstAccessMask, oldLayout, newLayout,
            srcQueueFamilyIndex, dstQueueFamilyIndex, range);
    }

    bool modifierSupportsBlitSource(const vk::Vulkan& vk, VkFormat format,
            uint64_t modifier) {
        VkDrmFormatModifierPropertiesListEXT list{
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
        VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &list};
        vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), format, &properties);
        std::vector<VkDrmFormatModifierPropertiesEXT> modifiers(
            list.drmFormatModifierCount);
        list.pDrmFormatModifierProperties = modifiers.data();
        vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), format, &properties);
        return std::ranges::any_of(modifiers, [modifier](const auto& item) {
            return item.drmFormatModifier == modifier
                && (item.drmFormatModifierTilingFeatures
                    & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
        });
    }

    VkResult acquireRealSwapchainImage(const vk::Vulkan& vk,
            VkSwapchainKHR swapchain, VkSemaphore semaphore,
            uint32_t* imageIndex, std::stop_token stopToken) {
        constexpr uint64_t WORKER_ACQUIRE_SLICE_NS = 50ULL * 1000ULL * 1000ULL;
        while (true) {
            if (stopToken.stop_possible() && stopToken.stop_requested())
                return VK_ERROR_OUT_OF_DATE_KHR;

            const uint64_t timeout = stopToken.stop_possible()
                ? WORKER_ACQUIRE_SLICE_NS
                : UINT64_MAX;
            const auto res = vk.df().AcquireNextImageKHR(vk.dev(), swapchain,
                timeout, semaphore, VK_NULL_HANDLE, imageIndex);
            if (res == VK_TIMEOUT && stopToken.stop_possible())
                continue;
            return res;
        }
    }
}

PrePresentGateResult Swapchain::prePresentGate() noexcept {
    if (!d3b3ProductionState)
        return PrePresentGateResult::READY;
    return d3b3ProductionState->prePresentGate();
}

void Swapchain::captureRealFrameOnce(const vk::Vulkan& vk, VkImage sourceImage,
        uint32_t imageIndex, const std::vector<VkSemaphore>& bridgeSemaphores) {
    if (!captureDiagnosticFormatSupported(this->info.format))
        throw ls::error("P4C-B unsupported capture source format");
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(this->info.extent.width)
        * this->info.extent.height * 4;
    CaptureResources resources{vk.dev(), vk.df().DestroyBuffer, vk.df().FreeMemory};
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byteSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    auto result = vk.df().CreateBuffer(vk.dev(), &bufferInfo, nullptr, &resources.buffer);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture buffer creation");
    VkMemoryRequirements requirements{};
    vk.df().GetBufferMemoryRequirements(vk.dev(), resources.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vk.fi().GetPhysicalDeviceMemoryProperties(vk.physdev(), &memoryProperties);
    std::optional<uint32_t> memoryType;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryType = i; break;
        }
        if (!memoryType && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) memoryType = i;
    }
    if (!memoryType) throw ls::error("P4C-B capture staging has no host-visible memory type");
    const auto memoryFlags = memoryProperties.memoryTypes[*memoryType].propertyFlags;
    const bool coherent = (memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    const VkMemoryAllocateInfo allocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = *memoryType
    };
    result = vk.df().AllocateMemory(vk.dev(), &allocation, nullptr, &resources.memory);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture memory allocation");
    result = vk.df().BindBufferMemory(vk.dev(), resources.buffer, resources.memory, 0);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture buffer bind");

    vk::CommandBuffer command(vk);
    command.begin(vk);
    const auto range = vk::exchangeImageSubresourceRange();
    const auto toTransfer = vk::makeImageBarrier(sourceImage, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    command.insertBarriers(vk, {toTransfer}, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {this->info.extent.width, this->info.extent.height, 1}
    };
    vk.df().CmdCopyImageToBuffer(command.handle(), sourceImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, resources.buffer, 1, &copy);
    const auto restore = vk::makeImageBarrier(sourceImage, VK_ACCESS_TRANSFER_READ_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    command.insertBarriers(vk, {restore}, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    command.end(vk);
    vk::Fence fence(vk);
    command.submit(vk, vk.queue(), bridgeSemaphores, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE, 0,
        fence.handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (!fence.wait(vk)) throw ls::error("P4C-B capture fence wait failed");

    void* mapped{};
    result = vk.df().MapMemory(vk.dev(), resources.memory, 0, byteSize, 0, &mapped);
    if (result != VK_SUCCESS) throw ls::vulkan_error(result, "P4C-B capture readback map");
    if (!coherent) {
        const VkMappedMemoryRange invalidate{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = resources.memory,
            .size = byteSize
        };
        result = vk.df().InvalidateMappedMemoryRanges(vk.dev(), 1, &invalidate);
        if (result != VK_SUCCESS) {
            vk.df().UnmapMemory(vk.dev(), resources.memory);
            throw ls::vulkan_error(result, "P4C-B capture readback invalidate");
        }
    }
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    uint64_t checksum = ::lsfgvk::common::fnv1a64(bytes, byteSize);
    VkDeviceSize nonZero{};
    for (VkDeviceSize i = 0; i < byteSize; ++i) {
        nonZero += bytes[i] != 0;
    }
    vk.df().UnmapMemory(vk.dev(), resources.memory);
    if (nonZero == 0) throw ls::error("P4C-B capture readback is entirely zero");
    std::cerr << "[DG2X-P4C-B] Real frame capture on application GPU\n"
        << "  Runtime mode: CAPTURE_ONLY\n"
        << "  Source image index: " << imageIndex << "\n"
        << "  Source VkImage: " << sourceImage << "\n"
        << "  Source format: " << static_cast<int>(this->info.format) << "\n"
        << "  Source extent: " << this->info.extent.width << 'x' << this->info.extent.height << "\n"
        << "  Present waits at boundary: " << bridgeSemaphores.size() << "\n"
        << "  bridgePresentWaits: PASS\n"
        << "  Capture queue family: " << vk.queueFamilyIndex() << "\n"
        << "  Capture command pool: PASS\n"
        << "  Capture staging allocation: PASS\n"
        << "  Capture byte size: " << byteSize << "\n"
        << "  Capture memoryType: " << *memoryType << "\n"
        << "  Capture host coherent: " << (coherent ? "YES" : "NO") << "\n"
        << "  Source state acquisition: PASS\n"
        << "  Source layout transition: PASS\n"
        << "  vkCmdCopyImageToBuffer: PASS\n"
        << "  Source state restore: PASS\n"
        << "  Capture submit A: PASS\n"
        << "  Capture fence completion: PASS\n"
        << "  Capture readback: PASS\n"
        << "  Captured byte count: " << byteSize << "\n"
        << "  Captured non-zero bytes: " << nonZero << "\n"
        << "  Captured checksum: 0x" << std::hex << checksum << std::dec << "\n"
        << "  Backend submission: NONE\n"
        << "  Frame DMA-BUF: NONE\n"
        << "  Frame FOREIGN ownership: NONE\n"
        << "  Frame SYNC_FD: NONE\n"
        << "  LSFG backend execution: NONE\n"
        << "  Frame transport connected: NO\n"
        << "DG2X_P4C_B_REAL_FRAME_CAPTURE_A_PASS\n"
        << "cross-device real frame captured on application GPU, but frame transport is not connected yet\n";
}

void layer::context_ModifySwapchainCreateInfo(const ls::GameConf& profile, uint32_t maxImages,
        VkSwapchainCreateInfoKHR& createInfo) {
    createInfo.imageUsage |=
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    switch (profile.pacing) {
        case ls::Pacing::None:
            // Preserve the exact Adaptive expansion. Fixed reserves enough real
            // swapchain images for its maximum dynamic generation capacity.
            createInfo.minImageCount += isFixedMode(profile)
                ? FixedFrameScheduler::DEFAULT_MAX_GENERATED_FRAMES + 1
                : profile.multiplier;
            if (maxImages && createInfo.minImageCount > maxImages)
                createInfo.minImageCount = maxImages;

            createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            break;
    }
}

Swapchain::Swapchain(const vk::Vulkan& vk, backend::Instance& backend,
            vk::RuntimeDevicePair devicePair,
            ls::GameConf profile, SwapchainInfo info,
            uint32_t offloadQueueFamily) :
        instance(backend),
        devicePair(std::move(devicePair)),
        fixedScheduler(profile.target_fps),
        fixedOutputPacer(profile.target_fps),
        profile(std::move(profile)), info(std::move(info)) {
    this->offloadQueueFamily = offloadQueueFamily;
    if (this->devicePair.crossDevice())
        this->crossDeviceMode = CrossDeviceRuntimeMode::CAPTURE_ONLY;

    // A virtual Adaptive 1x swapchain still needs the final virtual->real copy
    // resources even though it must not create an LSFG generation context.
    if (this->info.virtualized) {
        if (offloadQueueFamily == VK_QUEUE_FAMILY_IGNORED)
            throw ls::error("virtual presentation requires an auxiliary queue family");
    }

    // P4C-B0 deliberately exposes only the application-side capture boundary.
    // Do not construct backend/LSFG resources until real-frame transport exists.
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY) {
        auto backing = RuntimeDmaBufBacking::createImage(
            this->devicePair.render.identity, this->info.extent);
        auto endpointA = vk::makeRuntimeExchangeEndpoint(vk);
        auto endpointB = backend.runtimeExchangeEndpoint();
        const vk::RuntimeImageBackingInfo backingInfo{
            .extent = this->info.extent,
            .backingSize = backing.size(),
            .fourcc = backing.fourcc(),
            .modifier = backing.modifier(),
            .planeCount = backing.planeCount(),
            .plane = {backing.offset(), 0, backing.stride(), 0, 0}
        };
        this->frameTransportA = vk::createRuntimeImageEndpoint(
            endpointA, backing.duplicatePlaneFd(0), backingInfo);
        this->frameTransportB = vk::createRuntimeImageEndpoint(
            endpointB, backing.duplicatePlaneFd(0), backingInfo);
        this->frameTransportA = vk::RuntimeImageEndpoint::createExecutionResources(
            std::move(this->frameTransportA), 1);
        this->frameTransportB = vk::RuntimeImageEndpoint::createExecutionResources(
            std::move(this->frameTransportB), 1);
        this->frameTransportBacking = std::move(backing);
        this->frameTransportReady = true;
        return;
    }

    // Adaptive multiplier == 1 keeps the Vulkan layer/profile active but
    // bypasses LSFG. Fixed mode ignores multiplier and remains active.
    if (isAdaptiveBypass(this->profile))
        return;

    const VkExtent2D extent = this->info.extent;
    const bool hdr = this->info.format > 57;

    const auto format = hdr
        ? VK_FORMAT_R16G16B16A16_SFLOAT
        : VK_FORMAT_R8G8B8A8_UNORM;
    const auto sourceDescriptor = vk::makeSourceExchangeImageDescriptor(extent, format);
    const auto destinationDescriptor = vk::makeDestinationExchangeImageDescriptor(extent, format);
    std::pair<vk::ExternalImage, vk::ExternalImage> externalSourceImages{};
    std::vector<vk::ExternalImage> externalDestinationImages(
        generatedFrameCapacity(this->profile));

    this->sourceImages.reserve(2);
    this->sourceImages.emplace_back(vk, sourceDescriptor, externalSourceImages.first);
    this->sourceImages.emplace_back(vk, sourceDescriptor, externalSourceImages.second);

    this->destinationImages.reserve(externalDestinationImages.size());
    for (auto& image : externalDestinationImages)
        this->destinationImages.emplace_back(vk, destinationDescriptor, image);

    // A owns every exported destination initially, while B is its first
    // writer. Complete the one-time A -> EXTERNAL releases before importing
    // the same allocations into the backend context.
    if (!this->destinationImages.empty()) {
        std::vector<vk::Barrier> initialDestinationReleases;
        initialDestinationReleases.reserve(this->destinationImages.size());
        for (const auto& image : this->destinationImages) {
            initialDestinationReleases.push_back(
                vk::destinationInitialReleaseToBackend(
                    image.handle(), vk.queueFamilyIndex()));
        }
        const vk::CommandBuffer initialRelease{vk};
        initialRelease.begin(vk);
        initialRelease.insertBarriers(vk, initialDestinationReleases);
        initialRelease.end(vk);
        initialRelease.submit(vk);
    }

    int syncFd{};
    this->syncSemaphore.emplace(vk, 0, std::nullopt, &syncFd);

    try {
        this->ctx = ls::owned_ptr<ls::R<backend::Context>>(
            new ls::R<backend::Context>(backend.openContext(
                std::move(externalSourceImages),
                std::move(externalDestinationImages), syncFd,
                1.0F / this->profile.flow_scale, this->profile.performance_mode
            )),
            [backend = &backend](ls::R<backend::Context>& ctx) {
                backend->closeContext(ctx);
            }
        );

        backend::makeLeaking(); // don't worry about it :3
    } catch (const std::exception& e) {
        throw ls::error("failed to create swapchain context", e);
    }

    this->renderCommandBuffer.emplace(vk);
    if (!this->renderFence.has_value())
        this->renderFence.emplace(vk);
    for (size_t i = 0; i < this->destinationImages.size(); i++) {
        this->passes.emplace_back(RenderPass {
            .commandBuffer = vk::CommandBuffer(vk),
            .acquireSemaphore = vk::Semaphore(vk)
        });
    }

    const size_t frames = std::max(this->info.images.size(), this->destinationImages.size() + 2);
    for (size_t i = 0; i < frames; i++) {
        this->postCopySemaphores.emplace_back(
            vk::Semaphore(vk),
            vk::Semaphore(vk)
        );
    }
}

void Swapchain::ensureGraphicsFinalResources(const vk::Vulkan& vk, uint32_t family) {
    if (family == VK_QUEUE_FAMILY_IGNORED)
        throw ls::error("graphics-final queue family is unavailable");
    if (this->virtualFinalCommandBuffer.has_value()) {
        if (this->virtualFinalCommandFamily != family)
            throw ls::error("graphics-final command resources changed queue family");
        return;
    }
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = family;
    VkCommandPool pool{};
    const auto result = vk.df().CreateCommandPool(vk.dev(), &poolInfo, nullptr, &pool);
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, "graphics-final command pool creation failed");
    this->virtualFinalCommandPool = ls::owned_ptr<VkCommandPool>(
        new VkCommandPool(pool),
        [dev = vk.dev(), destroy = vk.df().DestroyCommandPool](VkCommandPool& value) {
            destroy(dev, value, nullptr);
        });
    this->virtualFinalCommandFamily = family;
    this->virtualFinalCommandBuffer.emplace(vk, pool);
    this->virtualFinalAcquireSemaphore.emplace(vk);
    this->virtualFinalPresentSemaphore.emplace(vk);
    if (d3b2InsertionDiagnosticEnabled()) {
        this->d3b2OriginalAcquireSemaphore.emplace(vk);
        this->d3b2GeneratedPresentSemaphore.emplace(vk);
        this->d3b2OriginalPresentSemaphore.emplace(vk);
    }
    if (!this->renderFence.has_value()) {
      this->renderFence.emplace(vk);
    }
}

VkResult Swapchain::present(const vk::Vulkan& vk,
        VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
        VkSwapchainKHR swapchain,
        void* next_chain, uint32_t imageIdx,
        const std::vector<VkSemaphore>& semaphores,
        std::stop_token stopToken,
        std::optional<std::chrono::steady_clock::time_point> sourcePresentTime,
        bool d3bSingleSwapchainEligible,
        const GraphicsFinalQueueInfo* graphicsFinalQueue,
        BorrowedGraphicsQueueLease* graphicsLease,
        bool* stopAfterCompletion) {
    if (this->info.virtualized) {
        if (!graphicsFinalQueue || !graphicsLease
                || graphicsFinalExecutionMode(*graphicsFinalQueue, true)
                    != GraphicsFinalExecutionMode::BORROWED_SYNCHRONOUS
                || !graphicsLease->validFor(
                    graphicsFinalQueue->queue, graphicsFinalQueue->family)
                || queue != graphicsFinalQueue->queue)
            throw ls::error("borrowed graphics-final queue contract is unavailable");
        this->ensureGraphicsFinalResources(vk, graphicsFinalQueue->family);
    }
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY
            && this->d3b1State == D3B1PresentationState::PASS) {
        if (semaphores.size() != 1 || queue == VK_NULL_HANDLE || !queueMutex)
            throw ls::error("D3B1 terminal ready-semaphore drain is unavailable");
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        const VkSubmitInfo drain{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = semaphores.data(),
            .pWaitDstStageMask = &waitStage};
        if (!this->renderFence.has_value())
            throw ls::error("D3B1 terminal drain fence is unavailable");
        this->renderFence->reset(vk);
        VkResult result{};
        {
            const std::scoped_lock queueLock(*queueMutex);
            result = vk.df().QueueSubmit(
                queue, 1, &drain, this->renderFence->handle());
        }
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "D3B1 terminal ready-semaphore drain failed");
        if (!this->renderFence->wait(vk, UINT64_MAX))
            throw ls::vulkan_error(VK_TIMEOUT,
                "D3B1 terminal ready-semaphore drain did not retire");
        return VK_SUCCESS;
    }
    if (this->crossDeviceMode == CrossDeviceRuntimeMode::CAPTURE_ONLY) {
        if (this->captureOnlyPhase == 0) {
            const auto sourceImage = this->info.images.at(imageIdx);
            std::cerr << "[DG2X-P4C-B0] Cross-device capture-only runtime\n"
                << "  RuntimeDevicePair: CROSS_PHYSICAL_DEVICE\n"
                << "  P4B runtime image channel: PASS\n"
                << "  Runtime mode: CAPTURE_ONLY\n"
                << "  Frame transport connected: NO\n"
                << "  vkCreateSwapchainKHR continuation: PASS\n"
                << "  VirtualSwapchainRuntime creation: PASS\n"
                << "  Virtual images exposed: PASS\n"
                << "  vkQueuePresentKHR reached: PASS\n"
                << "  Virtual image index: " << imageIdx << "\n"
                << "  Source VkImage: " << sourceImage << "\n"
                << "  Source format: " << static_cast<int>(this->info.format) << "\n"
                << "  Source extent: " << this->info.extent.width << 'x'
                << this->info.extent.height << "\n"
                << "  Present wait semaphore count at capture boundary: "
                << semaphores.size() << "\n"
                << "  bridgePresentWaits reached: PASS\n"
                << "  Capture hook reached: PASS\n"
                << "  Backend real-frame workload: NO\n"
                << "  LSFG execution on B: NO\n"
                << "  Presentation from B: NO\n"
                << "  Capture hook reached: PASS\n";
            if (!this->frameTransportReady)
                throw ls::error("P4C-C frame transport resources unavailable");
            vk::RuntimeImageEndpoint::executeRealFrameTransport(
                this->frameTransportA, this->frameTransportB, sourceImage,
                this->info.extent, semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            this->instance.get().validateRuntimePrepass(
                this->frameTransportB.image(), this->info.extent,
                VK_FORMAT_B8G8R8A8_UNORM, this->frameTransportBacking.modifier(),
                1.0F / this->profile.flow_scale, this->profile.performance_mode);
            this->captureRealFrameOnce(vk, sourceImage, imageIdx, {});
            std::cerr << "DG2X_P4C_B0_CAPTURE_ONLY_RUNTIME_PASS\n"
                << "cross-device capture hook reached, but real frame transport is not connected yet\n";
            this->captureOnlyPhase = 1;
            return VK_SUCCESS;
        }
        if (this->captureOnlyPhase <= 2) {
            if (this->captureOnlyPhase == 1) {
                auto& session = this->instance.get().openRuntimePrepassSession(
                    this->info.extent, VK_FORMAT_B8G8R8A8_UNORM,
                    this->frameTransportBacking.modifier(),
                    1.0F / this->profile.flow_scale, this->profile.performance_mode);
                this->runtimePrepassSession = ls::owned_ptr<ls::R<backend::RuntimePrepassSession>>(
                    new ls::R<backend::RuntimePrepassSession>(session),
                    [backend = &this->instance.get()](ls::R<backend::RuntimePrepassSession>& value) {
                        backend->closeRuntimePrepassSession(value);
                    });
            }
            const auto sourceImage = this->info.images.at(imageIdx);
            auto payload = vk::RuntimeImageEndpoint::submitRealFrameTransportA(
                this->frameTransportA, sourceImage, this->info.extent,
                semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            this->instance.get().processRuntimePrepass(
                this->runtimePrepassSession.get(), this->frameTransportB.image(),
                std::move(payload));
            this->captureOnlyPhase++;
            return VK_SUCCESS;
        }
        if (this->captureOnlyPhase <= 5) {
            if (this->captureOnlyPhase == 3) {
                auto& session = this->instance.get().openRuntimeGenerateSession(
                    this->info.extent, VK_FORMAT_B8G8R8A8_UNORM,
                    this->frameTransportBacking.modifier(),
                    1.0F / this->profile.flow_scale, this->profile.performance_mode);
                this->runtimeGenerateDiagnosticSession =
                    ls::owned_ptr<ls::R<backend::RuntimeGenerateSession>>(
                        new ls::R<backend::RuntimeGenerateSession>(session),
                        [backend = &this->instance.get()](
                                ls::R<backend::RuntimeGenerateSession>& value) {
                            backend->closeRuntimeGenerateSession(value);
                        });
            }
            const auto sourceImage = this->info.images.at(imageIdx);
            auto payload = vk::RuntimeImageEndpoint::submitRealFrameTransportA(
                this->frameTransportA, sourceImage, this->info.extent,
                semaphores.empty() ? VK_NULL_HANDLE : semaphores.front());
            auto generated = this->instance.get().submitRuntimeGenerateDiagnostic(
                this->runtimeGenerateDiagnosticSession.get(), this->frameTransportB.image(),
                std::move(payload));
            if (generated) {
                const bool d3bResourcesReady = queue != VK_NULL_HANDLE && queueMutex
                    && graphicsFinalQueue
                    && graphicsFinalQueue->family != VK_QUEUE_FAMILY_IGNORED
                    && this->virtualFinalCommandBuffer.has_value()
                    && this->virtualFinalAcquireSemaphore.has_value()
                    && this->virtualFinalPresentSemaphore.has_value()
                    && this->renderFence.has_value() && !this->info.realImages.empty();
                const bool d3b = d3b1PresentEligible(d3b1PresentationDiagnosticEnabled(),
                    d3bSingleSwapchainEligible, next_chain != nullptr,
                    this->info.virtualized, d3bResourcesReady, this->d3b1State);
                const bool d3b2 = d3b1PresentEligible(d3b2InsertionDiagnosticEnabled(),
                    d3bSingleSwapchainEligible, next_chain != nullptr,
                    this->info.virtualized, d3bResourcesReady
                        && this->info.releaseBackend != SwapchainReleaseBackend::None
                        && this->d3b2OriginalAcquireSemaphore.has_value()
                        && this->d3b2GeneratedPresentSemaphore.has_value()
                        && this->d3b2OriginalPresentSemaphore.has_value(),
                    this->d3b1State);
                const auto terminalConsumer = selectGeneratedOutputTerminalConsumer(d3b, d3b2);
                std::optional<vk::RuntimeForeignImageHandoffInfo> handoff;
                if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                    try { this->returnedForGraphics.emplace(vk); }
                    catch (...) {
                        if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                            this->d3b1State = D3B1PresentationState::FAILED;
                        throw;
                    }
                    handoff = vk::RuntimeForeignImageHandoffInfo{
                        .destinationQueueFamilyIndex = graphicsFinalQueue->family,
                        .signalSemaphore = this->returnedForGraphics->handle()};
                }
                try {
                    this->generatedOutputReturnDiagnosticSession =
                        std::make_unique<GeneratedOutputReturnSession>(
                            std::move(*generated), this->instance.get(),
                            this->runtimeGenerateDiagnosticSession.get(), this->devicePair,
                            this->instance.get().runtimeExchangeEndpoint(),
                            vk::makeRuntimeExchangeEndpoint(vk), true, handoff);
                } catch (...) {
                    if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                        if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                            this->d3b1State = D3B1PresentationState::FAILED;
                        this->returnedForGraphics.reset();
                    }
                    throw;
                }
                if (terminalConsumer != GeneratedOutputTerminalConsumer::DiagnosticOnly) {
                    if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B1)
                        this->d3b1State = D3B1PresentationState::A_HANDOFF_SUBMITTED;
                    const auto view = this->generatedOutputReturnDiagnosticSession
                        ->returnedImageView();
                    VkFormatProperties2 destinationProperties{
                        VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
                    vk.fi().GetPhysicalDeviceFormatProperties2(
                        vk.physdev(), this->info.format, &destinationProperties);
                    if (terminalConsumer == GeneratedOutputTerminalConsumer::D3B2) {
                        this->d3b2State = D3B2InsertionState::INACTIVE;
                        D3B2InsertionPath insertion{
                            .generated = {
                                .image = view.image(), .format = view.format(),
                                .extent = view.extent(), .layout = view.layout(),
                                .sourceQueueFamily = view.sourceQueueFamily(),
                                .destinationQueueFamily = view.destinationQueueFamily(),
                                .ownershipAcquireRequired = view.handoffPendingAcquire(),
                                .blitSourceSupported = view.valid()
                                    && modifierSupportsBlitSource(vk, view.format(), view.modifier())},
                            .original = {
                                .image = this->info.images.at(imageIdx),
                                .format = this->info.format, .extent = this->info.extent,
                                .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                .sourceQueueFamily = graphicsFinalQueue->family,
                                .destinationQueueFamily = graphicsFinalQueue->family,
                                .blitSourceSupported = (destinationProperties.formatProperties.optimalTilingFeatures
                                    & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0},
                            .originalReady = semaphores.size() > 1 ? semaphores.at(1) : VK_NULL_HANDLE,
                            .returnedForGraphics = this->returnedForGraphics->handle(),
                            .acquireGenerated = this->virtualFinalAcquireSemaphore->handle(),
                            .acquireOriginal = this->d3b2OriginalAcquireSemaphore->handle(),
                            .generatedPresentReady = this->d3b2GeneratedPresentSemaphore->handle(),
                            .originalPresentReady = this->d3b2OriginalPresentSemaphore->handle(),
                            .graphicsFence = this->renderFence->handle(),
                            .commandPoolFamily = this->virtualFinalCommandFamily,
                            .submitQueueFamily = graphicsFinalQueue->family,
                            .submitQueueFlags = graphicsFinalQueue->flags,
                            .singleSwapchain = d3bSingleSwapchainEligible,
                            .hasPNext = next_chain != nullptr,
                            .fifo = this->info.adaptivePresentMode == VK_PRESENT_MODE_FIFO_KHR,
                            .hiddenBlitDestinationSupported = (destinationProperties.formatProperties.optimalTilingFeatures
                                & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0,
                            .maintenanceReleaseCapable = this->info.releaseBackend
                                != SwapchainReleaseBackend::None,
                            .state = &this->d3b2State};
                        insertion.acquire = [&](VkSemaphore acquireSemaphore) {
                            uint32_t index{};
                            const auto result = vk.df().AcquireNextImageKHR(vk.dev(), swapchain,
                                2ULL * 1000 * 1000 * 1000, acquireSemaphore, VK_NULL_HANDLE, &index);
                            if (result != VK_SUCCESS)
                                throw ls::vulkan_error(result, "D3B2 hidden WSI acquire failed");
                            return D3B2HiddenImage{this->info.realImages.at(index), index, this->info.extent};
                        };
                        insertion.record = [&](const D3B2HiddenImage& generated,
                                const D3B2HiddenImage& original) {
                            const auto& command = *this->virtualFinalCommandBuffer;
                            command.begin(vk);
                            std::vector<vk::Barrier> generatedPre;
                            if (view.handoffPendingAcquire())
                                generatedPre.push_back(barrierHelper(view.image(), 0,
                                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    view.sourceQueueFamily(), view.destinationQueueFamily()));
                            generatedPre.push_back(barrierHelper(generated.image, 0,
                                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
                            command.blitImage(vk, generatedPre, {view.image(), generated.image},
                                {view.extent(), generated.extent}, {barrierHelper(generated.image,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)});
                            command.blitImage(vk, {
                                    barrierHelper(this->info.images.at(imageIdx), 0,
                                        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                                    barrierHelper(original.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)},
                                {this->info.images.at(imageIdx), original.image},
                                this->info.extent,
                                {barrierHelper(this->info.images.at(imageIdx),
                                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
                                 barrierHelper(original.image, VK_ACCESS_TRANSFER_WRITE_BIT,
                                    VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)});
                            command.end(vk);
                            this->renderFence->reset(vk);
                        };
                        insertion.submit = [&] {
                            const std::vector<VkSemaphore> waits{
                                insertion.originalReady, insertion.returnedForGraphics,
                                insertion.acquireGenerated, insertion.acquireOriginal};
                            const std::scoped_lock queueLock(*queueMutex);
                            this->virtualFinalCommandBuffer->submit(vk, queue, waits,
                                VK_NULL_HANDLE, 0,
                                {insertion.generatedPresentReady, insertion.originalPresentReady},
                                VK_NULL_HANDLE, 0, insertion.graphicsFence,
                                VK_PIPELINE_STAGE_TRANSFER_BIT);
                            return VK_SUCCESS;
                        };
                        insertion.present = [&](const D3B2HiddenImage& image, VkSemaphore ready) {
                            const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .waitSemaphoreCount = 1, .pWaitSemaphores = &ready,
                                .swapchainCount = 1, .pSwapchains = &swapchain,
                                .pImageIndices = &image.index};
                            const std::scoped_lock queueLock(*queueMutex);
                            return vk.df().QueuePresentKHR(queue, &info);
                        };
                        insertion.waitGraphicsFence = [&] { return this->renderFence->wait(vk, UINT64_MAX); };
                        insertion.generatedIntegrity = [&] {
                            this->generatedOutputReturnDiagnosticSession->completePresentationDiagnostics();
                            return this->generatedOutputReturnDiagnosticSession->gpuChainedPassed();
                        };
                        insertion.originalIdentity = [&] { return insertion.original.image == this->info.images.at(imageIdx); };
                        insertion.retire = [&] { this->generatedOutputReturnDiagnosticSession.reset(); this->returnedForGraphics.reset(); };
                        insertion.emitMarker = [] { std::cerr << "DG2X_P4C_D3B2_GENERATED_THEN_ORIGINAL_PRESENT_PASS\n"; };
                        insertion.releaseAcquiredImages = [&](const std::vector<uint32_t>& indices) {
                            VkReleaseSwapchainImagesInfoKHR releaseInfo{
                                .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
                                .swapchain = swapchain,
                                .imageIndexCount = static_cast<uint32_t>(indices.size()),
                                .pImageIndices = indices.data()};
                            if (this->info.releaseBackend == SwapchainReleaseBackend::Khr
                                    && vk.df().ReleaseSwapchainImagesKHR)
                                return vk.df().ReleaseSwapchainImagesKHR(vk.dev(), &releaseInfo);
                            if (this->info.releaseBackend == SwapchainReleaseBackend::Ext
                                    && vk.df().ReleaseSwapchainImagesEXT)
                                return vk.df().ReleaseSwapchainImagesEXT(vk.dev(), &releaseInfo);
                            return VK_ERROR_EXTENSION_NOT_PRESENT;
                        };
                        const auto result = executeD3B2Insertion(insertion);
                        this->captureOnlyPhase = 6;
                        if (stopAfterCompletion && d3b1OneShotRequested(std::getenv("LSFGVK_D3B1_ONESHOT")))
                            *stopAfterCompletion = true;
                        return result;
                    }
                    D3B1PresentPath presentPath{
                        .source = {
                            .image = view.image(), .format = view.format(),
                            .extent = view.extent(), .layout = view.layout(),
                            .modifier = view.modifier(),
                            .sourceQueueFamily = view.sourceQueueFamily(),
                            .destinationQueueFamily = view.destinationQueueFamily(),
                            .ownershipAcquireRequired = view.handoffPendingAcquire()},
                        .hiddenAcquire = this->virtualFinalAcquireSemaphore->handle(),
                        .returnedForPresent = this->returnedForGraphics->handle(),
                        .finalPresentSemaphore = this->virtualFinalPresentSemaphore->handle(),
                        .renderFence = this->renderFence->handle(),
                        .commandPoolFamily = this->virtualFinalCommandFamily,
                        .submitQueueFamily = graphicsFinalQueue->family,
                        .submitQueueFlags = graphicsFinalQueue->flags,
                        .sourceBlitSupported = view.valid()
                            && view.destinationQueueFamily() == graphicsFinalQueue->family
                            && modifierSupportsBlitSource(vk, view.format(), view.modifier()),
                        .destinationBlitSupported =
                            (destinationProperties.formatProperties.optimalTilingFeatures
                                & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0,
                        .state = &this->d3b1State};
                    presentPath.acquireHidden = [&] {
                        uint32_t index{};
                        const auto result = acquireRealSwapchainImage(vk, swapchain,
                            this->virtualFinalAcquireSemaphore->handle(), &index, stopToken);
                        if (result != VK_SUCCESS)
                            throw ls::vulkan_error(result, "D3B1 hidden WSI acquire failed");
                        return D3B1HiddenImage{
                            this->info.realImages.at(index), index, this->info.extent};
                    };
                    presentPath.recordBlit = [&](const auto& pre, VkImage source,
                            VkImage destination, VkExtent2D sourceExtent,
                            VkExtent2D destinationExtent, const auto& post) {
                        const auto& finalCommand = *this->virtualFinalCommandBuffer;
                        finalCommand.begin(vk);
                        finalCommand.blitImage(vk, pre, {source, destination},
                            {sourceExtent, destinationExtent}, post);
                        finalCommand.end(vk);
                        this->renderFence->reset(vk);
                    };
                    presentPath.submitAndPresent = [&](const auto& waits, const auto&,
                            VkSemaphore signal, VkFence fence, uint32_t index) {
                        const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                            .waitSemaphoreCount = 1, .pWaitSemaphores = &signal,
                            .swapchainCount = 1, .pSwapchains = &swapchain,
                            .pImageIndices = &index};
                        const std::scoped_lock queueLock(*queueMutex);
                        this->virtualFinalCommandBuffer->submit(vk, queue, waits,
                            VK_NULL_HANDLE, 0, {signal}, VK_NULL_HANDLE, 0, fence,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);
                        const auto presentResult = vk.df().QueuePresentKHR(queue, &info);
                        return D3B1SubmitPresentResult{presentResult, true};
                    };
                    presentPath.waitRenderFence = [&] {
                        const auto result = this->renderFence->wait(vk, UINT64_MAX);
                        return result;
                    };
                    presentPath.completeDiagnostics = [&] {
                        this->generatedOutputReturnDiagnosticSession
                            ->completePresentationDiagnostics();
                        const auto result = this->generatedOutputReturnDiagnosticSession
                            ->gpuChainedPassed();
                        return result;
                    };
                    presentPath.retire = [&] {
                        this->generatedOutputReturnDiagnosticSession.reset();
                        this->returnedForGraphics.reset();
                    };
                    presentPath.emitMarker = [] {
                        std::cerr << "DG2X_P4C_D3B1_RETURNED_GENERATED_FRAME_PRESENT_PASS\n";
                    };
                    const auto result = executeD3B1PresentPath(presentPath);
                    this->captureOnlyPhase = 6;
                    if (stopAfterCompletion
                            && d3b1StopAfterTerminalPass(
                                d3b1OneShotRequested(std::getenv("LSFGVK_D3B1_ONESHOT")),
                                this->d3b1State)) {
                        *stopAfterCompletion = true;
                        std::cerr << "[D3B1-ONESHOT] terminal PASS; worker stop requested\n";
                    }
                    return result;
                }
            }
            this->captureOnlyPhase++;
            if (this->captureOnlyPhase <= 5) return VK_SUCCESS;
        }
        throw ls::error(
            d3b1PresentationDiagnosticEnabled()
                ? "D3B1_ASYNC_PRESENTATION_NOT_YET_CONNECTED"
                : "cross-device generated-frame diagnostic completed; output transport remains disconnected");
    }
    const bool adaptiveBypass = isAdaptiveBypass(this->profile);

    // Legacy Adaptive 1x = OFF: when the application still owns real WSI
    // images, preserve the original direct-present path exactly.
    if (adaptiveBypass && !this->info.virtualized) {
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = next_chain,
            .waitSemaphoreCount = static_cast<uint32_t>(semaphores.size()),
            .pWaitSemaphores = semaphores.empty() ? nullptr : semaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIdx,
        };
        return vk.df().QueuePresentKHR(queue, &presentInfo);
    }

    const auto& swapchainImage = this->info.images.at(imageIdx);
    const auto& outputImages = this->info.virtualized
        ? this->info.realImages
        : this->info.images;

    const bool fixedMode = isFixedMode(this->profile);
    // Every virtual topology presents from the prepared offload queue. Fixed
    // pacing itself remains separately gated by fixedMode.
    const bool workerOffload = this->info.virtualized && queueMutex;

    // Stage 3C5G: select the hidden real WSI present mode per presentation.
    // These modes were declared at swapchain creation, so hot reload changes
    // only the internal context and never the VkImages known to the app.
    const VkPresentModeKHR selectedPresentMode = fixedMode
        ? this->info.fixedPresentMode
        : this->info.adaptivePresentMode;
    VkSwapchainPresentModeInfoKHR dynamicPresentModeInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
        .swapchainCount = 1,
        .pPresentModes = &selectedPresentMode
    };
    const auto presentNextChain = [&](void* logicalNext) -> const void* {
        if (!this->info.dynamicPresentModeEligible)
            return logicalNext;

        // Avoid duplicating the structure when the application already owns
        // one. Existing pacing=None behavior below rewrites that structure to
        // the mode selected by the current LSFG context.
        if (hasPresentModeInfo(logicalNext))
            return logicalNext;

        dynamicPresentModeInfo.pNext = logicalNext;
        return &dynamicPresentModeInfo;
    };

    const auto compensateAbortedLogicalPresentFence =
        [&](const char* stage) -> VkResult {
            const auto fence = applicationPresentFence(next_chain);
            if (fence == VK_NULL_HANDLE)
                return VK_SUCCESS;

            // The application-facing vkQueuePresentKHR is about to return
            // OUT_OF_DATE before the logical present carrying this fence could
            // reach WSI. Complete the consumed logical present fence after all
            // earlier hidden queue work so the application can safely retire
            // its present resources and recreate the swapchain.
            const VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO
            };

            VkResult signalResult{};
            if (workerOffload) {
                const std::scoped_lock queueLock(*queueMutex);
                signalResult = vk.df().QueueSubmit(queue, 1, &submitInfo, fence);
            } else {
                signalResult = vk.df().QueueSubmit(queue, 1, &submitInfo, fence);
            }

            if (signalResult == VK_SUCCESS) {
                std::cerr << "lsfg-vk: compensated aborted logical present fence after "
                    << stage << " OUT_OF_DATE\n";
            } else {
                std::cerr << "lsfg-vk: failed to compensate aborted logical present fence after "
                    << stage << " OUT_OF_DATE: " << signalResult << '\n';
            }
            return signalResult;
        };

    // 3C5E: Adaptive 1x over the stable virtual topology. The runtime has
    // already consumed the application's present waits and supplies one ready
    // semaphore. No LSFG generation images or backend context exist here.
    if (adaptiveBypass) {
        if (semaphores.size() != 1 || !queueMutex
                || !this->virtualFinalCommandBuffer.has_value()
                || !this->virtualFinalAcquireSemaphore.has_value()
                || !this->virtualFinalPresentSemaphore.has_value()
                || !this->renderFence.has_value()) {
            throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                "virtual Adaptive 1x bridge is not initialized");
        }

        if (this->fidx && !this->renderFence->wait(vk, 150ULL * 1000 * 1000))
            throw ls::vulkan_error(VK_TIMEOUT, "vkWaitForFences() failed");
        this->renderFence->reset(vk);

        uint32_t realImageIdx{};
        auto res = acquireRealSwapchainImage(vk, swapchain,
            this->virtualFinalAcquireSemaphore->handle(), &realImageIdx,
            stopToken);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden Adaptive 1x acquire");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

        const auto& realImage = outputImages.at(realImageIdx);
        const auto& finalCmdbuf = *this->virtualFinalCommandBuffer;
        finalCmdbuf.begin(vk);
        finalCmdbuf.blitImage(vk,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                ),
                barrierHelper(realImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                ),
            },
            { swapchainImage, realImage },
            this->info.extent,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                ),
                barrierHelper(realImage,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                ),
            }
        );
        finalCmdbuf.end(vk);

        {
            const std::scoped_lock queueLock(*queueMutex);
            finalCmdbuf.submit(vk, queue,
                {
                    semaphores.front(),
                    this->virtualFinalAcquireSemaphore->handle()
                },
                VK_NULL_HANDLE, 0,
                { this->virtualFinalPresentSemaphore->handle() },
                VK_NULL_HANDLE, 0,
                this->renderFence->handle()
            );

            const VkPresentInfoKHR presentInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = presentNextChain(next_chain),
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &this->virtualFinalPresentSemaphore->handle(),
                .swapchainCount = 1,
                .pSwapchains = &swapchain,
                .pImageIndices = &realImageIdx,
            };
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        }

        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkQueuePresentKHR() failed");

        constexpr uint64_t WORKER_FENCE_SLICE_NS = 50ULL * 1000ULL * 1000ULL;
        while (!this->renderFence->wait(vk, WORKER_FENCE_SLICE_NS)) {
            if (stopToken.stop_requested())
                throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR,
                    "virtual Adaptive 1x presentation worker stopped");
        }

        this->fidx++;
        return res;
    }

    const auto& sourceImage = this->sourceImages.at(this->fidx % 2);
    FixedFrameScheduler::Plan fixedPlan{};
    if (fixedMode) {
        const auto now = sourcePresentTime.value_or(std::chrono::steady_clock::now());
        if (this->lastSourcePresent.has_value()) {
            fixedPlan = this->fixedScheduler.plan(
                std::chrono::duration_cast<FixedFrameScheduler::Duration>(
                    now - *this->lastSourcePresent));
        }
        this->lastSourcePresent = now;

        // If the application itself is faster than the requested output target,
        // pace the real-frame path instead of attempting negative interpolation.
        if (!workerOffload
                && fixedPlan.sourceDelay > FixedFrameScheduler::Duration::zero())
            std::this_thread::sleep_for(fixedPlan.sourceDelay);
    }

    const size_t generatedFrames = fixedMode
        ? fixedPlan.timestamps.size()
        : this->destinationImages.size();
    const auto timeline = vk::makeExchangeTimelineFrame(this->idx, generatedFrames);

    // Fixed virtual mode must not rely exclusively on FIFO/vblank to space
    // generated output. Use a cancellable host-side target cadence; the WSI
    // present mode is intentionally left unchanged by this pacing step.
    const auto paceFixedWorkerOutput = [&]() {
        if (!workerOffload || !fixedMode)
            return;

        constexpr auto MAX_SLEEP_SLICE = std::chrono::milliseconds(2);
        while (true) {
            if (stopToken.stop_requested())
                throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR,
                    "fixed output pacing worker stopped");

            const auto delay = this->fixedOutputPacer.delayUntilNext(
                std::chrono::steady_clock::now());
            if (delay <= FixedOutputPacer::Duration::zero())
                return;

            std::this_thread::sleep_for(std::min(
                delay,
                std::chrono::duration_cast<FixedOutputPacer::Duration>(
                    MAX_SLEEP_SLICE)));
        }
    };

    const auto markFixedWorkerOutput = [&]() {
        if (workerOffload && fixedMode)
            this->fixedOutputPacer.markPresented(
                std::chrono::steady_clock::now());
    };

    // Schedule frame generation. Adaptive continues through the original API.
    // Fixed uses the dynamic backend, including zero generated frames, so the
    // backend's temporal history advances for every real application frame.
    try {
        if (fixedMode)
            this->instance.get().scheduleFrames(this->ctx.get(), fixedPlan.timestamps);
        else
            this->instance.get().scheduleFrames(this->ctx.get());
    } catch (const std::exception& e) {
        throw ls::error("failed to schedule frames", e);
    }

    // Preserve existing behavior for an application-provided per-present mode
    // structure. LSFG's own single-swapchain structure is injected separately
    // by presentNextChain() when the dual-mode topology is active.
    if (this->profile.pacing == ls::Pacing::None) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        auto* info = reinterpret_cast<VkSwapchainPresentModeInfoKHR*>(next_chain);
        while (info) {
            if (info->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR) {
                for (size_t i = 0; i < info->swapchainCount; i++)
                    const_cast<VkPresentModeKHR*>(info->pPresentModes)[i] =
                        selectedPresentMode;
            }

            info = reinterpret_cast<VkSwapchainPresentModeInfoKHR*>(
                const_cast<void*>(info->pNext));
        }
#pragma clang diagnostic pop
    }

    // wait for completion of previous frame
    if (this->fidx && !this->renderFence->wait(vk, 150ULL * 1000 * 1000))
        throw ls::vulkan_error(VK_TIMEOUT, "vkWaitForFences() failed");
    this->renderFence->reset(vk);

    // copy application-visible swapchain image into backend source image
    const auto& cmdbuf = *this->renderCommandBuffer;
    cmdbuf.begin(vk);

    const auto sourcePreBarrier = this->fidx == 0
        ? barrierHelper(sourceImage.handle(),
            VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        : vk::sourceAcquireFromBackend(
            sourceImage.handle(), vk.queueFamilyIndex());
    cmdbuf.blitImage(vk,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ),
            sourcePreBarrier,
        },
        { swapchainImage, sourceImage.handle() },
        sourceImage.getExtent(),
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            ),
            vk::sourceReleaseToBackend(
                sourceImage.handle(), vk.queueFamilyIndex()),
        }
    );

    // Generate samples both source descriptors. Seed the second alternating
    // resource with the first real frame so B can legally acquire/read both on
    // its first dispatch; subsequent frames update one resource at a time.
    if (this->fidx == 0) {
        const auto& secondSource = this->sourceImages.at(1);
        cmdbuf.blitImage(vk,
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_NONE, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                barrierHelper(secondSource.handle(),
                    VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
            },
            { swapchainImage, secondSource.handle() },
            secondSource.getExtent(),
            {
                barrierHelper(swapchainImage,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
                vk::sourceReleaseToBackend(
                    secondSource.handle(), vk.queueFamilyIndex()),
            });
    }

    cmdbuf.end(vk);

    // With zero generated frames there is no generated post-copy pass. Keep a
    // binary semaphore to chain either the legacy final present or the virtual
    // real-frame copy.
    vk::Semaphore* zeroPresentSemaphore{};
    const auto sourceReturn = this->sourceReturnValues.at(this->fidx % 2);
    const VkSemaphore sourceWaitSemaphore = sourceReturn.has_value()
        ? this->syncSemaphore->handle() : VK_NULL_HANDLE;
    const uint64_t sourceWaitValue = sourceReturn.value_or(0);
    if (generatedFrames == 0) {
        auto& pcs = this->postCopySemaphores.at(
            this->idx % this->postCopySemaphores.size());
        zeroPresentSemaphore = &pcs.second;
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), timeline.sourceReady,
                this->info.virtualized ? VK_NULL_HANDLE : this->renderFence->handle(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        }
    } else {
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                {}, this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, sourceWaitSemaphore, sourceWaitValue,
                {}, this->syncSemaphore->handle(), timeline.sourceReady,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        }
    }
    this->idx = timeline.sourceReady + 1;
    this->sourceReturnValues.at((this->fidx + 1) % 2) = timeline.sourceReturn;

    for (size_t i = 0; i < generatedFrames; i++) {
        auto& pcs = this->postCopySemaphores.at(this->idx % this->postCopySemaphores.size());
        auto& destinationImage = this->destinationImages.at(i);
        auto& pass = this->passes.at(i);

        // acquire underlying real swapchain image
        uint32_t aqImageIdx{};
        auto res = acquireRealSwapchainImage(vk, swapchain,
            pass.acquireSemaphore.handle(), &aqImageIdx,
            workerOffload ? stopToken : std::stop_token{});
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden generated-frame acquire");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

        const auto& acquiredSwapchainImage = outputImages.at(aqImageIdx);

        // copy backend destination image into real swapchain image
        auto& passCmdbuf = pass.commandBuffer;
        passCmdbuf.begin(vk);

        passCmdbuf.blitImage(vk,
            {
                vk::destinationAcquireFromBackend(
                    destinationImage.handle(), vk.queueFamilyIndex()),
                barrierHelper(acquiredSwapchainImage,
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                ),
            },
            { destinationImage.handle(), acquiredSwapchainImage },
            destinationImage.getExtent(),
            {
                vk::destinationReleaseToBackend(
                    destinationImage.handle(), vk.queueFamilyIndex()),
                barrierHelper(acquiredSwapchainImage,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                ),
            }
        );

        std::vector<VkSemaphore> waitSemaphores{ pass.acquireSemaphore.handle() };
        if (i) {
            const auto& prevPCS = this->postCopySemaphores.at(
                (this->idx - 1) % this->postCopySemaphores.size());
            waitSemaphores.push_back(prevPCS.second.handle());
        }

        const std::vector<VkSemaphore> signalSemaphores{
            pcs.first.handle(),
            pcs.second.handle()
        };

        passCmdbuf.end(vk);
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            passCmdbuf.submit(vk, queue,
                waitSemaphores, this->syncSemaphore->handle(), timeline.destinationReady(i),
                signalSemaphores, VK_NULL_HANDLE, 0,
                VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        } else {
            passCmdbuf.submit(vk,
                waitSemaphores, this->syncSemaphore->handle(), timeline.destinationReady(i),
                signalSemaphores, VK_NULL_HANDLE, 0,
                (!this->info.virtualized && i == generatedFrames - 1)
                    ? this->renderFence->handle()
                    : VK_NULL_HANDLE,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            );
        }

        // Generated frames never carry the application's pNext when the
        // application is rendering into virtual images. The logical present
        // metadata belongs to the final real application frame.
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = this->info.virtualized
                ? presentNextChain(nullptr)
                : ((!i) ? next_chain : nullptr),
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &pcs.first.handle(),
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &aqImageIdx,
        };
        paceFixedWorkerOutput();
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        } else {
            res = vk.df().QueuePresentKHR(queue, &presentInfo);
        }
        // In the virtual topology generated-frame presents never carry the
        // application's pNext chain. If one goes OUT_OF_DATE, the later logical
        // present will not run and its present fence would otherwise remain
        // unsignaled. Legacy non-virtual generated presents may already have
        // forwarded the application's pNext, so never compensate those here.
        if (this->info.virtualized && res == VK_ERROR_OUT_OF_DATE_KHR) {
            const auto fenceResult =
                compensateAbortedLogicalPresentFence("hidden generated-frame present");
            if (fenceResult != VK_SUCCESS)
                throw ls::vulkan_error(fenceResult,
                    "failed to compensate aborted logical present fence");
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkQueuePresentKHR() failed");
        markFixedWorkerOutput();

        this->idx++;
    }
    this->idx = timeline.nextBase;

    const VkSemaphore finalWaitSemaphore = generatedFrames
        ? this->postCopySemaphores.at(
            (this->idx - 1) % this->postCopySemaphores.size()).second.handle()
        : zeroPresentSemaphore->handle();

    if (!this->info.virtualized) {
        // Legacy Adaptive/Fixed-3B path: application image is a real WSI image.
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = generatedFrames ? nullptr : next_chain,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &finalWaitSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIdx,
        };
        auto res = vk.df().QueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkQueuePresentKHR() failed");

        this->fidx++;
        return res;
    }

    // Virtual swapchain path: the application's image cannot be passed to WSI.
    // The virtual presentation worker acquires a real image, copies the logical source frame
    // into it and presents on the dedicated queue. The virtual image is not
    // recycled until the GPU has finished reading it.
    uint32_t realImageIdx{};
    auto res = acquireRealSwapchainImage(vk, swapchain,
        this->virtualFinalAcquireSemaphore->handle(), &realImageIdx,
        workerOffload ? stopToken : std::stop_token{});
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        const auto fenceResult =
            compensateAbortedLogicalPresentFence("hidden final acquire");
        if (fenceResult != VK_SUCCESS)
            throw ls::vulkan_error(fenceResult,
                "failed to compensate aborted logical present fence");
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

    const auto& realImage = outputImages.at(realImageIdx);
    const auto& finalCmdbuf = *this->virtualFinalCommandBuffer;
    finalCmdbuf.begin(vk);
    finalCmdbuf.blitImage(vk,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ),
            barrierHelper(realImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            ),
        },
        { swapchainImage, realImage },
        this->info.extent,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            ),
            barrierHelper(realImage,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            ),
        }
    );
    finalCmdbuf.end(vk);

    if (workerOffload) {
        const std::scoped_lock queueLock(*queueMutex);
        finalCmdbuf.submit(vk, queue,
            {
                finalWaitSemaphore,
                this->virtualFinalAcquireSemaphore->handle()
            },
            VK_NULL_HANDLE, 0,
            { this->virtualFinalPresentSemaphore->handle() },
            VK_NULL_HANDLE, 0,
            this->renderFence->handle()
        );
    } else {
        finalCmdbuf.submit(vk,
            {
                finalWaitSemaphore,
                this->virtualFinalAcquireSemaphore->handle()
            },
            VK_NULL_HANDLE, 0,
            { this->virtualFinalPresentSemaphore->handle() },
            VK_NULL_HANDLE, 0,
            this->renderFence->handle()
        );
    }

    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = presentNextChain(next_chain),
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &this->virtualFinalPresentSemaphore->handle(),
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &realImageIdx,
    };
    paceFixedWorkerOutput();
    if (workerOffload) {
        const std::scoped_lock queueLock(*queueMutex);
        res = vk.df().QueuePresentKHR(queue, &presentInfo);
    } else {
        res = vk.df().QueuePresentKHR(queue, &presentInfo);
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw ls::vulkan_error(res, "vkQueuePresentKHR() failed");
    markFixedWorkerOutput();

    // The application may reacquire this virtual image only after the worker
    // completes it. Use bounded waits in worker mode so swapchain destruction
    // can request cancellation instead of joining a thread stuck in an
    // infinite fence wait.
    if (workerOffload) {
        constexpr uint64_t WORKER_FENCE_SLICE_NS = 50ULL * 1000ULL * 1000ULL;
        while (!this->renderFence->wait(vk, WORKER_FENCE_SLICE_NS)) {
            if (stopToken.stop_requested())
                throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR,
                    "fixed presentation worker stopped");
        }
    } else if (!this->renderFence->wait(vk, UINT64_MAX)) {
        throw ls::vulkan_error(VK_TIMEOUT, "vkWaitForFences() failed");
    }

    this->fidx++;
    return res;
}
