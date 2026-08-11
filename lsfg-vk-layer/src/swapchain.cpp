/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "swapchain.hpp"
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
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
    [[nodiscard]] bool isFixedMode(const ls::GameConf& profile) {
        return profile.frame_generation_mode == ls::FrameGenerationMode::Fixed;
    }

    [[nodiscard]] bool isAdaptiveBypass(const ls::GameConf& profile) {
        return !isFixedMode(profile) && profile.multiplier == 1;
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
            VkImageLayout newLayout) {
        return VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = srcAccessMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = handle,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
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
            ls::GameConf profile, SwapchainInfo info) :
        instance(backend),
        fixedScheduler(profile.target_fps),
        fixedOutputPacer(profile.target_fps),
        profile(std::move(profile)), info(std::move(info)) {
    // Adaptive multiplier == 1 keeps the Vulkan layer/profile active but
    // bypasses LSFG. Fixed mode ignores multiplier and remains active.
    if (isAdaptiveBypass(this->profile))
        return;

    const VkExtent2D extent = this->info.extent;
    const bool hdr = this->info.format > 57;

    std::vector<int> sourceFds(2);
    std::vector<int> destinationFds(generatedFrameCapacity(this->profile));

    this->sourceImages.reserve(sourceFds.size());
    for (int& fd : sourceFds)
        this->sourceImages.emplace_back(vk,
            extent, hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            std::nullopt, &fd);

    this->destinationImages.reserve(destinationFds.size());
    for (int& fd : destinationFds)
        this->destinationImages.emplace_back(vk,
            extent, hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            std::nullopt, &fd);

    int syncFd{};
    this->syncSemaphore.emplace(vk, 0, std::nullopt, &syncFd);

    try {
        this->ctx = ls::owned_ptr<ls::R<backend::Context>>(
            new ls::R<backend::Context>(backend.openContext(
                { sourceFds.at(0), sourceFds.at(1) }, destinationFds, syncFd,
                extent.width, extent.height,
                hdr, 1.0F / this->profile.flow_scale, this->profile.performance_mode
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
    this->renderFence.emplace(vk);
    if (this->info.virtualized) {
        this->virtualFinalCommandBuffer.emplace(vk);
        this->virtualFinalAcquireSemaphore.emplace(vk);
        this->virtualFinalPresentSemaphore.emplace(vk);
    }
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

VkResult Swapchain::present(const vk::Vulkan& vk,
        VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
        VkSwapchainKHR swapchain,
        void* next_chain, uint32_t imageIdx,
        const std::vector<VkSemaphore>& semaphores,
        std::stop_token stopToken,
        std::optional<std::chrono::steady_clock::time_point> sourcePresentTime) {
    // Adaptive 1x = OFF: pass the original application frame straight through.
    // Fixed mode intentionally ignores multiplier.
    if (isAdaptiveBypass(this->profile)) {
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
    const auto& sourceImage = this->sourceImages.at(this->fidx % 2);

    const bool fixedMode = isFixedMode(this->profile);
    const bool workerOffload = this->info.virtualized && fixedMode && queueMutex;
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

    // Fixed virtual mode must not rely exclusively on FIFO/vblank to space
    // generated output. Use a cancellable host-side target cadence; the WSI
    // present mode is intentionally left unchanged by this pacing step.
    const auto paceFixedWorkerOutput = [&]() {
        if (!workerOffload)
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
        if (workerOffload)
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

    // update present mode when not using pacing
    if (this->profile.pacing == ls::Pacing::None) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        auto* info = reinterpret_cast<VkSwapchainPresentModeInfoEXT*>(next_chain);
        while (info) {
            if (info->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT) {
                for (size_t i = 0; i < info->swapchainCount; i++)
                    const_cast<VkPresentModeKHR*>(info->pPresentModes)[i] =
                        VK_PRESENT_MODE_FIFO_KHR;
            }

            info = reinterpret_cast<VkSwapchainPresentModeInfoEXT*>(const_cast<void*>(info->pNext));
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

    cmdbuf.blitImage(vk,
        {
            barrierHelper(swapchainImage,
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ),
            barrierHelper(sourceImage.handle(),
                VK_ACCESS_NONE,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            ),
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
        }
    );

    cmdbuf.end(vk);

    // With zero generated frames there is no generated post-copy pass. Keep a
    // binary semaphore to chain either the legacy final present or the virtual
    // real-frame copy.
    vk::Semaphore* zeroPresentSemaphore{};
    if (generatedFrames == 0) {
        auto& pcs = this->postCopySemaphores.at(
            this->idx % this->postCopySemaphores.size());
        zeroPresentSemaphore = &pcs.second;
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, VK_NULL_HANDLE, 0,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), this->idx++,
                VK_NULL_HANDLE
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, VK_NULL_HANDLE, 0,
                { zeroPresentSemaphore->handle() },
                this->syncSemaphore->handle(), this->idx++,
                this->info.virtualized ? VK_NULL_HANDLE : this->renderFence->handle()
            );
        }
    } else {
        if (workerOffload) {
            const std::scoped_lock queueLock(*queueMutex);
            cmdbuf.submit(vk, queue,
                semaphores, VK_NULL_HANDLE, 0,
                {}, this->syncSemaphore->handle(), this->idx++
            );
        } else {
            cmdbuf.submit(vk,
                semaphores, VK_NULL_HANDLE, 0,
                {}, this->syncSemaphore->handle(), this->idx++
            );
        }
    }

    for (size_t i = 0; i < generatedFrames; i++) {
        auto& pcs = this->postCopySemaphores.at(this->idx % this->postCopySemaphores.size());
        auto& destinationImage = this->destinationImages.at(i);
        auto& pass = this->passes.at(i);

        // acquire underlying real swapchain image
        uint32_t aqImageIdx{};
        auto res = acquireRealSwapchainImage(vk, swapchain,
            pass.acquireSemaphore.handle(), &aqImageIdx,
            workerOffload ? stopToken : std::stop_token{});
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkAcquireNextImageKHR() failed");

        const auto& acquiredSwapchainImage = outputImages.at(aqImageIdx);

        // copy backend destination image into real swapchain image
        auto& passCmdbuf = pass.commandBuffer;
        passCmdbuf.begin(vk);

        passCmdbuf.blitImage(vk,
            {
                barrierHelper(destinationImage.handle(),
                    VK_ACCESS_NONE,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                ),
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
                waitSemaphores, this->syncSemaphore->handle(), this->idx,
                signalSemaphores, VK_NULL_HANDLE, 0,
                VK_NULL_HANDLE
            );
        } else {
            passCmdbuf.submit(vk,
                waitSemaphores, this->syncSemaphore->handle(), this->idx,
                signalSemaphores, VK_NULL_HANDLE, 0,
                (!this->info.virtualized && i == generatedFrames - 1)
                    ? this->renderFence->handle()
                    : VK_NULL_HANDLE
            );
        }

        // Generated frames never carry the application's pNext when the
        // application is rendering into virtual images. The logical present
        // metadata belongs to the final real application frame.
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = (!this->info.virtualized && !i) ? next_chain : nullptr,
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
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw ls::vulkan_error(res, "vkQueuePresentKHR() failed");
        markFixedWorkerOutput();

        this->idx++;
    }

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
    // The Fixed worker acquires a real image, copies the logical source frame
    // into it and presents on the dedicated queue. The virtual image is not
    // recycled until the GPU has finished reading it.
    uint32_t realImageIdx{};
    auto res = acquireRealSwapchainImage(vk, swapchain,
        this->virtualFinalAcquireSemaphore->handle(), &realImageIdx,
        workerOffload ? stopToken : std::stop_token{});
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
        .pNext = next_chain,
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
