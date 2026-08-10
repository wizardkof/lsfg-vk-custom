/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "virtual_swapchain_runtime.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

VirtualSwapchainRuntime::VirtualSwapchainRuntime(const vk::Vulkan& vk,
        VkQueue offloadQueue,
        std::shared_ptr<std::mutex> offloadMutex,
        size_t imageCount,
        const VirtualSwapchainImageSpec& spec) :
        vk(std::cref(vk)),
        offloadQueue(offloadQueue),
        offloadMutex(std::move(offloadMutex)),
        state(imageCount) {
    if (this->offloadQueue == VK_NULL_HANDLE || !this->offloadMutex)
        throw ls::error("virtual swapchain requires a dedicated graphics queue");
    if (!spec.supported())
        throw ls::error("virtual swapchain image specification is not supported");

    this->images.reserve(imageCount);
    for (size_t i = 0; i < imageCount; ++i) {
        auto formatList = spec.makeFormatListInfo();
        const void* pNext = spec.hasFormatList ? &formatList : nullptr;

        this->images.emplace_back(vk,
            spec.extent,
            spec.format,
            spec.usage,
            std::nullopt,
            std::nullopt,
            spec.imageOptions(pNext));
    }
}

VkResult VirtualSwapchainRuntime::getImages(uint32_t* count, VkImage* images) const noexcept {
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!images) {
        *count = static_cast<uint32_t>(this->images.size());
        return VK_SUCCESS;
    }

    const auto requested = *count;
    const auto available = static_cast<uint32_t>(this->images.size());
    const auto copied = std::min(requested, available);

    for (uint32_t i = 0; i < copied; ++i)
        images[i] = this->images.at(i).handle();

    *count = copied;
    return requested < available ? VK_INCOMPLETE : VK_SUCCESS;
}

VkResult VirtualSwapchainRuntime::signalAcquire(
        VkSemaphore semaphore, VkFence fence) const noexcept {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = semaphore == VK_NULL_HANDLE ? 0U : 1U;
    submitInfo.pSignalSemaphores = semaphore == VK_NULL_HANDLE ? nullptr : &semaphore;

    std::scoped_lock lock(*this->offloadMutex);
    return this->vk.get().df().QueueSubmit(
        this->offloadQueue, 1, &submitInfo, fence);
}

VkResult VirtualSwapchainRuntime::acquire(uint64_t timeout,
        VkSemaphore semaphore, VkFence fence, uint32_t* imageIndex) noexcept {
    if (!imageIndex)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::optional<uint32_t> acquired;
    if (timeout == 0) {
        acquired = this->state.tryAcquire();
    } else if (timeout == UINT64_MAX) {
        acquired = this->state.waitAcquire(VirtualSwapchainState::Duration::max());
    } else {
        using Rep = VirtualSwapchainState::Duration::rep;
        constexpr auto maxRep = static_cast<uint64_t>(std::numeric_limits<Rep>::max());
        const auto clamped = std::min(timeout, maxRep);
        acquired = this->state.waitAcquire(
            VirtualSwapchainState::Duration(static_cast<Rep>(clamped)));
    }

    if (!acquired.has_value()) {
        if (this->state.stopped())
            return VK_ERROR_OUT_OF_DATE_KHR;
        return timeout == 0 ? VK_NOT_READY : VK_TIMEOUT;
    }

    const auto res = this->signalAcquire(semaphore, fence);
    if (res != VK_SUCCESS) {
        (void)this->state.release(*acquired);
        return res;
    }

    *imageIndex = *acquired;
    return VK_SUCCESS;
}

bool VirtualSwapchainRuntime::beginPresent(uint32_t imageIndex) noexcept {
    const auto serial = this->presentSerial.fetch_add(1);
    if (!this->state.queuePresent(imageIndex, serial))
        return false;

    const auto present = this->state.waitPresent(VirtualSwapchainState::Duration::zero());
    return present.has_value()
        && present->imageIndex == imageIndex
        && present->serial == serial;
}

void VirtualSwapchainRuntime::completePresent(uint32_t imageIndex) noexcept {
    (void)this->state.complete(imageIndex);
}

void VirtualSwapchainRuntime::stop() noexcept {
    this->state.stop();
}

std::vector<VkImage> VirtualSwapchainRuntime::imageHandles() const {
    std::vector<VkImage> handles;
    handles.reserve(this->images.size());
    for (const auto& image : this->images)
        handles.push_back(image.handle());
    return handles;
}
