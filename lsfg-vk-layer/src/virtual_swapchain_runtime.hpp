/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "virtual_swapchain_image_spec.hpp"
#include "virtual_swapchain_state.hpp"
#include "graphics_final_queue.hpp"
#include "d3b3_production_seams.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

    /// Owns the application-visible images and asynchronous presentation worker
    /// of a virtual swapchain. The worker is deliberately independent from the
    /// Root-owned Swapchain object so hot reload never changes the lifetime of
    /// VkImage handles already returned to the application.
    class VirtualSwapchainRuntime {
    public:
        using PrePresentGate = std::function<PrePresentGateResult()>;
        using Presenter = std::function<VkResult(
            uint32_t imageIndex,
            VkSemaphore readySemaphore,
            VkSemaphore originalReadySemaphore,
            void* nextChain,
            std::stop_token stopToken,
            std::chrono::steady_clock::time_point sourcePresentTime,
            bool d3bSingleSwapchainEligible,
            const GraphicsFinalQueueInfo& graphicsFinalQueue,
            BorrowedGraphicsQueueLease& lease,
            bool& stopAfterCompletion)>;

        VirtualSwapchainRuntime(const vk::Vulkan& vk,
            VkQueue offloadQueue,
            std::shared_ptr<std::mutex> offloadMutex,
            size_t imageCount,
            const VirtualSwapchainImageSpec& spec,
            bool d3b2ReleaseCapable = false);
        ~VirtualSwapchainRuntime();

        [[nodiscard]] VkResult getImages(uint32_t* count, VkImage* images) const noexcept;

        [[nodiscard]] VkResult acquire(uint64_t timeout,
            VkSemaphore semaphore, VkFence fence, uint32_t* imageIndex) noexcept;

        /// Start the single consumer that owns virtual presentation work.
        void startWorker(Presenter presenter);

        /// Install the owner-driven retirement gate. The callback is invoked
        /// on the application thread before a new lease or bridge submit.
        /// The default gate is a no-op for frozen D3B2/D3B1 paths.
        void setPrePresentGate(PrePresentGate gate);

        /// Consume the application's present wait semaphores on the source
        /// presentation queue, then publish the virtual image to the worker.
        /// This keeps future application frames from blocking the dedicated
        /// offload queue ahead of the current worker frame. When synchronous
        /// is true, this call waits for the worker result so nextChain remains
        /// valid for the full presentation operation.
        [[nodiscard]] VkResult queuePresent(VkQueue sourceQueue,
            uint32_t sourceQueueFamily,
            uint32_t sourceQueueIndex,
            VkQueueFlags sourceQueueFlags,
            bool surfacePresentSupported,
            uint32_t imageIndex,
            const std::vector<VkSemaphore>& waitSemaphores,
            void* nextChain,
            bool synchronous) noexcept;

        void stop() noexcept;

        [[nodiscard]] std::vector<VkImage> imageHandles() const;
        [[nodiscard]] bool workerRunning() const noexcept;
        [[nodiscard]] VkResult workerResult() const noexcept;

        VirtualSwapchainRuntime(const VirtualSwapchainRuntime&) = delete;
        VirtualSwapchainRuntime& operator=(const VirtualSwapchainRuntime&) = delete;
        VirtualSwapchainRuntime(VirtualSwapchainRuntime&&) = delete;
        VirtualSwapchainRuntime& operator=(VirtualSwapchainRuntime&&) = delete;

    private:
        struct Completion {
            std::mutex mutex;
            std::condition_variable cv;
            VkResult result{VK_SUCCESS};
            bool done{};
        };

        struct Job {
            void* nextChain{};
            std::chrono::steady_clock::time_point sourcePresentTime;
            std::shared_ptr<Completion> completion;
            GraphicsFinalQueueInfo graphicsFinalQueue;
            std::shared_ptr<BorrowedGraphicsQueueLease> borrowedLease;
            bool d3bSingleSwapchainEligible{};
        };

        [[nodiscard]] VkResult signalAcquire(VkSemaphore semaphore, VkFence fence) const noexcept;
        [[nodiscard]] VkResult bridgePresentWaits(VkQueue sourceQueue,
            uint32_t imageIndex,
            const std::vector<VkSemaphore>& waitSemaphores) const noexcept;
        void workerLoop(std::stop_token stopToken) noexcept;
        void finishCompletion(const std::shared_ptr<Completion>& completion,
            VkResult result) noexcept;
        void failPending(VkResult result) noexcept;

        ls::R<const vk::Vulkan> vk;
        VkQueue offloadQueue{VK_NULL_HANDLE};
        std::shared_ptr<std::mutex> offloadMutex;
        std::vector<vk::Image> images;
        std::vector<vk::Semaphore> readySemaphores;
        std::vector<vk::Semaphore> originalReadySemaphores;
        VirtualSwapchainState state;

        std::atomic<uint64_t> presentSerial{1};
        std::atomic<VkResult> asyncResult{VK_SUCCESS};
        std::atomic_bool stopping{false};
        bool d3b2ReleaseCapable{};

        mutable std::mutex jobsMutex;
        std::unordered_map<uint64_t, Job> jobs;
        Presenter presenter;
        std::jthread worker;
        PrePresentGate prePresentGate;
    };

}
