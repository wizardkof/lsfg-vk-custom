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
#include "device_retirement_reactor.hpp"
#include "presented_physical_image_lease.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

    struct VirtualImageBacking final {
        VirtualImageBacking(const vk::Vulkan&, const VirtualSwapchainImageSpec&,
            const vk::ImageCreateOptions&);
        vk::Image image;
        vk::Semaphore ready;
        vk::Semaphore originalReady;
    };

    enum class VirtualPresentCompletionStatus : uint8_t {
        RETIRED, NOT_READY, DEVICE_LOST, FAILED
    };

    /// Move-only authority for resources protected by a pending present fence.
    /// Concrete terminal operations retain their fence, semaphore and cleanup
    /// capabilities here; the runtime never reconstructs them from raw handles.
    class VirtualPresentPendingOperation {
    public:
        VirtualPresentPendingOperation() = default;
        VirtualPresentPendingOperation(const VirtualPresentPendingOperation&) = delete;
        VirtualPresentPendingOperation& operator=(const VirtualPresentPendingOperation&) = delete;
        virtual ~VirtualPresentPendingOperation() = default;
        virtual void activateCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>&) noexcept {}
        virtual void redirectCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>&) noexcept {}
        [[nodiscard]] virtual uint64_t swapchainLifecycleIdentity() const noexcept {
            return 0;
        }
        [[nodiscard]] virtual VirtualPresentCompletionStatus tryComplete() noexcept = 0;
    };

    enum class PresentExecutionStatus : uint8_t {
        COMPLETED, PENDING, FAILED
    };

    enum class PresentResultOrigin : uint8_t {
        INTERNAL, DOWNSTREAM_LOGICAL_PRESENT
    };

    enum class PresentTransactionPhase : uint8_t { PRE_COMMIT, POST_COMMIT };

    [[nodiscard]] VkResult publicPresentResult(VkResult,
        PresentResultOrigin, PresentTransactionPhase) noexcept;
    [[nodiscard]] VkResult publicPrePresentGateResult(
        PrePresentGateResult) noexcept;
    [[nodiscard]] VkResult publicBridgePresentResult(VkResult) noexcept;
    [[nodiscard]] bool allowedPublicPresentResult(VkResult) noexcept;

    struct PresentExecutionResult {
        PresentExecutionStatus status{PresentExecutionStatus::FAILED};
        VkResult result{VK_ERROR_UNKNOWN};
        PresentResultOrigin origin{PresentResultOrigin::INTERNAL};
        std::unique_ptr<VirtualPresentPendingOperation> pending;

        [[nodiscard]] static PresentExecutionResult completed(VkResult result,
            PresentResultOrigin origin) noexcept;
        [[nodiscard]] static PresentExecutionResult pendingCompletion(VkResult result,
            std::unique_ptr<VirtualPresentPendingOperation>,
            PresentResultOrigin origin) noexcept;
        [[nodiscard]] static PresentExecutionResult failed(VkResult result,
            PresentResultOrigin origin = PresentResultOrigin::INTERNAL) noexcept;
    };

    /// Per-runtime, per-image persistent completion storage. Only the worker
    /// mutates it, so ownership cannot be overwritten or shared with Root.
    class VirtualPresentPendingSet {
    public:
        struct Entry {
            uint32_t imageIndex{};
            uint64_t lifecycleIdentity{};
            std::unique_ptr<VirtualPresentPendingOperation> operation;
        };
        class ReservedInstall {
        public:
            ReservedInstall() = default;
            ReservedInstall(const ReservedInstall&) = delete;
            ReservedInstall& operator=(const ReservedInstall&) = delete;
            ReservedInstall(ReservedInstall&&) noexcept = default;
            ReservedInstall& operator=(ReservedInstall&&) noexcept = default;
            [[nodiscard]] bool valid() const noexcept { return node.size() == 1; }
        private:
            std::list<Entry> node;
            friend class VirtualPresentPendingSet;
        };
        [[nodiscard]] std::optional<ReservedInstall> reserveInstall(
            uint32_t imageIndex, uint64_t lifecycleIdentity) noexcept;
        [[nodiscard]] bool installReserved(ReservedInstall&,
            std::unique_ptr<VirtualPresentPendingOperation>,
            const std::shared_ptr<DeviceRetirementWakeTarget>& wakeTarget = {}) noexcept;
        [[nodiscard]] bool install(uint32_t imageIndex, uint64_t lifecycleIdentity,
            std::unique_ptr<VirtualPresentPendingOperation>,
            const std::shared_ptr<DeviceRetirementWakeTarget>& wakeTarget = {}) noexcept;
        [[nodiscard]] VirtualPresentCompletionStatus tryCompleteOnce(
            VirtualSwapchainState&) noexcept;
        [[nodiscard]] bool contains(uint32_t imageIndex) const noexcept;
        [[nodiscard]] size_t size() const noexcept { return entries.size(); }
        [[nodiscard]] bool empty() const noexcept { return entries.empty(); }
    private:
        std::list<Entry> entries;
        friend class DeferredVirtualRetirementOwner;
    };

    /// Per-device physical retirement owner. It receives only already-activated
    /// operations after their runtime state machine has stopped.
    class DeferredVirtualRetirementOwner final :
            public DeviceRetirementWakeTarget,
            public std::enable_shared_from_this<DeferredVirtualRetirementOwner> {
    public:
        DeferredVirtualRetirementOwner() = default;
        ~DeferredVirtualRetirementOwner() = default;
        void adopt(VirtualPresentPendingSet&) noexcept;
        void notifyDeviceRetirement() noexcept override;
        [[nodiscard]] size_t pendingCount() const noexcept;
    private:
        std::list<VirtualPresentPendingSet::Entry> operations;
        mutable std::mutex mutex;
    };

    /// Owns the application-visible images and asynchronous presentation worker
    /// of a virtual swapchain. The worker is deliberately independent from the
    /// Root-owned Swapchain object so hot reload never changes the lifetime of
    /// VkImage handles already returned to the application.
    class VirtualSwapchainRuntime {
    public:
        class BatchPresentReservation {
        public:
            BatchPresentReservation() = default;
            BatchPresentReservation(const BatchPresentReservation&) = delete;
            BatchPresentReservation& operator=(const BatchPresentReservation&) = delete;
            BatchPresentReservation(BatchPresentReservation&&) noexcept;
            BatchPresentReservation& operator=(BatchPresentReservation&&) noexcept;
            ~BatchPresentReservation() noexcept;
            [[nodiscard]] bool valid() const noexcept;
            [[nodiscard]] bool commitAfterBridge() noexcept;
            [[nodiscard]] bool abortCleanly() noexcept;
            void retainIndeterminate() noexcept { terminal = true; }
            [[nodiscard]] bool installPublishedCompletion() noexcept;
            [[nodiscard]] uint32_t imageIndex() const noexcept {
                return stateReservation.imageIndex;
            }
            [[nodiscard]] uint64_t serial() const noexcept {
                return stateReservation.serial;
            }
            [[nodiscard]] const std::shared_ptr<VirtualImageBacking>&
            imageBacking() const noexcept { return backing; }
            [[nodiscard]] const std::shared_ptr<
                std::unique_ptr<VirtualPresentPendingOperation>>&
            completionPublication() const noexcept { return publication; }
        private:
            VirtualSwapchainRuntime* owner{};
            VirtualSwapchainState::PreparedBatchPresent stateReservation;
            VirtualPresentPendingSet::ReservedInstall pendingReservation;
            std::shared_ptr<VirtualImageBacking> backing;
            std::shared_ptr<std::unique_ptr<VirtualPresentPendingOperation>> publication;
            bool bridgeCommitted{};
            bool terminal{};
            friend class VirtualSwapchainRuntime;
        };
        using PrePresentGate = std::function<PrePresentGateResult()>;
        using Presenter = std::function<PresentExecutionResult(
            uint32_t imageIndex,
            VkSemaphore readySemaphore,
            VkSemaphore originalReadySemaphore,
            void* nextChain,
            std::stop_token stopToken,
            std::chrono::steady_clock::time_point sourcePresentTime,
            bool d3bSingleSwapchainEligible,
            const GraphicsFinalQueueInfo& graphicsFinalQueue,
            BorrowedGraphicsQueueLease& lease,
            bool& stopAfterCompletion,
            PresentedPhysicalImageIdentity* presentedIdentity,
            std::shared_ptr<void> gpuBacking)>;

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
            bool synchronous,
            PresentedPhysicalImageIdentity* presentedIdentity = nullptr) noexcept;
        [[nodiscard]] std::optional<BatchPresentReservation>
        reserveBatchPresent(uint32_t imageIndex, VkResult* failure = nullptr) noexcept;

        void stop() noexcept;
        [[nodiscard]] bool stopAndDetach(
            const std::shared_ptr<DeferredVirtualRetirementOwner>&) noexcept;
        [[nodiscard]] bool installPendingForTesting(
            uint32_t imageIndex, uint64_t lifecycleIdentity,
            std::unique_ptr<VirtualPresentPendingOperation>) noexcept;

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
            std::shared_ptr<PresentedPhysicalImageIdentity> presentedIdentity;
            std::shared_ptr<VirtualImageBacking> imageBacking;
            bool d3bSingleSwapchainEligible{};
        };

        [[nodiscard]] VkResult signalAcquire(VkSemaphore semaphore, VkFence fence) const noexcept;
        [[nodiscard]] VkResult bridgePresentWaits(VkQueue sourceQueue,
            uint32_t imageIndex, uint64_t epoch,
            const std::vector<VkSemaphore>& waitSemaphores) const noexcept;
        void workerLoop(std::stop_token stopToken) noexcept;
        [[nodiscard]] VirtualPresentCompletionStatus tryCompletePendingPresents() noexcept;
        void finishCompletion(const std::shared_ptr<Completion>& completion,
            VkResult result) noexcept;
        void failPending(VkResult result) noexcept;

        ls::R<const vk::Vulkan> vk;
        VkQueue offloadQueue{VK_NULL_HANDLE};
        std::shared_ptr<std::mutex> offloadMutex;
        std::vector<std::shared_ptr<VirtualImageBacking>> imageBackings;
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

        VirtualPresentPendingSet pendingPresents;
        mutable std::mutex pendingPresentsMutex;
        std::shared_ptr<DeviceRetirementWakeTarget> retirementWakeTarget;
    };

}
