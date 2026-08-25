/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
#include <unordered_map>
#endif

namespace lsfgvk::backend {

    class [[gnu::visibility("default")]] ContextImpl;
    class [[gnu::visibility("default")]] InstanceImpl;
    class [[gnu::visibility("default")]] RuntimePrepassSessionImpl;
    class [[gnu::visibility("default")]] RuntimeGenerateSessionImpl;
    class RuntimeGenerateDiagnosticPendingState;
    class RuntimeGenerationOperationRetirement;
    class RuntimeIngestPending;
    struct RuntimeGeneratedFrameTokenTestAccess;
    struct RuntimeGenerateDiagnosticPendingTestAccess;
    enum class TemporalSourceSlot : uint8_t;

    using Context = ContextImpl;
    using RuntimePrepassSession = RuntimePrepassSessionImpl;
    using RuntimeGenerateSession = RuntimeGenerateSessionImpl;

    struct RuntimeShadowSplitResourceSnapshot {
        VkCommandBuffer ingestCommand{};
        VkCommandBuffer generateCommand{};
        VkSemaphore ingestReady{};
        VkFence ingestFence{};
        uint32_t queueFamily{VK_QUEUE_FAMILY_IGNORED};
        bool available{};
    };
    struct RuntimeShadowIngestSnapshot {
        uint64_t frameId{};
        TemporalSourceSlot slot{};
        uint64_t epoch{};
        VkSemaphore waitSemaphore{};
        VkPipelineStageFlags waitStage{};
        VkCommandBuffer commandBuffer{};
        VkSemaphore signalSemaphore{};
        VkFence fence{};
        bool submitAccepted{};
        bool submissionRetired{};
        bool payloadRetired{};
        bool frameTransportReusable{};
        bool ingestReadyReusable{};
        bool failed{};
        bool deviceLost{};
        uint32_t shadowSubmitCount{};
        uint32_t shadowGenerateSubmitCount{};
    };
    struct RuntimeShadowGenerateSnapshot {
        uint64_t generationId{};
        uint64_t olderFrameId{};
        uint64_t newerFrameId{};
        TemporalSourceSlot olderSlot{};
        TemporalSourceSlot newerSlot{};
        VkSemaphore waitSemaphore{};
        VkPipelineStageFlags waitStage{};
        VkCommandBuffer commandBuffer{};
        VkSemaphore signalSemaphore{};
        VkFence fence{};
        VkImage generatedImage{};
        bool submitAccepted{};
        bool executionRetired{};
        bool ingestReadyReusable{};
        bool generationReadySignalOutstanding{};
        bool generationReadyWaitSubmitted{};
        bool generationReadyWaitRetired{};
        bool generationReadyReusable{};
        bool generatedOutputLive{};
        bool generatedOutputRetired{};
        bool failed{};
        bool deviceLost{};
        uint32_t totalShadowSubmitCount{};
        uint32_t bReturnSubmitCount{};
        uint32_t aReturnSubmitCount{};
    };

    using RuntimeGenerationId = uint64_t;
    enum class RuntimeGenerateMode : uint8_t { OneShot, SerialReusable };
    enum class RuntimeIngestIntent : uint8_t { WARMUP_TEMPORAL, GENERATE_SOURCE };
    enum class RuntimeIngestRetirementStatus : uint8_t { NOT_READY, RETIRED };
    enum class RuntimeRetirementStatus : uint8_t {
        NOT_READY, RETIRED, DEVICE_LOST, FAILED
    };

    enum class TemporalSourceSlot : uint8_t { Slot0 = 0, Slot1 = 1 };

    [[nodiscard]] constexpr size_t temporalSourceSlotIndex(TemporalSourceSlot slot) {
        return static_cast<size_t>(slot);
    }

    struct RuntimeTemporalPairIdentity {
        TemporalSourceSlot olderSlot{};
        TemporalSourceSlot newerSlot{};
        uint64_t olderFrameId{};
        uint64_t newerFrameId{};
        RuntimeGenerationId generationId{};
    };

    [[nodiscard]] constexpr bool validTemporalPair(
            const RuntimeTemporalPairIdentity& pair) noexcept {
        return temporalSourceSlotIndex(pair.olderSlot) < 2
            && temporalSourceSlotIndex(pair.newerSlot) < 2
            && pair.olderSlot != pair.newerSlot
            && pair.olderFrameId != 0 && pair.newerFrameId != 0
            && pair.generationId != 0;
    }

    [[nodiscard]] constexpr size_t temporalPairFrameIndex(
            const RuntimeTemporalPairIdentity& pair) {
        return temporalSourceSlotIndex(pair.newerSlot);
    }

    class [[gnu::visibility("default")]] RuntimeGeneratedFrameToken {
    public:
        RuntimeGeneratedFrameToken() noexcept = default;
        RuntimeGeneratedFrameToken(const RuntimeGeneratedFrameToken&) = delete;
        RuntimeGeneratedFrameToken& operator=(const RuntimeGeneratedFrameToken&) = delete;
        RuntimeGeneratedFrameToken(RuntimeGeneratedFrameToken&&) noexcept;
        RuntimeGeneratedFrameToken& operator=(RuntimeGeneratedFrameToken&&) noexcept;
        ~RuntimeGeneratedFrameToken() = default;
        [[nodiscard]] bool valid() const noexcept {
            return generation != 0 && image != VK_NULL_HANDLE && !sessionLifetime.expired();
        }
        [[nodiscard]] RuntimeGenerationId identity() const noexcept { return generation; }
        [[nodiscard]] VkImage imageHandle() const noexcept { return image; }
        [[nodiscard]] VkExtent2D extentValue() const noexcept { return extent; }
        [[nodiscard]] VkFormat formatValue() const noexcept { return format; }
        [[nodiscard]] VkImageLayout layoutValue() const noexcept { return layout; }
        [[nodiscard]] uint32_t queueFamily() const noexcept { return family; }
        void consume();
    private:
        friend class RuntimeGenerateSessionImpl;
        friend struct RuntimeGeneratedFrameTokenTestAccess;
        RuntimeGeneratedFrameToken(RuntimeGenerationId id, VkImage image, VkExtent2D extent,
            VkFormat format, VkImageLayout layout, uint32_t family,
            std::weak_ptr<const uint8_t> sessionLifetime) noexcept;
        RuntimeGenerationId generation{};
        VkImage image{};
        VkExtent2D extent{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        uint32_t family{};
        std::weak_ptr<const uint8_t> sessionLifetime;
    };

    struct [[gnu::visibility("default")]] RuntimeGeneratedFrameMetadata {
        RuntimeGenerationId generation{};
        VkExtent2D extent{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        size_t byteCount{};
        size_t nonzeroByteCount{};
        uint64_t checksum{};
    };

    struct [[gnu::visibility("default")]] RuntimeGenerateDiagnosticResult {
        RuntimeGeneratedFrameMetadata metadata;
        RuntimeGeneratedFrameToken frame;
    };

    class [[gnu::visibility("default")]] RuntimeGenerateDiagnosticPending {
    public:
        RuntimeGenerateDiagnosticPending() noexcept = default;
        RuntimeGenerateDiagnosticPending(const RuntimeGenerateDiagnosticPending&) = delete;
        RuntimeGenerateDiagnosticPending& operator=(const RuntimeGenerateDiagnosticPending&) = delete;
        RuntimeGenerateDiagnosticPending(RuntimeGenerateDiagnosticPending&&) noexcept;
        RuntimeGenerateDiagnosticPending& operator=(RuntimeGenerateDiagnosticPending&&) noexcept;
        ~RuntimeGenerateDiagnosticPending() = default;
        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] RuntimeGenerationId identity() const noexcept;
        [[nodiscard]] VkImage imageHandle() const noexcept;
        [[nodiscard]] VkExtent2D extentValue() const noexcept;
        [[nodiscard]] VkFormat formatValue() const noexcept;
        [[nodiscard]] VkImageLayout layoutValue() const noexcept;
        [[nodiscard]] uint32_t queueFamily() const noexcept;
        [[nodiscard]] VkSemaphore readinessSemaphore() const noexcept;
        [[nodiscard]] VkFence sourceReadRetirementFence() const noexcept;
        [[nodiscard]] RuntimeTemporalPairIdentity temporalPair() const noexcept;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        [[nodiscard]] bool transportConsumedForTesting() const noexcept;
        [[nodiscard]] bool retirementAuthorityIssuedForTesting() const noexcept;
        [[nodiscard]] bool operationRetiredForTesting() const noexcept;
#endif
        [[nodiscard]] RuntimeGenerationOperationRetirement operationRetirementAuthority();
        void consumeTransport();
    private:
        friend class RuntimeGenerateSessionImpl;
        friend struct RuntimeGenerateDiagnosticPendingTestAccess;
        explicit RuntimeGenerateDiagnosticPending(
            std::shared_ptr<RuntimeGenerateDiagnosticPendingState>) noexcept;
        RuntimeGenerateDiagnosticPending(RuntimeGenerationId, VkImage, VkExtent2D,
            VkFormat, VkImageLayout, uint32_t, VkSemaphore,
            std::weak_ptr<const uint8_t>);
        std::shared_ptr<RuntimeGenerateDiagnosticPendingState> pending;
    };

    class [[gnu::visibility("default")]] RuntimeGenerationOperationRetirement {
    public:
        RuntimeGenerationOperationRetirement() noexcept = default;
        RuntimeGenerationOperationRetirement(const RuntimeGenerationOperationRetirement&) = delete;
        RuntimeGenerationOperationRetirement& operator=(const RuntimeGenerationOperationRetirement&) = delete;
        RuntimeGenerationOperationRetirement(RuntimeGenerationOperationRetirement&&) noexcept = default;
        RuntimeGenerationOperationRetirement& operator=(RuntimeGenerationOperationRetirement&&) noexcept = default;
        [[nodiscard]] bool valid() const noexcept { return pending != nullptr; }
        [[nodiscard]] RuntimeGenerationId generationId() const noexcept;
    private:
        friend class RuntimeGenerateDiagnosticPending;
        friend class RuntimeGenerateSessionImpl;
        explicit RuntimeGenerationOperationRetirement(
            std::shared_ptr<RuntimeGenerateDiagnosticPendingState> value) : pending(std::move(value)) {}
        std::shared_ptr<RuntimeGenerateDiagnosticPendingState> pending;
    };

    ///
    /// Primitive exception class that deliveres a detailed error message
    ///
    class [[gnu::visibility("default")]] error : public std::runtime_error {
    public:
        ///
        /// Construct an error
        ///
        /// @param msg Error message.
        /// @param inner Inner exception.
        ///
        explicit error(const std::string &msg, const std::exception &inner);

        ///
        /// Construct an error
        ///
        /// @param msg Error message.
        ///
        explicit error(const std::string &msg);

        error(const error &) = default;
        error &operator=(const error &) = default;
        error(error &&) = default;
        error &operator=(error &&) = default;
        ~error() override;
    };

    /// Function type for picking a device based on its stable identity.
    using DevicePicker = std::function<bool(const vk::PhysicalDeviceIdentity& identity)>;
    /// One-shot observer for the complete backend-visible device inventory.
    using DeviceEnumerationObserver = std::function<void(
        const std::vector<vk::PhysicalDeviceSnapshot>& devices)>;

    ///
    /// Main entry point of the library
    ///
    class [[gnu::visibility("default")]] Instance {
    public:
        ///
        /// Create a lsfg-vk instance
        ///
        /// @param devicePicker Function that picks a physical device based on some identifiers.
        /// @param shaderDllPath Path to the Lossless.dll file to load shaders from.
        /// @param allowLowPrecision Whether to load low-precision (FP16) shaders if supported.
        ///
        /// @throws backend::error on failure
        ///
        Instance(
            const DevicePicker& devicePicker,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision,
            const DeviceEnumerationObserver& deviceObserver = {}
        );
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        /// Test-only construction seam for executing the real inactive shadow
        /// control path against externally supplied deterministic Vulkan dispatch.
        /// Normal builds expose neither this constructor nor a runtime selector.
        Instance(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
            vk::VulkanInstanceFuncs instanceFuncs, vk::VulkanDeviceFuncs deviceFuncs,
            const std::unordered_map<uint32_t, std::vector<uint8_t>>& shaderResources);
#endif

        /// Identity of the physical device selected for frame generation.
        [[nodiscard]] const vk::PhysicalDeviceIdentity& deviceIdentity() const;
        /// Complete physical-device inventory seen by the backend VkInstance.
        [[nodiscard]] const std::vector<vk::PhysicalDeviceSnapshot>& visibleDevices() const;
        /// Narrow Vulkan endpoint used by the P3D runtime cross-device control channel.
        /// Handles remain owned by backend::Instance.
        [[nodiscard]] vk::RuntimeExchangeEndpoint runtimeExchangeEndpoint() const;

        /// Run the real LSFG prepass against one frame already resident on the
        /// backend device. This is a diagnostic-only, zero-generation path.
        void validateRuntimePrepass(VkImage transportImage, VkExtent2D extent,
            VkFormat transportFormat, uint64_t transportModifier,
            float flow, bool perf);
        RuntimePrepassSession& openRuntimePrepassSession(VkExtent2D extent,
            VkFormat transportFormat, uint64_t transportModifier,
            float flow, bool perf);
        void processRuntimePrepass(RuntimePrepassSession& session,
            VkImage transportImage, vk::SyncFdPayload payload);
        void closeRuntimePrepassSession(const RuntimePrepassSession& session);
        RuntimeGenerateSession& openRuntimeGenerateSession(
            VkExtent2D extent, VkFormat transportFormat, uint64_t transportModifier,
            float flow, bool perf,
            RuntimeGenerateMode mode = RuntimeGenerateMode::OneShot);
        std::optional<RuntimeGenerateDiagnosticResult> processRuntimeGenerateDiagnostic(
            RuntimeGenerateSession& session,
            VkImage transportImage, vk::SyncFdPayload payload);
        std::optional<RuntimeGenerateDiagnosticPending> submitRuntimeGenerateDiagnostic(
            RuntimeGenerateSession& session,
            VkImage transportImage, vk::SyncFdPayload payload);
        std::optional<RuntimeGenerateDiagnosticPending> submitRuntimeGenerateExplicit(
            RuntimeGenerateSession& session, VkImage transportImage,
            vk::SyncFdPayload payload, TemporalSourceSlot destinationSlot,
            uint64_t frameId, std::optional<RuntimeTemporalPairIdentity> pair = std::nullopt);
        RuntimeGenerateDiagnosticResult completeRuntimeGenerateDiagnostic(
            RuntimeGenerateSession& session,
            RuntimeGenerateDiagnosticPending&& pending);
        void retireRuntimeGenerateOperation(RuntimeGenerateSession& session,
            RuntimeGenerationOperationRetirement&& authority);
        void closeRuntimeGenerateSession(
            const RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowSplitResourceSnapshot inspectRuntimeSplitResources(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] RuntimeShadowIngestSnapshot submitRuntimeIngest(
            RuntimeGenerateSession& session, VkImage transportImage,
            vk::SyncFdPayload payload, TemporalSourceSlot destinationSlot,
            uint64_t frameId, RuntimeIngestIntent intent);
        [[nodiscard]] RuntimeIngestRetirementStatus tryRetireRuntimeIngest(
            RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowIngestSnapshot inspectRuntimeIngest(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] RuntimeShadowGenerateSnapshot submitRuntimePrepassGenerate(
            RuntimeGenerateSession& session, RuntimeTemporalPairIdentity pair);
        [[nodiscard]] RuntimeRetirementStatus tryRetireRuntimePrepassGenerate(
            RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowGenerateSnapshot retireRuntimePrepassGenerate(
            RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowGenerateSnapshot inspectRuntimePrepassGenerate(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] std::optional<RuntimeGenerateDiagnosticPending>
            takeRuntimePrepassGeneratePending(RuntimeGenerateSession& session);
        void recordRuntimeBReturnSubmitted(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void recordRuntimeBReturnRejected(RuntimeGenerateSession& session,
            RuntimeGenerationId generation, bool deviceLost);
        void retireRuntimeBReturnWait(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void failRuntimeBReturnRetirement(RuntimeGenerateSession& session,
            RuntimeGenerationId generation, bool deviceLost);
        void releaseRuntimeGenerationReady(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void recordRuntimeAReturnSubmitted(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        [[nodiscard]] RuntimeShadowSplitResourceSnapshot inspectShadowSplitResources(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] RuntimeShadowIngestSnapshot submitShadowTemporalIngest(
            RuntimeGenerateSession& session, VkImage transportImage,
            vk::SyncFdPayload payload, TemporalSourceSlot destinationSlot,
            uint64_t frameId);
        [[nodiscard]] RuntimeShadowIngestSnapshot retireShadowTemporalIngest(
            RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowIngestSnapshot inspectShadowTemporalIngest(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] RuntimeShadowGenerateSnapshot submitShadowPrepassGenerate(
            RuntimeGenerateSession& session,
            RuntimeTemporalPairIdentity pair);
        [[nodiscard]] RuntimeShadowGenerateSnapshot retireShadowPrepassGenerate(
            RuntimeGenerateSession& session);
        [[nodiscard]] RuntimeShadowGenerateSnapshot inspectShadowPrepassGenerate(
            const RuntimeGenerateSession& session) const noexcept;
        [[nodiscard]] std::optional<RuntimeGenerateDiagnosticPending>
            takeShadowPrepassGeneratePending(RuntimeGenerateSession& session);
        void recordShadowBReturnSubmitted(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void recordShadowBReturnRejected(RuntimeGenerateSession& session,
            RuntimeGenerationId generation, bool deviceLost);
        void retireShadowBReturnWait(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void failShadowBReturnRetirement(RuntimeGenerateSession& session,
            RuntimeGenerationId generation, bool deviceLost);
        void releaseShadowGenerationReady(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
        void recordShadowAReturnSubmitted(RuntimeGenerateSession& session,
            RuntimeGenerationId generation);
#endif

        ///
        /// Open a frame generation context.
        ///
        /// The VkFormat of the exchanged images is inferred from whether hdr is true or false:
        /// - false: VK_FORMAT_R8G8B8A8_UNORM
        /// - true: VK_FORMAT_R16G16B16A16_SFLOAT
        ///
        /// The application and library must keep track of the frame index. When the next frame
        /// is ready, signal the syncFd with one increment (with the first trigger being 1).
        /// Each generated frame will increment the semaphore by one:
        /// - Application signals 1 -> Start generating with (curr, next) source images
        /// - Library signals 1 -> First frame between (curr, next) is ready
        /// - Library signals N -> N-th frame between (curr, next) is ready
        /// - Application signals N+1 -> Start generating with (next, curr) source images
        ///
        /// @param sourceImages Pair of exported source images; ownership is consumed.
        /// @param destImages Vector of exported output images; ownership is consumed.
        /// @param syncFd File descriptor for the timeline semaphore used for synchronization.
        /// @param flow Motion flow factor.
        /// @param perf Whether to enable performance mode.
        ///
        /// @throws backend::error on failure
        ///
        Context& openContext(
            std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
            std::vector<vk::ExternalImage> destImages,
            int syncFd,
            float flow, bool perf
        );

        ///
        /// Schedule a new set of generated frames.
        ///
        /// @param context Context to use.
        /// @throws backend::error on failure
        ///
        void scheduleFrames(Context& context);

        ///
        /// Schedule zero or more generated frames using explicit interpolation timestamps.
        ///
        /// The context still processes the real source frame when timestamps is empty,
        /// keeping temporal history and synchronization in step with the application.
        ///
        /// @param context Context to use.
        /// @param timestamps Strictly increasing normalized timestamps in the range (0, 1).
        /// @throws backend::error on failure.
        ///
        void scheduleFrames(
            Context& context,
            const std::vector<float>& timestamps
        );

        ///
        /// Close a frame generation context
        ///
        /// @param context Context to close.
        ///
        void closeContext(const Context& context);

        // Non-copyable and non-movable
        Instance(const Instance&) = delete;
        Instance& operator=(const Instance&) = delete;
        Instance(Instance&&) = delete;
        Instance& operator=(Instance&&) = delete;
        virtual ~Instance();
    private:
        std::unique_ptr<InstanceImpl> m_impl;

        std::vector<std::unique_ptr<Context>> m_contexts;
        std::vector<std::unique_ptr<RuntimePrepassSession>> m_runtimePrepassSessions;
        std::vector<std::unique_ptr<RuntimeGenerateSession>>
            m_runtimeGenerateDiagnosticSessions;
    };

    ///
    /// Make all lsfg-vk instances leaking.
    /// This is to workaround a bug in the Vulkan loader, which
    /// makes it impossible to destroy Vulkan instances and devices.
    ///
    void makeLeaking();

}
