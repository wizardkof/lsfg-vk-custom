#pragma once

#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-backend/runtime_operation_authority.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/vulkan_native_external_image_backing.hpp"

#include <optional>
#include <memory>
#include <vector>

namespace lsfgvk::layer {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    struct ShadowReturnExecutionForTesting final {};
#endif
    struct ProductionReturnExecution final {};

    enum class ReturnSubmissionFencePolicy : uint8_t {
        ACTIVE_COMPATIBLE,
        SHADOW_REAL
    };

    [[nodiscard]] constexpr VkFence returnSubmissionFence(
            ReturnSubmissionFencePolicy policy, VkFence shadowFence) noexcept {
        return policy == ReturnSubmissionFencePolicy::SHADOW_REAL
            ? shadowFence : VK_NULL_HANDLE;
    }

    enum class GeneratedOutputReturnState {
        EMPTY, D2_VALIDATED, TOKEN_CONSUMED, B_BACKING_READY, B_COPY_SUBMITTED,
        B_SYNC_EXPORTED, A_IMPORTED, A_COPY_SUBMITTED, A_VALIDATED, PASS, FAILED
    };

    class GeneratedOutputReturnStateMachine {
    public:
        void advance(GeneratedOutputReturnState expected, GeneratedOutputReturnState next);
        void fail() noexcept { current = GeneratedOutputReturnState::FAILED; }
        [[nodiscard]] GeneratedOutputReturnState state() const noexcept { return current; }
        [[nodiscard]] bool markerReady() const noexcept {
            return current == GeneratedOutputReturnState::PASS;
        }
    private:
        GeneratedOutputReturnState current{GeneratedOutputReturnState::EMPTY};
    };

    struct GeneratedOutputIntegrity {
        backend::RuntimeGenerationId generation{};
        VkExtent2D extent{};
        VkFormat format{};
        size_t byteCount{};
        size_t nonzeroByteCount{};
        uint64_t checksum{};
    };

    // Move-only boundary between the existing B-return submission/export and
    // the existing A-side import/handoff phase.  C1A exposes the boundary;
    // C1B will be the first caller allowed to stop at it.
    class RuntimeGeneratedBReturnPending {
    public:
        RuntimeGeneratedBReturnPending() noexcept = default;
        RuntimeGeneratedBReturnPending(const RuntimeGeneratedBReturnPending&) = delete;
        RuntimeGeneratedBReturnPending& operator=(const RuntimeGeneratedBReturnPending&) = delete;
        RuntimeGeneratedBReturnPending(RuntimeGeneratedBReturnPending&&) noexcept = default;
        RuntimeGeneratedBReturnPending& operator=(RuntimeGeneratedBReturnPending&&) noexcept = default;
        [[nodiscard]] bool valid() const noexcept { return bToAPayload.valid() && pending.valid(); }
        [[nodiscard]] backend::RuntimeTemporalPairIdentity identity() const noexcept {
            return pending.temporalPair();
        }
        [[nodiscard]] const vk::SyncFdPayload& payload() const noexcept { return bToAPayload; }
        [[nodiscard]] const backend::RuntimeSubmissionRetirement& bReturnAuthority() const noexcept {
            return bReturn;
        }
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        [[nodiscard]] bool acceptedSubmissionAuthorityForTesting() const noexcept {
            return pending.valid() && bReturn.valid() && fenceOwner && returnResources;
        }
        [[nodiscard]] VkFence bReturnFenceForTesting() const noexcept {
            return bReturn.fenceHandle();
        }
        [[nodiscard]] VkSemaphore generationReadyForTesting() const noexcept {
            return pending.readinessSemaphore();
        }
        [[nodiscard]] bool payloadOwnedForTesting() const noexcept {
            return bToAPayload.valid();
        }
#endif
    private:
        friend class GeneratedOutputReturnSession;
        RuntimeGeneratedBReturnPending(backend::RuntimeGenerateDiagnosticPending&& value,
                vk::SyncFdPayload&& exported,
                backend::RuntimeSubmissionRetirement&& authority = {},
                std::shared_ptr<VkFence> fence = {},
                std::shared_ptr<void> resources = {}) :
            pending(std::move(value)), bToAPayload(std::move(exported)),
            bReturn(std::move(authority)), fenceOwner(std::move(fence)),
            returnResources(std::move(resources)) {}
        backend::RuntimeGenerateDiagnosticPending pending;
        vk::SyncFdPayload bToAPayload;
        backend::RuntimeSubmissionRetirement bReturn;
        std::shared_ptr<VkFence> fenceOwner;
        std::shared_ptr<void> returnResources;
    };

    enum class GeneratedOutputCleanupState { NO_SUBMIT, B_SUBMITTED, A_SUBMITTED, A_COMPLETED };
    [[nodiscard]] constexpr bool generatedOutputNeedsDeviceIdle(
            GeneratedOutputCleanupState state) noexcept {
        return state == GeneratedOutputCleanupState::B_SUBMITTED
            || state == GeneratedOutputCleanupState::A_SUBMITTED;
    }

    [[nodiscard]] constexpr bool generatedOutputRoleBindingEligible(
            const vk::RuntimeDevicePair& pair,
            const vk::PhysicalDeviceIdentity& generation,
            const vk::PhysicalDeviceIdentity& render,
            bool captureOnly) noexcept {
        return pair.crossDevice() && captureOnly
            && pair.matchesGenerationDevice(generation)
            && pair.matchesRenderDevice(render)
            && !generation.samePhysicalDevice(render);
    }

    [[nodiscard]] bool generatedOutputIntegrityMatches(
        const GeneratedOutputIntegrity&, const GeneratedOutputIntegrity&) noexcept;

    enum class GpuChainedReturnState {
        EMPTY, D2_SUBMITTED, PENDING_GENERATION, TRANSPORT_CONSUMED,
        RETURN_B_PREPARED, RETURN_B_SUBMITTED, A_SUBMITTED, A_COMPLETED,
        B_VALIDATED, A_VALIDATED, PASS, FAILED
    };

    class GpuChainedReturnStateMachine {
    public:
        void advance(GpuChainedReturnState expected, GpuChainedReturnState next);
        void fail() noexcept { current = GpuChainedReturnState::FAILED; }
        [[nodiscard]] GpuChainedReturnState state() const noexcept { return current; }
        [[nodiscard]] bool markerReady() const noexcept {
            return current == GpuChainedReturnState::PASS;
        }
    private:
        GpuChainedReturnState current{GpuChainedReturnState::EMPTY};
    };

    struct GpuChainedMarkerGate {
        bool readinessDependencyUsed{};
        bool generateReturnHostWaitUsed{};
        bool bToAHostWaitUsed{};
        bool bValidated{};
        bool aValidated{};
        bool integrityExact{};
        bool synchronousD3A1Path{};
        [[nodiscard]] bool ready() const noexcept {
            return readinessDependencyUsed && !generateReturnHostWaitUsed && !bToAHostWaitUsed
                && bValidated && aValidated && integrityExact && !synchronousD3A1Path;
        }
    };

    class GeneratedOutputReturnSession {
    public:
        GeneratedOutputReturnSession(
            backend::RuntimeGenerateDiagnosticResult&&,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly);
        GeneratedOutputReturnSession(
            backend::RuntimeGenerateDiagnosticPending&&,
            backend::Instance&, backend::RuntimeGenerateSession&,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly,
            std::optional<vk::RuntimeForeignImageHandoffInfo> handoff = std::nullopt);
        GeneratedOutputReturnSession(ProductionReturnExecution,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        GeneratedOutputReturnSession(ShadowReturnExecutionForTesting,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly);
#endif
        GeneratedOutputReturnSession(const GeneratedOutputReturnSession&) = delete;
        GeneratedOutputReturnSession& operator=(const GeneratedOutputReturnSession&) = delete;
        ~GeneratedOutputReturnSession();
        [[nodiscard]] GeneratedOutputReturnState state() const noexcept { return states.state(); }
        [[nodiscard]] bool passed() const noexcept { return states.markerReady(); }
        [[nodiscard]] bool gpuChainedPassed() const noexcept { return chainedStates.markerReady(); }
        [[nodiscard]] bool presentationHandoffPending() const noexcept {
            return aReadbackPending.valid();
        }
        [[nodiscard]] vk::RuntimeForeignImageView returnedImageView() const noexcept {
            return aReadbackPending.imageView();
        }
        void completePresentationDiagnostics();
        [[nodiscard]] RuntimeGeneratedBReturnPending submitProductionBReturn(
            backend::RuntimeGenerateDiagnosticPending&&, backend::Instance&,
            backend::RuntimeGenerateSession&);
        [[nodiscard]] backend::ReturnedGeneratedOperation
            completeProductionGeneratedReturnOnA(
                RuntimeGeneratedBReturnPending&&, backend::Instance&,
                backend::RuntimeGenerateSession&,
                vk::RuntimeForeignImageHandoffInfo);
        [[nodiscard]] backend::RuntimeRetirementStatus
            tryRetireProductionBReturn(RuntimeGeneratedBReturnPending&,
                backend::Instance&, backend::RuntimeGenerateSession&);
        [[nodiscard]] backend::RuntimeRetirementStatus
            tryRetireProductionBReturn(backend::ReturnedGeneratedOperation&,
                backend::Instance&, backend::RuntimeGenerateSession&);
        [[nodiscard]] backend::RuntimeRetirementStatus
            releaseProductionGeneratedOutput(backend::ReturnedGeneratedOperation&,
                backend::Instance&, backend::RuntimeGenerateSession&);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        [[nodiscard]] RuntimeGeneratedBReturnPending submitShadowBReturnForTesting(
            backend::RuntimeGenerateDiagnosticPending&&, backend::Instance&,
            backend::RuntimeGenerateSession&);
        [[nodiscard]] backend::ReturnedGeneratedOperation
            completeShadowGeneratedReturnOnAForTesting(
                RuntimeGeneratedBReturnPending&&, backend::Instance&,
                backend::RuntimeGenerateSession&,
                vk::RuntimeForeignImageHandoffInfo);
        void retireShadowBReturnForTesting(backend::ReturnedGeneratedOperation&,
            backend::Instance&, backend::RuntimeGenerateSession&);
        void releaseShadowGeneratedOutputForTesting(backend::ReturnedGeneratedOperation&,
            backend::Instance&, backend::RuntimeGenerateSession&);
        void retireShadowAReturnForTesting(backend::ReturnedGeneratedOperation&);
        [[nodiscard]] std::optional<RuntimeGeneratedBReturnPending>
            takeAcceptedShadowFailureForTesting();
        void retireAcceptedShadowBReturnFailureForTesting(
            RuntimeGeneratedBReturnPending&, backend::Instance&,
            backend::RuntimeGenerateSession&);
        void releaseAcceptedShadowGeneratedOutputForTesting(
            RuntimeGeneratedBReturnPending&, backend::Instance&,
            backend::RuntimeGenerateSession&);
        [[nodiscard]] GpuChainedReturnState gpuChainedStateForTesting() const noexcept {
            return chainedStates.state();
        }
#endif
    private:
        void advance(GeneratedOutputReturnState expected, GeneratedOutputReturnState next);
        void execute(backend::RuntimeGenerateDiagnosticResult&&);
        void executeGpuChained(backend::RuntimeGenerateDiagnosticPending&&,
            backend::Instance&, backend::RuntimeGenerateSession&,
            std::optional<vk::RuntimeForeignImageHandoffInfo>);
        [[nodiscard]] RuntimeGeneratedBReturnPending submitGeneratedBReturn(
            backend::RuntimeGenerateDiagnosticPending&&,
            ReturnSubmissionFencePolicy, backend::Instance&,
            backend::RuntimeGenerateSession&, VkFence = VK_NULL_HANDLE);
        void completeGeneratedReturnOnA(RuntimeGeneratedBReturnPending&&,
            backend::Instance&, backend::RuntimeGenerateSession&,
            std::optional<vk::RuntimeForeignImageHandoffInfo>);
        [[nodiscard]] std::vector<uint8_t> transportGeneratedImage(
            VkImage sourceImage, VkExtent2D extent, VkFormat format,
            VkSemaphore generationReady, bool chained);
        void resetBCommands() noexcept;
        vk::RuntimeExchangeEndpoint generationEndpoint{};
        vk::RuntimeExchangeEndpoint renderEndpoint{};
        GeneratedOutputReturnStateMachine states;
        GpuChainedReturnStateMachine chainedStates;
        vk::VulkanNativeExternalImageBacking backing;
        vk::RuntimeImageEndpoint importedA;
        vk::SyncFdSemaphore signalB;
        VkCommandPool bCommandPool{};
        VkCommandBuffer bCommand{};
        GeneratedOutputIntegrity authoritative{};
        bool bSubmitted{};
        bool aCompleted{};
        vk::RuntimeForeignImageReadbackPending aReadbackPending;
        backend::RuntimeGenerateDiagnosticPending delayedGeneration;
        backend::Instance* delayedBackend{};
        backend::RuntimeGenerateSession* delayedBackendSession{};
        GeneratedOutputIntegrity expected{};
        std::optional<RuntimeGeneratedBReturnPending> acceptedShadowFailure;
        std::shared_ptr<const uint8_t> operationLifetime{std::make_shared<const uint8_t>(0)};
    };
}
