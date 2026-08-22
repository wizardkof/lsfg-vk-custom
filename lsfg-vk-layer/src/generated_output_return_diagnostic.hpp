#pragma once

#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/vulkan_native_external_image_backing.hpp"

#include <optional>
#include <vector>

namespace lsfgvk::layer {
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

    class GeneratedOutputReturnDiagnosticSession {
    public:
        GeneratedOutputReturnDiagnosticSession(
            backend::RuntimeGenerateDiagnosticResult&&,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly);
        GeneratedOutputReturnDiagnosticSession(
            backend::RuntimeGenerateDiagnosticPending&&,
            backend::Instance&, backend::RuntimeGenerateDiagnosticSession&,
            const vk::RuntimeDevicePair&, vk::RuntimeExchangeEndpoint generationEndpoint,
            vk::RuntimeExchangeEndpoint renderEndpoint, bool captureOnly);
        GeneratedOutputReturnDiagnosticSession(const GeneratedOutputReturnDiagnosticSession&) = delete;
        GeneratedOutputReturnDiagnosticSession& operator=(const GeneratedOutputReturnDiagnosticSession&) = delete;
        ~GeneratedOutputReturnDiagnosticSession();
        [[nodiscard]] GeneratedOutputReturnState state() const noexcept { return states.state(); }
        [[nodiscard]] bool passed() const noexcept { return states.markerReady(); }
        [[nodiscard]] bool gpuChainedPassed() const noexcept { return chainedStates.markerReady(); }
    private:
        void advance(GeneratedOutputReturnState expected, GeneratedOutputReturnState next);
        void execute(backend::RuntimeGenerateDiagnosticResult&&);
        void executeGpuChained(backend::RuntimeGenerateDiagnosticPending&&,
            backend::Instance&, backend::RuntimeGenerateDiagnosticSession&);
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
    };
}
