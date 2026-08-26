#pragma once

#include "d3b2_insertion_path.hpp"
#include "lsfg-vk-backend/runtime_operation_authority.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <array>
#include <mutex>
#include <stdexcept>
#include <string>

namespace lsfgvk::layer {

enum class PrePresentGateResult : uint8_t { READY, TIMEOUT, FAILED, DEVICE_LOST };

struct CurrentOriginalFrame {
    VkImage image{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t queueFamily{VK_QUEUE_FAMILY_IGNORED};
    uint32_t virtualImageIndex{};
    uint64_t frameId{};
    uintptr_t authorityScope{};
};

enum class ApplicationPresentWaitState : uint8_t { AVAILABLE, BRIDGED, FAILED };

struct ApplicationPresentWaitBridgeResult;

class ApplicationPresentWaitAuthority {
public:
    ApplicationPresentWaitAuthority(std::vector<VkSemaphore> waits,
        uint64_t epoch, uintptr_t scope);
    ApplicationPresentWaitAuthority(const ApplicationPresentWaitAuthority&) = delete;
    ApplicationPresentWaitAuthority& operator=(const ApplicationPresentWaitAuthority&) = delete;
    ApplicationPresentWaitAuthority(ApplicationPresentWaitAuthority&&) noexcept;
    ApplicationPresentWaitAuthority& operator=(ApplicationPresentWaitAuthority&&) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const std::vector<VkSemaphore>& waits() const noexcept { return values; }
    [[nodiscard]] uint64_t epoch() const noexcept { return epochValue; }
    [[nodiscard]] uintptr_t scope() const noexcept { return scopeValue; }
    [[nodiscard]] ApplicationPresentWaitState state() const noexcept { return current; }
private:
    friend class D3B3OriginalReadyAuthority;
    friend struct ApplicationPresentWaitBridgeResult;
    friend ApplicationPresentWaitBridgeResult bridgeApplicationPresentWaits(
        ApplicationPresentWaitAuthority&, VkQueue, const std::vector<VkSemaphore>&,
        VkSemaphore, const std::function<VkResult(VkQueue, const VkSubmitInfo&)>&);
    void bridgeAccepted();
    std::vector<VkSemaphore> values;
    uint64_t epochValue{};
    uintptr_t scopeValue{};
    ApplicationPresentWaitState current{ApplicationPresentWaitState::AVAILABLE};
};

class D3B3OriginalReadyAuthority {
public:
    D3B3OriginalReadyAuthority() noexcept = default;
    D3B3OriginalReadyAuthority(const D3B3OriginalReadyAuthority&) = delete;
    D3B3OriginalReadyAuthority& operator=(const D3B3OriginalReadyAuthority&) = delete;
    D3B3OriginalReadyAuthority(D3B3OriginalReadyAuthority&&) noexcept;
    D3B3OriginalReadyAuthority& operator=(D3B3OriginalReadyAuthority&&) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] VkSemaphore semaphore() const noexcept { return semaphoreValue; }
    [[nodiscard]] uint64_t epoch() const noexcept { return epochValue; }
    [[nodiscard]] uintptr_t scope() const noexcept { return scopeValue; }
    void waitSubmitted();
    void waitRetired();
    [[nodiscard]] backend::RuntimeBinarySemaphoreState state() const noexcept {
        return readiness.state();
    }
private:
    friend struct ApplicationPresentWaitBridgeResult;
    friend ApplicationPresentWaitBridgeResult bridgeApplicationPresentWaits(
        ApplicationPresentWaitAuthority&, VkQueue, const std::vector<VkSemaphore>&,
        VkSemaphore, const std::function<VkResult(VkQueue, const VkSubmitInfo&)>&);
    D3B3OriginalReadyAuthority(VkSemaphore, uint64_t, uintptr_t);
    VkSemaphore semaphoreValue{};
    uint64_t epochValue{};
    uintptr_t scopeValue{};
    backend::RuntimeBinarySemaphoreEpoch readiness;
};

struct ApplicationPresentWaitBridgeResult {
    VkResult result{VK_ERROR_UNKNOWN};
    std::optional<D3B3OriginalReadyAuthority> originalReady;
};

[[nodiscard]] ApplicationPresentWaitBridgeResult bridgeApplicationPresentWaits(
    ApplicationPresentWaitAuthority&, VkQueue,
    const std::vector<VkSemaphore>& signalSemaphores,
    VkSemaphore originalReadyTarget,
    const std::function<VkResult(VkQueue, const VkSubmitInfo&)>& queueSubmit);

enum class D3B3OriginalSourceState : uint8_t {
    AVAILABLE, TERMINAL_SUBMITTED, TERMINAL_RETIRED, RELEASED, FAILED
};

// A move-only lease for the intercepted original C image.  It does not
// duplicate or destroy the VkImage; its anchor is the existing swapchain
// lifetime, while the state records the terminal read that still protects it.
class D3B3OriginalSourceAuthority {
public:
    D3B3OriginalSourceAuthority() noexcept = default;
    D3B3OriginalSourceAuthority(const D3B3OriginalSourceAuthority&) = delete;
    D3B3OriginalSourceAuthority& operator=(const D3B3OriginalSourceAuthority&) = delete;
    D3B3OriginalSourceAuthority(D3B3OriginalSourceAuthority&&) noexcept = default;
    D3B3OriginalSourceAuthority& operator=(D3B3OriginalSourceAuthority&&) noexcept = default;
    D3B3OriginalSourceAuthority(CurrentOriginalFrame frame,
        std::shared_ptr<const uint8_t> lifetime) :
        value(frame), lifetime(std::move(lifetime)) {
        if (!valid() || value.frameId == 0)
            throw std::invalid_argument("invalid D3B3 original source authority");
    }
    [[nodiscard]] bool valid() const noexcept {
        return value.image != VK_NULL_HANDLE && value.frameId != 0 && lifetime
            && current != D3B3OriginalSourceState::RELEASED
            && current != D3B3OriginalSourceState::FAILED;
    }
    [[nodiscard]] const CurrentOriginalFrame& source() const noexcept { return value; }
    [[nodiscard]] D3B3OriginalSourceState state() const noexcept { return current; }
    void terminalSubmitted() {
        require(D3B3OriginalSourceState::AVAILABLE);
        current = D3B3OriginalSourceState::TERMINAL_SUBMITTED;
    }
    void terminalRetired() {
        require(D3B3OriginalSourceState::TERMINAL_SUBMITTED);
        current = D3B3OriginalSourceState::TERMINAL_RETIRED;
    }
    void release() {
        if (current != D3B3OriginalSourceState::TERMINAL_RETIRED)
            throw std::logic_error("original C was released before terminal read retirement");
        current = D3B3OriginalSourceState::RELEASED;
        lifetime.reset();
    }
    void fail() noexcept { current = D3B3OriginalSourceState::FAILED; }
private:
    void require(D3B3OriginalSourceState expected) const {
        if (current != expected)
            throw std::logic_error("invalid D3B3 original source transition");
    }
    CurrentOriginalFrame value{};
    std::shared_ptr<const uint8_t> lifetime;
    D3B3OriginalSourceState current{D3B3OriginalSourceState::AVAILABLE};
};

struct TemporalSourceRef {
    VkImage image{};
    size_t slot{};
    uint64_t frameId{};
    uint64_t generation{};
    bool valid{};
};

struct RuntimeFrameIngestResult {
    TemporalSourceRef source;
    VkSemaphore originalReady{};
    VkSemaphore transportReady{};
    VkFence completionFence{};
    bool temporaryPayloadRetained{};
};

struct GeneratedPairIdentity {
    uint64_t olderFrameId{};
    uint64_t newerFrameId{};
    uint64_t generationId{};
};

struct ReturnedGeneratedFrame {
    VkImage image{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t sourceQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    uint32_t destinationQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    VkSemaphore returnedForGraphics{};
    VkFence completionFence{};
    GeneratedPairIdentity identity{};
    std::shared_ptr<const uint8_t> lifetimeAnchor;
};

struct D3B3IngestDispatch {
    std::function<RuntimeFrameIngestResult(const CurrentOriginalFrame&, size_t)> execute;
};

struct D3B3GenerateDispatch {
    std::function<VkResult(const TemporalSourceRef&, const TemporalSourceRef&,
        uint64_t, float)> execute;
};

struct D3B3ReturnDispatch {
    std::function<std::optional<ReturnedGeneratedFrame>(GeneratedPairIdentity)> execute;
};

[[nodiscard]] RuntimeFrameIngestResult ingestCurrentFrame(
    const D3B3IngestDispatch&, const CurrentOriginalFrame&, size_t targetSlot);

[[nodiscard]] GeneratedPairIdentity generateTemporalPair(
    const D3B3GenerateDispatch&, const TemporalSourceRef&, const TemporalSourceRef&,
    uint64_t generationId, float interpolation = 0.5F);

[[nodiscard]] ReturnedGeneratedFrame returnGeneratedFrame(
    const D3B3ReturnDispatch&, GeneratedPairIdentity);

[[nodiscard]] VkResult executeReusableTerminal(D3B2InsertionPath&, bool reusableSlot);

enum class D3B3RetirementStatus : uint8_t { RETIRED, TIMEOUT, DEVICE_LOST, FAILURE };

struct D3B3RetirementDomains {
    bool inputFence{};
    bool bReturnFence{};
    bool aReturnComplete{};
    bool graphicsFence{};
    bool generatedPresentFence{};
    bool originalPresentFence{};
    bool temporaryPayloadRetired{};
    bool hiddenAcquisitionLeasesReleased{};
    bool generatedAuthorityRetired{};
};

enum class D3B3PairState : uint8_t {
    NON_TERMINAL_READY, TERMINAL_PREFLIGHTED, DESTINATIONS_ACQUIRED,
    TERMINAL_SUBMIT_PENDING, TERMINAL_SUBMITTED, GRAPHICS_RETIRED,
    PRESENTS_SUBMITTED, PRESENTS_RETIRED, PAIR_RETIRED, FAILED
};

struct D3B3PairTerminalDispatch {
    std::optional<D3B3OriginalReadyAuthority> originalReadyAuthority;
    std::function<D3B2InsertionPath(
        D3B2Source generated, D3B2Source original, VkSemaphore returnedForGraphics)> build;
    std::function<void()> retireGeneratedPresentFence;
    std::function<void()> retireOriginalPresentFence;
    std::function<D3B2RetirementResult()> tryRetireGeneratedPresentFence;
    std::function<D3B2RetirementResult()> tryRetireOriginalPresentFence;
};

class D3B3PairOperation {
public:
    D3B3PairOperation() noexcept = default;
    D3B3PairOperation(backend::ReturnedGeneratedOperation&& returned,
        D3B3OriginalSourceAuthority&& original,
        D3B3PairTerminalDispatch dispatch);
    D3B3PairOperation(const D3B3PairOperation&) = delete;
    D3B3PairOperation& operator=(const D3B3PairOperation&) = delete;
    D3B3PairOperation(D3B3PairOperation&&) noexcept;
    D3B3PairOperation& operator=(D3B3PairOperation&&) noexcept;
    ~D3B3PairOperation() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] D3B3PairState state() const noexcept { return current; }
    [[nodiscard]] backend::RuntimeTemporalPairIdentity identity() const noexcept {
        return returned.identity();
    }
    [[nodiscard]] const backend::ReturnedGeneratedOperation& returnedOperation() const noexcept {
        return returned;
    }
    [[nodiscard]] D3B3OriginalSourceState originalState() const noexcept {
        return original.state();
    }
    [[nodiscard]] backend::RuntimeBinarySemaphoreState returnedForGraphicsState() const noexcept {
        return returnedForGraphics.state();
    }
    [[nodiscard]] backend::RuntimeBinarySemaphoreState originalReadyState() const noexcept {
        return originalReady.state();
    }
    [[nodiscard]] VkSemaphore originalReadySemaphore() const noexcept {
        return originalReady.semaphore();
    }
    [[nodiscard]] bool returnedForGraphicsReusable() const noexcept {
        return returnedForGraphics.state()
            == backend::RuntimeBinarySemaphoreState::WAIT_RETIRED;
    }
    [[nodiscard]] bool aReturnRetired() const noexcept {
        return returned.aReturnRetired();
    }
    [[nodiscard]] backend::RuntimeAuthorityState aReturnState() const noexcept {
        return returned.aReturnAuthority().state();
    }
    [[nodiscard]] backend::RuntimeTemporaryPayloadState aReturnPayloadState() const noexcept {
        return returned.payloadAuthority().state();
    }
    [[nodiscard]] bool returnedImageAlive() const noexcept {
        return returned.valid() && returned.aReturnPending().imageView().valid();
    }
    [[nodiscard]] bool terminalCommandBufferReusable() const noexcept {
        return current == D3B3PairState::GRAPHICS_RETIRED
            || current == D3B3PairState::PRESENTS_SUBMITTED
            || current == D3B3PairState::PRESENTS_RETIRED
            || current == D3B3PairState::PAIR_RETIRED;
    }
    // True means neither hidden image remains an application-owned
    // acquisition lease. It does not claim that either index is reacquirable.
    [[nodiscard]] bool hiddenAcquisitionLeasesReleased() const noexcept {
        return hiddenAcquisitionLeasesReleasedValue;
    }
    [[nodiscard]] bool generatedPresentRetired() const noexcept { return generatedPresentRetiredValue; }
    [[nodiscard]] bool originalPresentRetired() const noexcept { return originalPresentRetiredValue; }
    [[nodiscard]] bool pairRetired() const noexcept {
        return current == D3B3PairState::PAIR_RETIRED;
    }

    void preflight();
    void submitTerminal();
    [[nodiscard]] D3B2RetirementResult tryRetireTerminal();
    [[nodiscard]] bool tryRetireAReturn();
    void retirePresentWaits();
    [[nodiscard]] D3B2RetirementResult tryRetirePresentWaits();
    void retirePair();
    void rejectReturnedForGraphicsReuse() const;

private:
    void bindCallbacks();
    void fail() noexcept;

    backend::ReturnedGeneratedOperation returned;
    D3B3OriginalSourceAuthority original;
    D3B3PairTerminalDispatch dispatch;
    backend::RuntimeBinarySemaphoreEpoch returnedForGraphics;
    D3B3OriginalReadyAuthority originalReady;
    std::optional<D3B2InsertionPath> terminalPath;
    std::optional<D3B2PendingInsertion> terminalPending;
    D3B2InsertionState terminalState{D3B2InsertionState::INACTIVE};
    D3B3PairState current{D3B3PairState::NON_TERMINAL_READY};
    bool generatedPresentSubmitted{};
    bool originalPresentSubmitted{};
    bool generatedPresentRetiredValue{};
    bool originalPresentRetiredValue{};
    bool generatedAcquisitionLeaseReleased{};
    bool originalAcquisitionLeaseReleased{};
    bool hiddenAcquisitionLeasesReleasedValue{true};
};

[[nodiscard]] D3B3RetirementStatus evaluateD3B3Retirement(
    const D3B3RetirementDomains&) noexcept;

enum class D3B3ShutdownPolicy : uint8_t { IMMEDIATE_ONE_SHOT, DEFER_UNTIL_FINITE_STOP };
[[nodiscard]] constexpr bool shouldStopWorker(
        D3B3ShutdownPolicy policy, bool finiteStopped) noexcept {
    return policy == D3B3ShutdownPolicy::IMMEDIATE_ONE_SHOT || finiteStopped;
}

enum class D3B3PendingState : uint8_t {
    EMPTY, ACTIVE, PENDING_RETIREMENT, RETIRING, RETIRED, FAILED
};

enum class D3B3TemporalState : uint8_t {
    EMPTY, CURRENT, RETAINED_HISTORY, READING_GENERATION, OVERWRITEABLE
};

struct D3B3TemporalProductionSlot {
    D3B3TemporalState state{D3B3TemporalState::EMPTY};
    uint64_t frameId{};
    uint64_t generation{};
};

struct D3B3PendingProductionOperation {
    D3B3PendingState state{D3B3PendingState::EMPTY};
    uint64_t olderFrameId{};
    uint64_t newerFrameId{};
    uint64_t generationId{};
    size_t olderSlot{};
    size_t newerSlot{};
    D3B3RetirementDomains domains{};
};

struct D3B3TerminalProductionSlot {
    VkCommandBuffer commandBuffer{};
    VkFence graphicsFence{};
    VkSemaphore generatedReady{};
    VkSemaphore originalReady{};
    VkFence generatedPresentFence{};
    VkFence originalPresentFence{};
    bool hiddenAcquisitionLeasesReleased{true};
};

enum class D3B3FiniteProductionState : uint8_t {
    WARMUP_EMPTY, HISTORY_A_READY, HISTORY_AB_READY, WAITING_REUSE,
    READY_NEXT, FINITE_STOPPED, FAILED
};

struct D3B3FiniteProductionCounters {
    uint32_t applicationFrames{};
    uint32_t warmupOriginals{};
    uint32_t generate{};
    uint32_t bReturns{};
    uint32_t aReturns{};
    uint32_t pairOperations{};
    uint32_t terminalSubmits{};
    uint32_t generatedPresents{};
    uint32_t originalPresents{};
    uint32_t internalPresents{};
    uint32_t gateTimeouts{};
    uint32_t markers{};
};

// Test/internal orchestration callbacks. The normal Swapchain route does not
// bind these callbacks; an explicit test dispatch lets this persistent state
// own and retire the real move-only PairOperation.
struct D3B3FiniteProductionOperations {
    std::function<bool(uint64_t, backend::TemporalSourceSlot, bool)> ingest;
    std::function<D3B3RetirementStatus()> retireWarmupIngest;
    std::function<bool(uint64_t, uint64_t, backend::TemporalSourceSlot,
        backend::TemporalSourceSlot, uint64_t)> generate;
    std::function<bool(uint64_t)> presentWarmupOriginal;
    std::function<backend::ReturnedGeneratedOperation(
        const backend::RuntimeTemporalPairIdentity&)> returnGenerated;
    std::function<D3B3OriginalSourceAuthority(
        const backend::RuntimeTemporalPairIdentity&)> makeOriginal;
    std::function<D3B3PairTerminalDispatch(
        const backend::RuntimeTemporalPairIdentity&)> makeTerminal;
    std::function<bool()> stopWorkerAndJoin;
    std::function<void(const std::string&)> record;
    std::function<void()> emitMarker;
};

class D3B3ProductionState {
public:
    using Retirement = std::function<D3B3RetirementStatus(
        const D3B3PendingProductionOperation&)>;

    explicit D3B3ProductionState(Retirement retirement = {});

    [[nodiscard]] PrePresentGateResult prePresentGate();
    bool beginPending(uint64_t olderFrameId, uint64_t newerFrameId,
        uint64_t generationId, size_t olderSlot, size_t newerSlot);
    bool updatePendingDomains(D3B3RetirementDomains domains);
    bool markPending(D3B3PendingProductionOperation operation);
    void setFailure() noexcept;
    void setEligibleFrameCount(uint32_t count) noexcept;
    void setFiniteStopped(bool stopped) noexcept;
    void configureFinite(D3B3FiniteProductionOperations operations);
    bool presentFinite(uint64_t frameId);
    bool finishFinite();
    [[nodiscard]] bool finiteStopped() const noexcept;
    [[nodiscard]] D3B3FiniteProductionState finiteState() const noexcept;
    [[nodiscard]] D3B3FiniteProductionCounters finiteCounters() const noexcept;
    [[nodiscard]] bool acceptsPendingEpoch(uint64_t generationId) const noexcept;
    [[nodiscard]] D3B3PendingProductionOperation pending() const;
    [[nodiscard]] std::array<D3B3TemporalProductionSlot, 2> temporalSlots() const;
    [[nodiscard]] uint32_t eligibleFrameCount() const noexcept;
    [[nodiscard]] uint32_t pairCount() const noexcept;
    [[nodiscard]] bool hasTerminalSlot() const noexcept { return true; }
    [[nodiscard]] const D3B3TerminalProductionSlot& terminalSlot() const noexcept { return terminal; }
    void recordPair(uint64_t olderFrameId, uint64_t newerFrameId);
    void recordPair(const backend::RuntimeTemporalPairIdentity& pair);

private:
    [[nodiscard]] D3B3RetirementStatus retireFinitePair();
    [[nodiscard]] bool finiteOperationsComplete() const noexcept;

    mutable std::mutex mutex;
    Retirement retirement;
    D3B3FiniteProductionOperations finiteOperations;
    D3B3PendingProductionOperation operation{};
    std::array<D3B3TemporalProductionSlot, 2> slots{};
    uint32_t eligibleFrames{};
    uint32_t pairs{};
    bool finiteStop{};
    D3B3TerminalProductionSlot terminal{};
    D3B3FiniteProductionState finiteCurrent{D3B3FiniteProductionState::WARMUP_EMPTY};
    D3B3FiniteProductionCounters finiteTotals{};
    std::optional<D3B3PairOperation> finitePair;
    uint64_t lastFiniteFrame{};
    uint64_t finiteGeneration{};
};

}
