#pragma once

#include "lsfg-vk-backend/lsfgvk.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace lsfgvk::layer {
class GeneratedOutputReturnDiagnosticSession;
}

namespace lsfgvk::backend {

struct RuntimeOperationAuthorityTestAccess;
struct RuntimeShadowIngestAuthorityAccess;

enum class RuntimeAuthorityState : uint8_t { PREPARED, SUBMITTED, RETIRED, FAILED };

class RuntimeSubmissionRetirement {
public:
    RuntimeSubmissionRetirement() noexcept = default;
    RuntimeSubmissionRetirement(const RuntimeSubmissionRetirement&) = delete;
    RuntimeSubmissionRetirement& operator=(const RuntimeSubmissionRetirement&) = delete;
    RuntimeSubmissionRetirement(RuntimeSubmissionRetirement&&) noexcept = default;
    RuntimeSubmissionRetirement& operator=(RuntimeSubmissionRetirement&&) noexcept = default;
    [[nodiscard]] bool valid() const noexcept { return fence != VK_NULL_HANDLE && lifetime; }
    [[nodiscard]] uint64_t epoch() const noexcept { return operationEpoch; }
    [[nodiscard]] VkFence fenceHandle() const noexcept { return fence; }
    [[nodiscard]] RuntimeAuthorityState state() const noexcept { return current; }
    void submitted() {
        require(RuntimeAuthorityState::PREPARED);
        current = RuntimeAuthorityState::SUBMITTED;
    }
    void retire(uint64_t epoch) {
        requireEpoch(epoch); require(RuntimeAuthorityState::SUBMITTED);
        current = RuntimeAuthorityState::RETIRED;
    }
    void fail() noexcept { current = RuntimeAuthorityState::FAILED; }
    [[nodiscard]] bool canResetFence() const noexcept {
        return current == RuntimeAuthorityState::RETIRED;
    }
    [[nodiscard]] static RuntimeSubmissionRetirement submittedFence(
            VkFence value, uint64_t epoch, std::shared_ptr<const uint8_t> anchor) {
        RuntimeSubmissionRetirement result(value, epoch, std::move(anchor));
        result.submitted();
        return result;
    }
private:
    friend struct RuntimeOperationAuthorityTestAccess;
    friend struct RuntimeShadowIngestAuthorityAccess;
    RuntimeSubmissionRetirement(VkFence value, uint64_t epoch,
            std::shared_ptr<const uint8_t> anchor) :
        fence(value), operationEpoch(epoch), lifetime(std::move(anchor)) {
        if (!valid() || operationEpoch == 0) throw std::invalid_argument("invalid submission authority");
    }
    void require(RuntimeAuthorityState expected) const {
        if (current != expected) throw std::logic_error("invalid submission authority transition");
    }
    void requireEpoch(uint64_t epoch) const {
        if (epoch == 0 || epoch != operationEpoch) throw std::logic_error("stale submission epoch");
    }
    VkFence fence{};
    uint64_t operationEpoch{};
    std::shared_ptr<const uint8_t> lifetime;
    RuntimeAuthorityState current{RuntimeAuthorityState::PREPARED};
};

enum class RuntimeTemporaryPayloadState : uint8_t {
    IMPORTED, WAIT_SUBMITTED, WAIT_RETIRED, FAILED
};

class RuntimeTemporarySemaphorePayload {
public:
    RuntimeTemporarySemaphorePayload() noexcept = default;
    RuntimeTemporarySemaphorePayload(const RuntimeTemporarySemaphorePayload&) = delete;
    RuntimeTemporarySemaphorePayload& operator=(const RuntimeTemporarySemaphorePayload&) = delete;
    RuntimeTemporarySemaphorePayload(RuntimeTemporarySemaphorePayload&&) noexcept = default;
    RuntimeTemporarySemaphorePayload& operator=(RuntimeTemporarySemaphorePayload&&) noexcept = default;
    [[nodiscard]] bool valid() const noexcept { return semaphore != VK_NULL_HANDLE && lifetime; }
    [[nodiscard]] uint64_t epoch() const noexcept { return operationEpoch; }
    [[nodiscard]] VkSemaphore semaphoreHandle() const noexcept { return semaphore; }
    [[nodiscard]] RuntimeTemporaryPayloadState state() const noexcept { return current; }
    void waitSubmitted(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeTemporaryPayloadState::IMPORTED)
            throw std::logic_error("temporary payload wait already submitted");
        current = RuntimeTemporaryPayloadState::WAIT_SUBMITTED;
    }
    void waitRetired(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeTemporaryPayloadState::WAIT_SUBMITTED)
            throw std::logic_error("temporary payload wait not pending");
        current = RuntimeTemporaryPayloadState::WAIT_RETIRED;
    }
    void fail() noexcept { current = RuntimeTemporaryPayloadState::FAILED; }
    [[nodiscard]] bool canReimportPayload() const noexcept {
        return current == RuntimeTemporaryPayloadState::WAIT_RETIRED;
    }
    [[nodiscard]] static RuntimeTemporarySemaphorePayload submittedWait(
            VkSemaphore value, uint64_t epoch,
            std::shared_ptr<const uint8_t> anchor) {
        RuntimeTemporarySemaphorePayload result(value, epoch, std::move(anchor));
        result.waitSubmitted(epoch);
        return result;
    }
private:
    friend struct RuntimeOperationAuthorityTestAccess;
    friend struct RuntimeShadowIngestAuthorityAccess;
    RuntimeTemporarySemaphorePayload(VkSemaphore value, uint64_t epoch,
            std::shared_ptr<const uint8_t> anchor) :
        semaphore(value), operationEpoch(epoch), lifetime(std::move(anchor)) {
        if (!valid() || operationEpoch == 0) throw std::invalid_argument("invalid temporary payload authority");
    }
    void requireEpoch(uint64_t epoch) const {
        if (epoch == 0 || epoch != operationEpoch) throw std::logic_error("stale temporary payload epoch");
    }
    VkSemaphore semaphore{};
    uint64_t operationEpoch{};
    std::shared_ptr<const uint8_t> lifetime;
    RuntimeTemporaryPayloadState current{RuntimeTemporaryPayloadState::IMPORTED};
};

enum class RuntimeBinarySemaphoreState : uint8_t {
    AVAILABLE, SIGNAL_RESERVED, SIGNAL_SUBMITTED, WAIT_SUBMITTED, WAIT_RETIRED, FAILED
};

class RuntimeBinarySemaphoreEpoch {
public:
    void reserveSignal(uint64_t epoch) {
        if (current != RuntimeBinarySemaphoreState::AVAILABLE || epoch == 0 || epoch <= lastEpoch)
            throw std::logic_error("binary semaphore signal is not reusable");
        lastEpoch = epoch; current = RuntimeBinarySemaphoreState::SIGNAL_RESERVED;
    }
    void producerSubmitted(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeBinarySemaphoreState::SIGNAL_RESERVED)
            throw std::logic_error("binary semaphore signal was not reserved");
        current = RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED;
    }
    void signalSubmitted(uint64_t epoch) {
        reserveSignal(epoch);
        producerSubmitted(epoch);
    }
    void producerFailed() noexcept { current = RuntimeBinarySemaphoreState::FAILED; }
    void waitSubmitted(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED)
            throw std::logic_error("binary semaphore has no pending signal");
        current = RuntimeBinarySemaphoreState::WAIT_SUBMITTED;
    }
    void consumerFailed(uint64_t epoch) { requireEpoch(epoch); current = RuntimeBinarySemaphoreState::FAILED; }
    void waitRetired(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeBinarySemaphoreState::WAIT_SUBMITTED)
            throw std::logic_error("binary semaphore wait is not pending");
        current = RuntimeBinarySemaphoreState::WAIT_RETIRED;
    }
    void makeAvailable(uint64_t epoch) {
        requireEpoch(epoch);
        if (current != RuntimeBinarySemaphoreState::WAIT_RETIRED)
            throw std::logic_error("binary semaphore wait has not retired");
        current = RuntimeBinarySemaphoreState::AVAILABLE;
    }
    [[nodiscard]] bool canSignalAgain() const noexcept { return current == RuntimeBinarySemaphoreState::AVAILABLE; }
    [[nodiscard]] RuntimeBinarySemaphoreState state() const noexcept { return current; }
private:
    void requireEpoch(uint64_t epoch) const {
        if (epoch == 0 || epoch != lastEpoch) throw std::logic_error("stale binary semaphore epoch");
    }
    uint64_t lastEpoch{};
    RuntimeBinarySemaphoreState current{RuntimeBinarySemaphoreState::AVAILABLE};
};

class RuntimeIngestPending {
public:
    RuntimeIngestPending() noexcept = default;
    RuntimeIngestPending(const RuntimeIngestPending&) = delete;
    RuntimeIngestPending& operator=(const RuntimeIngestPending&) = delete;
    RuntimeIngestPending(RuntimeIngestPending&&) noexcept = default;
    RuntimeIngestPending& operator=(RuntimeIngestPending&&) noexcept = default;
    [[nodiscard]] uint64_t frameIdentity() const noexcept { return frameId; }
    [[nodiscard]] TemporalSourceSlot destination() const noexcept { return slot; }
    [[nodiscard]] uint64_t epoch() const noexcept { return ingestEpoch; }
    [[nodiscard]] RuntimeSubmissionRetirement& submissionAuthority() noexcept { return submission; }
    [[nodiscard]] const RuntimeSubmissionRetirement& submissionAuthority() const noexcept { return submission; }
    [[nodiscard]] RuntimeTemporarySemaphorePayload& payloadAuthority() noexcept { return payload; }
    [[nodiscard]] const RuntimeTemporarySemaphorePayload& payloadAuthority() const noexcept { return payload; }
private:
    friend struct RuntimeOperationAuthorityTestAccess;
    friend struct RuntimeShadowIngestAuthorityAccess;
    RuntimeIngestPending(uint64_t frame, TemporalSourceSlot destination, uint64_t epoch,
            RuntimeSubmissionRetirement&& submit, RuntimeTemporarySemaphorePayload&& imported,
            std::shared_ptr<const uint8_t> anchor) :
        frameId(frame), slot(destination), ingestEpoch(epoch), submission(std::move(submit)),
        payload(std::move(imported)), lifetime(std::move(anchor)) {
        if (frameId == 0 || temporalSourceSlotIndex(slot) >= 2 || ingestEpoch == 0 || !lifetime)
            throw std::invalid_argument("invalid ingest pending identity");
    }
    uint64_t frameId{};
    TemporalSourceSlot slot{};
    uint64_t ingestEpoch{};
    RuntimeSubmissionRetirement submission;
    RuntimeTemporarySemaphorePayload payload;
    std::shared_ptr<const uint8_t> lifetime;
};

class ReturnedGeneratedOperation {
public:
    ReturnedGeneratedOperation() noexcept = default;
    ReturnedGeneratedOperation(const ReturnedGeneratedOperation&) = delete;
    ReturnedGeneratedOperation& operator=(const ReturnedGeneratedOperation&) = delete;
    ReturnedGeneratedOperation(ReturnedGeneratedOperation&& other) noexcept :
        pair(std::exchange(other.pair, RuntimeTemporalPairIdentity{})),
        bReturn(std::move(other.bReturn)), aReturnSubmission(std::move(other.aReturnSubmission)),
        aReturn(std::move(other.aReturn)), payload(std::move(other.payload)),
        generatedOutput(std::move(other.generatedOutput)),
        generatedRetirement(std::move(other.generatedRetirement)),
        returnedForGraphics(std::exchange(other.returnedForGraphics, VK_NULL_HANDLE)),
        returnedConsumed(std::exchange(other.returnedConsumed, true)),
        generatedRetired(std::exchange(other.generatedRetired, false)),
        bFenceOwner(std::move(other.bFenceOwner)),
        returnResources(std::move(other.returnResources)), lifetime(std::move(other.lifetime)) {}
    ReturnedGeneratedOperation& operator=(ReturnedGeneratedOperation&& other) noexcept {
        if (this == &other) return *this;
        pair = std::exchange(other.pair, RuntimeTemporalPairIdentity{});
        bReturn = std::move(other.bReturn);
        aReturnSubmission = std::move(other.aReturnSubmission);
        aReturn = std::move(other.aReturn);
        payload = std::move(other.payload);
        generatedOutput = std::move(other.generatedOutput);
        generatedRetirement = std::move(other.generatedRetirement);
        returnedForGraphics = std::exchange(other.returnedForGraphics, VK_NULL_HANDLE);
        returnedConsumed = std::exchange(other.returnedConsumed, true);
        generatedRetired = std::exchange(other.generatedRetired, false);
        bFenceOwner = std::move(other.bFenceOwner);
        returnResources = std::move(other.returnResources);
        lifetime = std::move(other.lifetime);
        return *this;
    }
    [[nodiscard]] bool valid() const noexcept {
        const bool generatedAuthorityValid = generatedRetired
            || (generatedOutput.valid() && generatedRetirement.valid());
        return validTemporalPair(pair) && bReturn.valid() && aReturnSubmission.valid()
            && aReturn.valid() && payload.valid() && returnedForGraphics != VK_NULL_HANDLE
            && generatedAuthorityValid && returnResources && lifetime;
    }
    [[nodiscard]] RuntimeTemporalPairIdentity identity() const noexcept { return pair; }
    [[nodiscard]] VkSemaphore returnedReady() const noexcept { return returnedForGraphics; }
    [[nodiscard]] vk::RuntimeForeignImageReadbackPending& aReturnPending() noexcept { return aReturn; }
    [[nodiscard]] const vk::RuntimeForeignImageReadbackPending& aReturnPending() const noexcept { return aReturn; }
    [[nodiscard]] RuntimeSubmissionRetirement& bReturnAuthority() noexcept { return bReturn; }
    [[nodiscard]] const RuntimeSubmissionRetirement& bReturnAuthority() const noexcept { return bReturn; }
    [[nodiscard]] RuntimeSubmissionRetirement& aReturnAuthority() noexcept {
        return aReturnSubmission;
    }
    [[nodiscard]] const RuntimeSubmissionRetirement& aReturnAuthority() const noexcept {
        return aReturnSubmission;
    }
    [[nodiscard]] RuntimeTemporarySemaphorePayload& payloadAuthority() noexcept { return payload; }
    [[nodiscard]] const RuntimeTemporarySemaphorePayload& payloadAuthority() const noexcept { return payload; }
    [[nodiscard]] bool generatedOutputLive() const noexcept {
        return !generatedRetired && generatedOutput.valid() && generatedRetirement.valid();
    }
    [[nodiscard]] bool generatedOutputRetired() const noexcept { return generatedRetired; }
    // Terminal readiness is the submitted ownership phase. It accepts either
    // a still-pending or already-retired A fence without weakening retirement.
    [[nodiscard]] bool terminalReady() const noexcept {
        if (!valid() || !generatedRetired || !returnedForGraphicsOutstanding()
                || !aReturn.terminalReady())
            return false;
        const auto generation = pair.generationId;
        if (bReturn.epoch() != generation || aReturnSubmission.epoch() != generation
                || payload.epoch() != generation
                || bReturn.state() != RuntimeAuthorityState::RETIRED
                || aReturn.retirementFence() != aReturnSubmission.fenceHandle()
                || aReturn.importedWaitSemaphore() != payload.semaphoreHandle()
                || aReturn.returnedSignalSemaphore() != returnedForGraphics)
            return false;
        const bool submitted = aReturnSubmission.state() == RuntimeAuthorityState::SUBMITTED
            && payload.state() == RuntimeTemporaryPayloadState::WAIT_SUBMITTED
            && !aReturn.retirementObserved();
        const bool retired = aReturnSubmission.state() == RuntimeAuthorityState::RETIRED
            && payload.state() == RuntimeTemporaryPayloadState::WAIT_RETIRED
            && aReturn.retirementObserved();
        return submitted || retired;
    }
    [[nodiscard]] bool aReturnRetired() const noexcept {
        return validTemporalPair(pair) && aReturn.retirementObserved()
            && aReturnSubmission.epoch() == pair.generationId
            && payload.epoch() == pair.generationId
            && aReturnSubmission.state() == RuntimeAuthorityState::RETIRED
            && payload.state() == RuntimeTemporaryPayloadState::WAIT_RETIRED;
    }
    void rejectGeneratedOutputRetirement() const {
        if (bReturn.state() != RuntimeAuthorityState::RETIRED)
            throw std::logic_error("generated output still has an in-flight B-return reader");
    }
    void retireAReturn(uint64_t epoch) {
        if (!aReturn.retirementObserved())
            throw std::logic_error("A-return fence has not retired");
        aReturnSubmission.retire(epoch);
        payload.waitRetired(epoch);
    }
    [[nodiscard]] bool tryRetireAReturn() {
        if (aReturnRetired())
            return true;
        if (!validTemporalPair(pair) || !aReturn.valid()
                || aReturnSubmission.epoch() != pair.generationId
                || payload.epoch() != pair.generationId
                || aReturnSubmission.state() != RuntimeAuthorityState::SUBMITTED
                || payload.state() != RuntimeTemporaryPayloadState::WAIT_SUBMITTED)
            throw std::logic_error("A-return retirement authority is not pending");
        try {
            if (!aReturn.retirementObserved()
                    && !vk::RuntimeImageEndpoint::tryRetireForeignImageReadback(aReturn))
                return false;
            retireAReturn(pair.generationId);
            return true;
        } catch (...) {
            failAReturn();
            throw;
        }
    }
    void failAReturn() noexcept {
        aReturnSubmission.fail();
        payload.fail();
    }
    // A-return produces this binary signal for the future terminal waiter.  It
    // is deliberately not consumed here: the A-return fence proves producer
    // completion, never consumption by that future waiter.
    [[nodiscard]] bool returnedForGraphicsOutstanding() const noexcept {
        return returnedForGraphics != VK_NULL_HANDLE && !returnedConsumed;
    }
    void rejectReturnedForGraphicsReuse() const {
        if (returnedForGraphicsOutstanding())
            throw std::logic_error("returnedForGraphics is terminal-facing and outstanding");
    }
    void consumeReturnedForGraphics() {
        if (!returnedForGraphicsOutstanding())
            throw std::logic_error("returnedForGraphics is not outstanding");
        returnedConsumed = true;
    }
private:
    friend struct RuntimeOperationAuthorityTestAccess;
    friend class ::lsfgvk::layer::GeneratedOutputReturnDiagnosticSession;
    ReturnedGeneratedOperation(RuntimeTemporalPairIdentity identity,
            RuntimeSubmissionRetirement&& bSubmission,
            RuntimeSubmissionRetirement&& aSubmission,
            vk::RuntimeForeignImageReadbackPending&& aPending,
            RuntimeTemporarySemaphorePayload&& imported,
            RuntimeGenerateDiagnosticPending&& generated,
            RuntimeGenerationOperationRetirement&& generatedAuthority,
            VkSemaphore ready, std::shared_ptr<VkFence> returnFenceOwner,
            std::shared_ptr<void> resources, std::shared_ptr<const uint8_t> anchor) :
        pair(identity), bReturn(std::move(bSubmission)),
        aReturnSubmission(std::move(aSubmission)), aReturn(std::move(aPending)),
        payload(std::move(imported)), generatedOutput(std::move(generated)),
        generatedRetirement(std::move(generatedAuthority)), returnedForGraphics(ready),
        bFenceOwner(std::move(returnFenceOwner)), returnResources(std::move(resources)),
        lifetime(std::move(anchor)) {
        if (!validTemporalPair(pair) || returnedForGraphics == VK_NULL_HANDLE || !lifetime
                || !bReturn.valid() || !aReturnSubmission.valid() || !aReturn.valid()
                || !payload.valid() || !generatedOutput.valid()
                || !generatedRetirement.valid() || !bFenceOwner || !returnResources)
            throw std::invalid_argument("invalid returned generated operation");
    }
    void markGeneratedOutputRetired() {
        if (generatedRetired || generatedOutput.valid() || generatedRetirement.valid())
            throw std::logic_error("generated-output retirement ownership was not consumed");
        generatedRetired = true;
    }
    RuntimeTemporalPairIdentity pair{};
    RuntimeSubmissionRetirement bReturn;
    RuntimeSubmissionRetirement aReturnSubmission;
    vk::RuntimeForeignImageReadbackPending aReturn;
    RuntimeTemporarySemaphorePayload payload;
    RuntimeGenerateDiagnosticPending generatedOutput;
    RuntimeGenerationOperationRetirement generatedRetirement;
    VkSemaphore returnedForGraphics{};
    bool returnedConsumed{};
    bool generatedRetired{};
    std::shared_ptr<VkFence> bFenceOwner;
    std::shared_ptr<void> returnResources;
    std::shared_ptr<const uint8_t> lifetime;
};

}
