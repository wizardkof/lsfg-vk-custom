#include "lsfg-vk-backend/runtime_operation_authority.hpp"

#include <cassert>
#include <type_traits>

using namespace lsfgvk::backend;

namespace lsfgvk::backend {
struct RuntimeOperationAuthorityTestAccess {
    static RuntimeSubmissionRetirement submission(uintptr_t handle, uint64_t epoch,
            const std::shared_ptr<const uint8_t>& lifetime) {
        return {reinterpret_cast<VkFence>(handle), epoch, lifetime};
    }
    static RuntimeTemporarySemaphorePayload payload(uintptr_t handle, uint64_t epoch,
            const std::shared_ptr<const uint8_t>& lifetime) {
        return {reinterpret_cast<VkSemaphore>(handle), epoch, lifetime};
    }
    static RuntimeIngestPending ingest(uint64_t frame, TemporalSourceSlot slot, uint64_t epoch,
            RuntimeSubmissionRetirement&& submission,
            RuntimeTemporarySemaphorePayload&& payload,
            const std::shared_ptr<const uint8_t>& lifetime) {
        return {frame, slot, epoch, std::move(submission), std::move(payload), lifetime};
    }
};
}

int main() {
    static_assert(!std::is_copy_constructible_v<RuntimeSubmissionRetirement>);
    static_assert(!std::is_copy_constructible_v<RuntimeTemporarySemaphorePayload>);
    static_assert(!std::is_copy_constructible_v<RuntimeIngestPending>);
    static_assert(!std::is_copy_constructible_v<ReturnedGeneratedOperation>);

    auto lifetime = std::make_shared<const uint8_t>(0);
    auto submission = RuntimeOperationAuthorityTestAccess::submission(1, 1, lifetime);
    assert(!submission.canResetFence());
    submission.submitted();
    bool staleRejected = false;
    try { submission.retire(2); } catch (const std::logic_error&) { staleRejected = true; }
    assert(staleRejected);
    submission.retire(1);
    assert(submission.canResetFence());
    bool doubleRetireRejected = false;
    try { submission.retire(1); } catch (const std::logic_error&) { doubleRetireRejected = true; }
    assert(doubleRetireRejected);

    auto payload = RuntimeOperationAuthorityTestAccess::payload(2, 1, lifetime);
    assert(!payload.canReimportPayload());
    payload.waitSubmitted(1);
    payload.waitRetired(1);
    assert(payload.canReimportPayload());

    RuntimeBinarySemaphoreEpoch binary;
    binary.reserveSignal(1);
    assert(binary.state() == RuntimeBinarySemaphoreState::SIGNAL_RESERVED);
    binary.producerSubmitted(1);
    bool doubleSignalRejected = false;
    try { binary.signalSubmitted(2); } catch (const std::logic_error&) { doubleSignalRejected = true; }
    assert(doubleSignalRejected);
    binary.waitSubmitted(1);
    binary.waitRetired(1);
    binary.makeAvailable(1);
    assert(binary.canSignalAgain());
    binary.signalSubmitted(2);
    binary.consumerFailed(2);
    assert(binary.state() == RuntimeBinarySemaphoreState::FAILED);
    assert(!binary.canSignalAgain());

    auto ingestSubmission = RuntimeOperationAuthorityTestAccess::submission(3, 3, lifetime);
    auto ingestPayload = RuntimeOperationAuthorityTestAccess::payload(4, 3, lifetime);
    auto ingest = RuntimeOperationAuthorityTestAccess::ingest(5,
        TemporalSourceSlot::Slot0, 3, std::move(ingestSubmission),
        std::move(ingestPayload), lifetime);
    assert(ingest.frameIdentity() == 5);
    assert(ingest.destination() == TemporalSourceSlot::Slot0);
    auto moved = std::move(ingest);
    assert(moved.epoch() == 3);
    auto slot1Submission = RuntimeOperationAuthorityTestAccess::submission(7, 4, lifetime);
    auto slot1Payload = RuntimeOperationAuthorityTestAccess::payload(8, 4, lifetime);
    auto slot1 = RuntimeOperationAuthorityTestAccess::ingest(6,
        TemporalSourceSlot::Slot1, 4, std::move(slot1Submission),
        std::move(slot1Payload), lifetime);
    assert(slot1.frameIdentity() == 6);
    assert(slot1.destination() == TemporalSourceSlot::Slot1);
    assert(slot1.epoch() == 4);

    // An ingest fence retires the command/payload authorities, not the binary
    // semaphore signal. Only a future Generate wait may consume that signal.
    RuntimeBinarySemaphoreEpoch ingestReady;
    ingestReady.reserveSignal(7);
    ingestReady.producerSubmitted(7);
    auto realFence = RuntimeOperationAuthorityTestAccess::submission(5, 7, lifetime);
    auto realPayload = RuntimeOperationAuthorityTestAccess::payload(6, 7, lifetime);
    realFence.submitted();
    realPayload.waitSubmitted(7);
    realFence.retire(7);
    realPayload.waitRetired(7);
    assert(realFence.canResetFence());
    assert(realPayload.canReimportPayload());
    assert(ingestReady.state() == RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED);
    assert(!ingestReady.canSignalAgain());
    bool postFenceSignalRejected = false;
    try { ingestReady.reserveSignal(8); }
    catch (const std::logic_error&) { postFenceSignalRejected = true; }
    assert(postFenceSignalRejected);

    // The accepted Generate submit consumes the outstanding signal epoch. Its
    // source-read fence, not ingestFence, is the proof that permits reuse.
    ingestReady.waitSubmitted(7);
    assert(ingestReady.state() == RuntimeBinarySemaphoreState::WAIT_SUBMITTED);
    assert(!ingestReady.canSignalAgain());
    bool staleWaitRejected = false;
    try { ingestReady.waitRetired(8); }
    catch (const std::logic_error&) { staleWaitRejected = true; }
    assert(staleWaitRejected);
    ingestReady.waitRetired(7);
    assert(!ingestReady.canSignalAgain());
    ingestReady.makeAvailable(7);
    assert(ingestReady.canSignalAgain());

    RuntimeBinarySemaphoreEpoch failedProducer;
    failedProducer.producerFailed();
    assert(!failedProducer.canSignalAgain());
}
