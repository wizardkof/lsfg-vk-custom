#include "d3b3_production_seams.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"

#include <utility>

#include <stdexcept>

namespace lsfgvk::layer {

#ifdef LSFGVK_D3B3_FINITE_TESTING
D3B3PairOperation::D3B3PairOperation(
        backend::ReturnedGeneratedOperation&& value,
        D3B3OriginalSourceAuthority&& source,
        D3B3PairTerminalDispatch terminal) :
    returned(std::move(value)), original(std::move(source)),
    dispatch(std::move(terminal)) {
    if (!returned.valid() || !original.valid() || !dispatch.build
            || !dispatch.retireGeneratedPresentFence
            || !dispatch.retireOriginalPresentFence
            || !dispatch.confirmHiddenDestinationsReusable)
        throw std::invalid_argument("invalid D3B3 Pair1 terminal authorities");

    const auto pair = returned.identity();
    if (!backend::validTemporalPair(pair)
            || pair.newerFrameId != original.source().frameId
            || !returned.terminalReady())
        throw std::invalid_argument("D3B3 Pair1 requires terminal-ready B0C authorities");

    returnedForGraphics.signalSubmitted(pair.generationId);
}

D3B3PairOperation::D3B3PairOperation(D3B3PairOperation&& other) noexcept :
    returned(std::move(other.returned)), original(std::move(other.original)),
    dispatch(std::move(other.dispatch)),
    returnedForGraphics(std::move(other.returnedForGraphics)),
    terminalPath(std::move(other.terminalPath)), terminalState(other.terminalState),
    current(other.current), generatedPresentSubmitted(other.generatedPresentSubmitted),
    originalPresentSubmitted(other.originalPresentSubmitted),
    generatedPresentRetiredValue(other.generatedPresentRetiredValue),
    originalPresentRetiredValue(other.originalPresentRetiredValue),
    hiddenEmpty(other.hiddenEmpty) {
    bindCallbacks();
    other.current = D3B3PairState::FAILED;
    other.terminalPath.reset();
}

D3B3PairOperation& D3B3PairOperation::operator=(D3B3PairOperation&& other) noexcept {
    if (this == &other) return *this;
    returned = std::move(other.returned);
    original = std::move(other.original);
    dispatch = std::move(other.dispatch);
    returnedForGraphics = std::move(other.returnedForGraphics);
    terminalPath = std::move(other.terminalPath);
    terminalState = other.terminalState;
    current = other.current;
    generatedPresentSubmitted = other.generatedPresentSubmitted;
    originalPresentSubmitted = other.originalPresentSubmitted;
    generatedPresentRetiredValue = other.generatedPresentRetiredValue;
    originalPresentRetiredValue = other.originalPresentRetiredValue;
    hiddenEmpty = other.hiddenEmpty;
    bindCallbacks();
    other.current = D3B3PairState::FAILED;
    other.terminalPath.reset();
    return *this;
}

bool D3B3PairOperation::valid() const noexcept {
    return current != D3B3PairState::FAILED
        && current != D3B3PairState::PAIR_RETIRED
        && returned.valid() && original.valid();
}

void D3B3PairOperation::preflight() {
    if (current != D3B3PairState::NON_TERMINAL_READY)
        throw std::logic_error("D3B3 Pair1 preflight is not the next transition");
    try {
        const auto view = returned.aReturnPending().imageView();
        if (!view.valid() || view.layout() != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                || view.sourceQueueFamily() == VK_QUEUE_FAMILY_IGNORED
                || view.destinationQueueFamily() == VK_QUEUE_FAMILY_IGNORED
                || original.source().queueFamily == VK_QUEUE_FAMILY_IGNORED
                || dispatch.originalReady == VK_NULL_HANDLE)
            throw std::runtime_error("D3B3 Pair1 source preflight failed");

        const D3B2Source generated{
            .image = view.image(), .format = view.format(), .extent = view.extent(),
            .layout = view.layout(), .sourceQueueFamily = view.sourceQueueFamily(),
            .destinationQueueFamily = view.destinationQueueFamily(),
            .ownershipAcquireRequired = view.handoffPendingAcquire(),
            .blitSourceSupported = true};
        const auto& source = original.source();
        const D3B2Source originalSource{
            .image = source.image, .format = source.format, .extent = source.extent,
            .layout = source.layout, .sourceQueueFamily = source.queueFamily,
            .destinationQueueFamily = source.queueFamily, .blitSourceSupported = true};

        terminalPath.emplace(dispatch.build(
            generated, originalSource, returned.returnedReady()));
        auto& path = *terminalPath;
        path.generated = generated;
        path.original = originalSource;
        path.originalReady = dispatch.originalReady;
        path.returnedForGraphics = returned.returnedReady();
        path.state = &terminalState;
        if (!path.presentFences.enabled() || !path.presentWithFence
                || !path.submitWithWaitStage
                || path.returnedForGraphicsWaitStage
                    != d3b2FirstTerminalConsumerStage())
            throw std::runtime_error("D3B3 Pair1 terminal retirement preflight failed");
        validateD3B2InsertionPreflight(path);
        bindCallbacks();
        current = D3B3PairState::TERMINAL_PREFLIGHTED;
    } catch (...) {
        fail();
        throw;
    }
}

void D3B3PairOperation::submitTerminal() {
    if (current != D3B3PairState::TERMINAL_PREFLIGHTED || !terminalPath)
        throw std::logic_error("D3B3 Pair1 terminal submit is not preflighted");
    current = D3B3PairState::TERMINAL_SUBMIT_PENDING;
    try {
        static_cast<void>(executeD3B2Insertion(*terminalPath));
        if (current != D3B3PairState::GRAPHICS_RETIRED)
            throw std::logic_error("D3B3 Pair1 terminal did not retire its graphics waiter");
    } catch (...) {
        fail();
        throw;
    }
}

bool D3B3PairOperation::tryRetireAReturn() {
    if (current == D3B3PairState::PAIR_RETIRED || !returned.valid())
        throw std::logic_error("D3B3 Pair1 has no live A-return authority");
    try {
        return returned.tryRetireAReturn();
    } catch (...) {
        fail();
        throw;
    }
}

void D3B3PairOperation::retirePresentWaits() {
    if (current == D3B3PairState::GRAPHICS_RETIRED) {
        try {
            dispatch.retireGeneratedPresentFence();
            generatedPresentRetiredValue = true;
            dispatch.retireOriginalPresentFence();
            originalPresentRetiredValue = true;
            current = D3B3PairState::PRESENTS_RETIRED;
        } catch (...) {
            fail();
            throw;
        }
    } else if (current != D3B3PairState::PRESENTS_RETIRED) {
        throw std::logic_error("D3B3 Pair1 presents are not pending retirement");
    }

    if (!hiddenEmpty) {
        if (dispatch.confirmHiddenDestinationsReusable())
            hiddenEmpty = true;
    }
}

void D3B3PairOperation::retirePair() {
    if (current != D3B3PairState::PRESENTS_RETIRED
            || !generatedPresentRetiredValue || !originalPresentRetiredValue
            || !hiddenEmpty || !returnedForGraphicsReusable()
            || !returned.aReturnRetired()
            || original.state() != D3B3OriginalSourceState::TERMINAL_RETIRED)
        throw std::logic_error("D3B3 Pair1 retirement domains are incomplete");
    try {
        // The returned image and the original C lease remain owned by this
        // operation until this point, even though both reads were already
        // proven complete at the graphics fence.
        original.release();
        terminalPath.reset();
        returned = {};
        current = D3B3PairState::PAIR_RETIRED;
    } catch (...) {
        fail();
        throw;
    }
}

void D3B3PairOperation::rejectReturnedForGraphicsReuse() const {
    if (!returnedForGraphicsReusable())
        throw std::logic_error("returnedForGraphics waiter has not retired");
}

void D3B3PairOperation::fail() noexcept {
    current = D3B3PairState::FAILED;
    if (original.state() == D3B3OriginalSourceState::TERMINAL_SUBMITTED)
        original.fail();
}

void D3B3PairOperation::bindCallbacks() {
    if (!terminalPath) return;
    auto& path = *terminalPath;
    path.onDestinationsAcquired = [this] {
        if (current != D3B3PairState::TERMINAL_SUBMIT_PENDING)
            throw std::logic_error("D3B3 Pair1 destinations acquired out of order");
        hiddenEmpty = false;
        current = D3B3PairState::DESTINATIONS_ACQUIRED;
    };
    path.onRecordBegin = [this] {
        if (current != D3B3PairState::DESTINATIONS_ACQUIRED)
            throw std::logic_error("D3B3 Pair1 record began out of order");
    };
    path.onRecordEnd = [this] {
        if (current != D3B3PairState::DESTINATIONS_ACQUIRED)
            throw std::logic_error("D3B3 Pair1 record ended out of order");
    };
    path.onTerminalSubmitAccepted = [this] {
        if (current != D3B3PairState::DESTINATIONS_ACQUIRED)
            throw std::logic_error("D3B3 Pair1 submit accepted out of order");
        returned.consumeReturnedForGraphics();
        returnedForGraphics.waitSubmitted(identity().generationId);
        original.terminalSubmitted();
        current = D3B3PairState::TERMINAL_SUBMITTED;
    };
    path.onGeneratedPresentAccepted = [this] {
        if (current != D3B3PairState::TERMINAL_SUBMITTED)
            throw std::logic_error("D3B3 generated present submitted out of order");
        generatedPresentSubmitted = true;
    };
    path.onOriginalPresentAccepted = [this] {
        if (!generatedPresentSubmitted
                || current != D3B3PairState::TERMINAL_SUBMITTED)
            throw std::logic_error("D3B3 original present submitted out of order");
        originalPresentSubmitted = true;
        current = D3B3PairState::PRESENTS_SUBMITTED;
    };
    path.onGraphicsFenceRetired = [this] {
        if (!generatedPresentSubmitted || !originalPresentSubmitted
                || current != D3B3PairState::PRESENTS_SUBMITTED)
            throw std::logic_error("D3B3 graphics retirement occurred out of order");
        returnedForGraphics.waitRetired(identity().generationId);
        original.terminalRetired();
        current = D3B3PairState::GRAPHICS_RETIRED;
    };
}

RuntimeFrameIngestResult ingestCurrentFrame(
        const D3B3IngestDispatch& dispatch, const CurrentOriginalFrame& frame,
        size_t targetSlot) {
    if (!dispatch.execute || !frame.image || frame.frameId == 0
            || frame.queueFamily == VK_QUEUE_FAMILY_IGNORED)
        throw std::invalid_argument("invalid D3B3 current-frame ingest");
    auto result = dispatch.execute(frame, targetSlot);
    if (!result.source.valid || result.source.slot != targetSlot
            || result.source.frameId != frame.frameId || !result.transportReady
            || !result.completionFence)
        throw std::runtime_error("D3B3 ingest result lacks bounded readiness");
    return result;
}

GeneratedPairIdentity generateTemporalPair(
        const D3B3GenerateDispatch& dispatch, const TemporalSourceRef& older,
        const TemporalSourceRef& newer, uint64_t generationId, float interpolation) {
    if (!dispatch.execute || !older.valid || !newer.valid || older.slot == newer.slot
            || older.frameId == newer.frameId || generationId == 0)
        throw std::invalid_argument("invalid D3B3 temporal pair");
    if (dispatch.execute(older, newer, generationId, interpolation) != VK_SUCCESS)
        throw std::runtime_error("D3B3 Generate submission failed");
    return {older.frameId, newer.frameId, generationId};
}

ReturnedGeneratedFrame returnGeneratedFrame(
        const D3B3ReturnDispatch& dispatch, GeneratedPairIdentity identity) {
    if (!dispatch.execute || identity.olderFrameId == 0 || identity.newerFrameId == 0
            || identity.olderFrameId == identity.newerFrameId)
        throw std::invalid_argument("invalid D3B3 generated return identity");
    auto result = dispatch.execute(identity);
    if (!result.has_value() || !result->image || !result->returnedForGraphics
            || !result->completionFence || result->identity.olderFrameId != identity.olderFrameId
            || result->identity.newerFrameId != identity.newerFrameId
            || result->identity.generationId != identity.generationId)
        throw std::runtime_error("D3B3 return lacks bounded lifetime authority");
    return std::move(*result);
}

VkResult executeReusableTerminal(D3B2InsertionPath& path, bool reusableSlot) {
    if (reusableSlot && !path.presentFences.enabled())
        throw std::invalid_argument("reusable D3B3 terminal requires present fences");
    return executeD3B2Insertion(path);
}
#endif

D3B3RetirementStatus evaluateD3B3Retirement(
        const D3B3RetirementDomains& domains) noexcept {
    if (domains.inputFence && domains.bReturnFence && domains.aReturnComplete
            && domains.graphicsFence && domains.generatedPresentFence
            && domains.originalPresentFence && domains.temporaryPayloadRetired
            && domains.hiddenLedgerEmpty && domains.generatedAuthorityRetired)
        return D3B3RetirementStatus::RETIRED;
    return D3B3RetirementStatus::TIMEOUT;
}

D3B3ProductionState::D3B3ProductionState(Retirement callback) : retirement(std::move(callback)) {}

PrePresentGateResult D3B3ProductionState::prePresentGate() {
    D3B3PendingProductionOperation snapshot;
#ifdef LSFGVK_D3B3_FINITE_TESTING
    bool finitePairActive{};
#endif
    {
        std::scoped_lock lock(mutex);
        if (operation.state == D3B3PendingState::FAILED)
            return PrePresentGateResult::FAILED;
        if (operation.state == D3B3PendingState::EMPTY)
            return PrePresentGateResult::READY;
        snapshot = operation;
        operation.state = D3B3PendingState::RETIRING;
#ifdef LSFGVK_D3B3_FINITE_TESTING
        finitePairActive = finitePair.has_value();
#endif
    }
#ifdef LSFGVK_D3B3_FINITE_TESTING
    const auto status = finitePairActive
        ? retireFinitePair()
        : (retirement ? retirement(snapshot) : evaluateD3B3Retirement(snapshot.domains));
#else
    const auto status = retirement ? retirement(snapshot) : evaluateD3B3Retirement(snapshot.domains);
#endif
    std::scoped_lock lock(mutex);
    if (status == D3B3RetirementStatus::RETIRED) {
        operation = {};
#ifdef LSFGVK_D3B3_FINITE_TESTING
        if (!finiteStop && finiteTotals.applicationFrames >= 2)
            finiteCurrent = D3B3FiniteProductionState::READY_NEXT;
#endif
        return PrePresentGateResult::READY;
    }
    operation.state = D3B3PendingState::PENDING_RETIREMENT;
    if (status == D3B3RetirementStatus::TIMEOUT)
        return PrePresentGateResult::TIMEOUT;
    if (status == D3B3RetirementStatus::DEVICE_LOST)
        return PrePresentGateResult::DEVICE_LOST;
    operation.state = D3B3PendingState::FAILED;
    return PrePresentGateResult::FAILED;
}

bool D3B3ProductionState::beginPending(uint64_t older, uint64_t newer,
        uint64_t generation, size_t oldSlot, size_t newSlot) {
    std::scoped_lock lock(mutex);
    if (operation.state != D3B3PendingState::EMPTY)
        return false;
    operation = {D3B3PendingState::ACTIVE, older, newer, generation, oldSlot, newSlot, {}};
    return true;
}

bool D3B3ProductionState::updatePendingDomains(D3B3RetirementDomains domains) {
    std::scoped_lock lock(mutex);
    if (operation.state == D3B3PendingState::EMPTY
            || operation.state == D3B3PendingState::FAILED)
        return false;
    operation.domains = domains;
    operation.state = D3B3PendingState::PENDING_RETIREMENT;
    return true;
}

bool D3B3ProductionState::markPending(D3B3PendingProductionOperation value) {
    std::scoped_lock lock(mutex);
    if (operation.state != D3B3PendingState::EMPTY)
        return false;
    value.state = D3B3PendingState::PENDING_RETIREMENT;
    operation = value;
    return true;
}

void D3B3ProductionState::setFailure() noexcept {
    std::scoped_lock lock(mutex);
    operation.state = D3B3PendingState::FAILED;
#ifdef LSFGVK_D3B3_FINITE_TESTING
    finiteCurrent = D3B3FiniteProductionState::FAILED;
#endif
}

#ifdef LSFGVK_D3B3_FINITE_TESTING
void D3B3ProductionState::setFiniteStopped(bool value) noexcept {
    std::scoped_lock lock(mutex);
    finiteStop = value;
    if (value) finiteCurrent = D3B3FiniteProductionState::FINITE_STOPPED;
}
#endif

void D3B3ProductionState::setEligibleFrameCount(uint32_t value) noexcept { std::scoped_lock lock(mutex); eligibleFrames = value; }

#ifdef LSFGVK_D3B3_FINITE_TESTING
void D3B3ProductionState::configureFinite(D3B3FiniteProductionOperations operations) {
    if (!operations.ingest || !operations.generate || !operations.presentWarmupOriginal
            || !operations.returnGenerated || !operations.makeOriginal
            || !operations.makeTerminal || !operations.stopWorkerAndJoin
            || !operations.record || !operations.emitMarker)
        throw std::invalid_argument("D3B3 finite production operations are incomplete");
    std::scoped_lock lock(mutex);
    if (finiteTotals.applicationFrames != 0 || finitePair
            || operation.state != D3B3PendingState::EMPTY)
        throw std::logic_error("D3B3 finite production state is already active");
    finiteOperations = std::move(operations);
    finiteCurrent = D3B3FiniteProductionState::WARMUP_EMPTY;
    finiteStop = false;
}

bool D3B3ProductionState::finiteOperationsComplete() const noexcept {
    return finiteOperations.ingest && finiteOperations.generate
        && finiteOperations.presentWarmupOriginal && finiteOperations.returnGenerated
        && finiteOperations.makeOriginal && finiteOperations.makeTerminal
        && finiteOperations.stopWorkerAndJoin && finiteOperations.record
        && finiteOperations.emitMarker;
}

bool D3B3ProductionState::presentFinite(uint64_t frameId) {
    D3B3FiniteProductionOperations operations;
    uint32_t frameNumber{};
    {
        std::scoped_lock lock(mutex);
        if (!finiteOperationsComplete() || finiteStop
                || finiteCurrent == D3B3FiniteProductionState::FAILED
                || finiteCurrent == D3B3FiniteProductionState::FINITE_STOPPED
                || frameId == 0 || frameId <= lastFiniteFrame
                || finiteTotals.applicationFrames >= 5)
            return false;
        operations = finiteOperations;
        frameNumber = finiteTotals.applicationFrames + 1;
    }

    try {
        // The P1D3 gate is consulted before the next frame can create any
        // ingest, bridge, return, or terminal work. A timeout is retryable and
        // does not consume the application frame.
        if (frameNumber >= 3) {
            const auto gate = prePresentGate();
            if (gate != PrePresentGateResult::READY) {
                std::scoped_lock lock(mutex);
                if (gate == PrePresentGateResult::TIMEOUT) {
                    ++finiteTotals.gateTimeouts;
                    finiteCurrent = D3B3FiniteProductionState::WAITING_REUSE;
                } else {
                    finiteCurrent = D3B3FiniteProductionState::FAILED;
                }
                return false;
            }
        }

        const auto record = [&](const char* event) { operations.record(event); };
        if (frameNumber == 1) {
            record("APP_A");
            record("INGEST");
            if (!operations.ingest(frameId, backend::TemporalSourceSlot::Slot0, true)
                    || !operations.presentWarmupOriginal(frameId))
                throw std::runtime_error("D3B3 warmup A failed");
            {
                std::scoped_lock lock(mutex);
                slots[0] = {D3B3TemporalState::RETAINED_HISTORY, frameId, 0};
                slots[1] = {D3B3TemporalState::OVERWRITEABLE, 0, 0};
                ++finiteTotals.applicationFrames;
                ++finiteTotals.warmupOriginals;
                ++eligibleFrames;
                lastFiniteFrame = frameId;
                finiteCurrent = D3B3FiniteProductionState::HISTORY_A_READY;
            }
            return true;
        }

        if (frameNumber == 2) {
            record("APP_B");
            record("INGEST");
            if (!operations.ingest(frameId, backend::TemporalSourceSlot::Slot1, false)
                    || !operations.presentWarmupOriginal(frameId))
                throw std::runtime_error("D3B3 warmup B failed");
            {
                std::scoped_lock lock(mutex);
                if (slots[0].state != D3B3TemporalState::RETAINED_HISTORY)
                    throw std::logic_error("D3B3 warmup B history invariant failed");
                slots[1] = {D3B3TemporalState::RETAINED_HISTORY, frameId, 0};
                slots[0].state = D3B3TemporalState::OVERWRITEABLE;
                ++finiteTotals.applicationFrames;
                ++finiteTotals.warmupOriginals;
                ++eligibleFrames;
                lastFiniteFrame = frameId;
                finiteCurrent = D3B3FiniteProductionState::HISTORY_AB_READY;
            }
            return true;
        }

        std::array<D3B3TemporalProductionSlot, 2> currentSlots;
        {
            std::scoped_lock lock(mutex);
            currentSlots = slots;
        }
        size_t olderIndex{};
        size_t newerIndex{};
        bool foundOlder{};
        bool foundNewer{};
        for (size_t index = 0; index < currentSlots.size(); ++index) {
            if (currentSlots[index].state == D3B3TemporalState::RETAINED_HISTORY) {
                olderIndex = index;
                foundOlder = true;
            } else if (currentSlots[index].state == D3B3TemporalState::OVERWRITEABLE) {
                newerIndex = index;
                foundNewer = true;
            }
        }
        if (!foundOlder || !foundNewer || olderIndex == newerIndex)
            throw std::logic_error("D3B3 finite temporal rotation invariant failed");

        const auto olderSlot = olderIndex == 0
            ? backend::TemporalSourceSlot::Slot0 : backend::TemporalSourceSlot::Slot1;
        const auto newerSlot = newerIndex == 0
            ? backend::TemporalSourceSlot::Slot0 : backend::TemporalSourceSlot::Slot1;
        const uint64_t generationId = finiteGeneration + 1;
        const backend::RuntimeTemporalPairIdentity identity{
            .olderSlot = olderSlot,
            .newerSlot = newerSlot,
            .olderFrameId = currentSlots[olderIndex].frameId,
            .newerFrameId = frameId,
            .generationId = generationId};

        if (!beginPending(identity.olderFrameId, identity.newerFrameId,
                identity.generationId, olderIndex, newerIndex))
            throw std::logic_error("D3B3 finite pending operation already exists");

        const char* applicationEvent = frameNumber == 3 ? "APP_C"
            : (frameNumber == 4 ? "APP_D" : "APP_E");
        record(applicationEvent);
        record("INGEST");
        if (!operations.ingest(frameId, newerSlot, false))
            throw std::runtime_error("D3B3 current ingest failed");
        {
            std::scoped_lock lock(mutex);
            slots[newerIndex] = {D3B3TemporalState::CURRENT, frameId, generationId};
        }

        record("GENERATE");
        if (!operations.generate(identity.olderFrameId, identity.newerFrameId,
                identity.olderSlot, identity.newerSlot, generationId))
            throw std::runtime_error("D3B3 Generate failed");
        record("B_RETURN");
        auto returned = operations.returnGenerated(identity);
        record("A_RETURN");
        auto original = operations.makeOriginal(identity);
        auto terminalDispatch = operations.makeTerminal(identity);
        record("PAIR_READY");
        finitePair.emplace(std::move(returned), std::move(original),
            std::move(terminalDispatch));
        {
            std::scoped_lock lock(mutex);
            ++finiteTotals.bReturns;
            ++finiteTotals.aReturns;
            ++finiteTotals.pairOperations;
        }

        record("TERMINAL_PREFLIGHT");
        finitePair->preflight();
        record("TERMINAL_SUBMIT");
        finitePair->submitTerminal();
        finitePair->retirePresentWaits();

        {
            std::scoped_lock lock(mutex);
            ++finiteTotals.generate;
            ++finiteTotals.terminalSubmits;
            ++finiteTotals.generatedPresents;
            ++finiteTotals.originalPresents;
            finiteTotals.internalPresents += 2;
            ++pairs;
            finiteGeneration = generationId;
            finiteTotals.applicationFrames++;
            ++eligibleFrames;
            lastFiniteFrame = frameId;
            slots[olderIndex] = {D3B3TemporalState::RETAINED_HISTORY,
                identity.olderFrameId, generationId};
            slots[newerIndex] = {D3B3TemporalState::CURRENT,
                identity.newerFrameId, generationId};
            operation.domains = {
                true, true, finitePair->aReturnRetired(), true,
                finitePair->generatedPresentRetired(),
                finitePair->originalPresentRetired(),
                finitePair->aReturnPayloadState()
                    == backend::RuntimeTemporaryPayloadState::WAIT_RETIRED,
                finitePair->hiddenLedgerEmpty(), true};
            operation.state = D3B3PendingState::PENDING_RETIREMENT;
            finiteCurrent = D3B3FiniteProductionState::READY_NEXT;
        }

        if (finiteTotals.pairOperations == 3 && !finishFinite())
            return finiteCurrent == D3B3FiniteProductionState::WAITING_REUSE;
        return true;
    } catch (...) {
        setFailure();
        return false;
    }
}

bool D3B3ProductionState::finishFinite() {
    {
        std::scoped_lock lock(mutex);
        if (finiteTotals.applicationFrames != 5 || finiteTotals.pairOperations != 3
                || finiteCurrent == D3B3FiniteProductionState::FAILED)
            return false;
    }
    const auto gate = prePresentGate();
    if (gate == PrePresentGateResult::TIMEOUT) {
        std::scoped_lock lock(mutex);
        ++finiteTotals.gateTimeouts;
        finiteCurrent = D3B3FiniteProductionState::WAITING_REUSE;
        return false;
    }
    if (gate != PrePresentGateResult::READY) {
        setFailure();
        return false;
    }

    D3B3FiniteProductionOperations operations;
    {
        std::scoped_lock lock(mutex);
        operations = finiteOperations;
    }
    if (!operations.stopWorkerAndJoin()) {
        setFailure();
        return false;
    }
    {
        std::scoped_lock lock(mutex);
        finiteStop = true;
        finiteCurrent = D3B3FiniteProductionState::FINITE_STOPPED;
        ++finiteTotals.markers;
    }
    operations.record("DG2X_P4C_D3B3_FINITE_A_E_SEQUENCE_PASS");
    operations.emitMarker();
    return true;
}

D3B3RetirementStatus D3B3ProductionState::retireFinitePair() {
    try {
        if (!finitePair)
            return D3B3RetirementStatus::RETIRED;
        finitePair->retirePresentWaits();
        if (!finitePair->hiddenLedgerEmpty())
            return D3B3RetirementStatus::TIMEOUT;
        if (!finitePair->tryRetireAReturn())
            return D3B3RetirementStatus::TIMEOUT;
        {
            std::scoped_lock lock(mutex);
            operation.domains.aReturnComplete = finitePair->aReturnRetired();
            operation.domains.temporaryPayloadRetired =
                finitePair->aReturnPayloadState()
                    == backend::RuntimeTemporaryPayloadState::WAIT_RETIRED;
        }
        finitePair->retirePair();
        const auto pendingOperation = operation;
        {
            std::scoped_lock lock(mutex);
            slots[pendingOperation.olderSlot].state = D3B3TemporalState::OVERWRITEABLE;
            slots[pendingOperation.newerSlot].state = D3B3TemporalState::RETAINED_HISTORY;
            finitePair.reset();
        }
        return D3B3RetirementStatus::RETIRED;
    } catch (const ls::vulkan_error& error) {
        return error.error() == VK_ERROR_DEVICE_LOST
            ? D3B3RetirementStatus::DEVICE_LOST : D3B3RetirementStatus::FAILURE;
    } catch (...) {
        return D3B3RetirementStatus::FAILURE;
    }
}
#endif

D3B3PendingProductionOperation D3B3ProductionState::pending() const { std::scoped_lock lock(mutex); return operation; }
std::array<D3B3TemporalProductionSlot, 2> D3B3ProductionState::temporalSlots() const { std::scoped_lock lock(mutex); return slots; }
uint32_t D3B3ProductionState::eligibleFrameCount() const noexcept { std::scoped_lock lock(mutex); return eligibleFrames; }
uint32_t D3B3ProductionState::pairCount() const noexcept { std::scoped_lock lock(mutex); return pairs; }
#ifdef LSFGVK_D3B3_FINITE_TESTING
bool D3B3ProductionState::finiteStopped() const noexcept { std::scoped_lock lock(mutex); return finiteStop; }
D3B3FiniteProductionState D3B3ProductionState::finiteState() const noexcept { std::scoped_lock lock(mutex); return finiteCurrent; }
D3B3FiniteProductionCounters D3B3ProductionState::finiteCounters() const noexcept { std::scoped_lock lock(mutex); return finiteTotals; }
bool D3B3ProductionState::acceptsPendingEpoch(uint64_t generationId) const noexcept {
    std::scoped_lock lock(mutex);
    return generationId != 0 && operation.state != D3B3PendingState::EMPTY
        && operation.state != D3B3PendingState::FAILED
        && operation.generationId == generationId && finitePair
        && finitePair->identity().generationId == generationId;
}
#endif
void D3B3ProductionState::recordPair(uint64_t older, uint64_t newer) {
    std::scoped_lock lock(mutex);
    if (pairs < 3) ++pairs;
    slots[0] = {D3B3TemporalState::RETAINED_HISTORY, older, pairs};
    slots[1] = {D3B3TemporalState::CURRENT, newer, pairs};
}

void D3B3ProductionState::recordPair(const backend::RuntimeTemporalPairIdentity& pair) {
    std::scoped_lock lock(mutex);
    if (!backend::validTemporalPair(pair))
        throw std::invalid_argument("invalid D3B3 production pair identity");
    if (pairs < 3) ++pairs;
    slots[backend::temporalSourceSlotIndex(pair.olderSlot)] = {
        D3B3TemporalState::RETAINED_HISTORY, pair.olderFrameId, pair.generationId};
    slots[backend::temporalSourceSlotIndex(pair.newerSlot)] = {
        D3B3TemporalState::CURRENT, pair.newerFrameId, pair.generationId};
}

}
