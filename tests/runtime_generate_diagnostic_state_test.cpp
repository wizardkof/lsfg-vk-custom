/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "runtime_generate_diagnostic_state.hpp"

#include <cassert>
#include <stdexcept>

using lsfgvk::backend::RuntimeGenerateDiagnosticAction;
using lsfgvk::backend::RuntimeGenerateDiagnosticState;
using lsfgvk::backend::RuntimeGenerateSerialPhase;
using lsfgvk::backend::RuntimeGenerateSerialState;

int main() {
    RuntimeGenerateDiagnosticState state;

    const auto seed = state.advance();
    assert(seed.action == RuntimeGenerateDiagnosticAction::SEED_ONLY);
    assert(seed.directFrames == 1 && seed.rotations == 0 && seed.frameIndex == 0);
    assert(seed.olderSlot == 0 && seed.newerSlot == 0);
    assert(state.generatedFrames() == 0);
    assert(!state.finalPassReady());
    assert(!state.completionResultAvailable());

    const auto d2a = state.advance();
    assert(d2a.action == RuntimeGenerateDiagnosticAction::GAMMA_DELTA);
    assert(d2a.directFrames == 2 && d2a.rotations == 1 && d2a.frameIndex == 1);
    assert(d2a.olderSlot == 0 && d2a.newerSlot == 1);
    assert(state.generatedFrames() == 0);
    assert(!state.finalPassReady());
    assert(!state.completionResultAvailable());

    const auto d2b = state.advance();
    assert(d2b.action == RuntimeGenerateDiagnosticAction::GENERATE);
    assert(d2b.directFrames == 3 && d2b.rotations == 2 && d2b.frameIndex == 0);
    assert(d2b.olderSlot == 1 && d2b.newerSlot == 0);
    assert(d2b.generateDescriptorSet == 0);
    assert(state.generatedFrames() == 1);
    assert(!state.finalPassReady());
    assert(!state.completionResultAvailable());
    assert(!state.pendingGeneration());

    state.recordSubmittedGeneration();
    assert(state.pendingGeneration());
    assert(!state.finalPassReady());
    assert(!state.completionResultAvailable());
    bool rejectedOverlappingPending = false;
    try { state.recordSubmittedGeneration(); }
    catch (const std::logic_error&) { rejectedOverlappingPending = true; }
    assert(rejectedOverlappingPending);

    state.recordValidatedOutput();
    assert(!state.pendingGeneration());
    assert(state.finalPassReady());
    assert(state.completionResultAvailable());
    state.issueCompletionResult();
    assert(!state.completionResultAvailable());
    bool rejectedSecondResult = false;
    try { state.issueCompletionResult(); }
    catch (const std::logic_error&) { rejectedSecondResult = true; }
    assert(rejectedSecondResult);

    bool rejectedExtraGenerate = false;
    try { static_cast<void>(state.advance()); }
    catch (const std::logic_error&) { rejectedExtraGenerate = true; }
    assert(rejectedExtraGenerate);

    RuntimeGenerateDiagnosticState failed;
    static_cast<void>(failed.advance());
    failed.recordFailure();
    bool rejectedAfterFailure = false;
    try { static_cast<void>(failed.advance()); }
    catch (const std::logic_error&) { rejectedAfterFailure = true; }
    assert(rejectedAfterFailure);
    assert(!failed.finalPassReady());
    assert(!failed.completionResultAvailable());
    bool rejectedFailedResult = false;
    try { failed.issueCompletionResult(); }
    catch (const std::logic_error&) { rejectedFailedResult = true; }
    assert(rejectedFailedResult);

    RuntimeGenerateSerialState serial;
    for (uint64_t epoch = 1; epoch <= 3; ++epoch) {
        assert(serial.canSubmit());
        serial.submit(epoch);
        assert(!serial.canSubmit());
        bool rejectedOverlap = false;
        try { serial.submit(epoch + 1); }
        catch (const std::logic_error&) { rejectedOverlap = true; }
        assert(rejectedOverlap);
        serial.sourceReadsRetired(epoch);
        assert(!serial.canSubmit());
        serial.outputInUse(epoch);
        serial.retireOperation(epoch);
        assert(serial.currentPhase() == RuntimeGenerateSerialPhase::READY_NEXT);
    }
    assert(serial.currentEpoch() == 3);
    bool rejectedStale = false;
    try { serial.retireOperation(2); }
    catch (const std::logic_error&) { rejectedStale = true; }
    assert(rejectedStale);

    RuntimeGenerateSerialState serialFailure;
    serialFailure.submit(1);
    serialFailure.fail();
    assert(!serialFailure.canSubmit());
    bool rejectedFailedSerial = false;
    try { serialFailure.submit(2); }
    catch (const std::logic_error&) { rejectedFailedSerial = true; }
    assert(rejectedFailedSerial);
}
