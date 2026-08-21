/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "runtime_generate_diagnostic_state.hpp"

#include <cassert>
#include <stdexcept>

using lsfgvk::backend::RuntimeGenerateDiagnosticAction;
using lsfgvk::backend::RuntimeGenerateDiagnosticState;

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

    state.recordValidatedOutput();
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
}
