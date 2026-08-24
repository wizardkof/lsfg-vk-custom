#include "d3b3_production_seams.hpp"

#include <cassert>

using namespace lsfgvk::layer;

int main() {
    D3B3ProductionState state;
    assert(state.prePresentGate() == PrePresentGateResult::READY);
    assert(state.hasTerminalSlot());
    assert(state.temporalSlots().size() == 2);
    assert(state.beginPending(1, 2, 1, 0, 1));
    assert(!state.beginPending(2, 3, 2, 1, 0));
    assert(state.prePresentGate() == PrePresentGateResult::TIMEOUT);

    D3B3PendingProductionOperation retired = state.pending();
    retired.domains = {true, true, true, true, true, true, true, true, true};
    D3B3ProductionState retiring([&](const auto&) { return D3B3RetirementStatus::RETIRED; });
    assert(retiring.markPending(retired));
    assert(retiring.prePresentGate() == PrePresentGateResult::READY);
    retiring.recordPair(2, 3);
    retiring.recordPair(3, 4);
    retiring.recordPair(4, 5);
    retiring.recordPair(5, 6);
    assert(retiring.pairCount() == 3);
    assert(retiring.eligibleFrameCount() == 0);
    return 0;
}
