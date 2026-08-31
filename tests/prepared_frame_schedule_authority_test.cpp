/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/prepared_frame_schedule_authority.hpp"

#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace {
bool failAllocations{};
size_t deleteCalls{};
}

void* operator new(std::size_t size) {
    if (failAllocations) throw std::bad_alloc();
    if (void* allocation = std::malloc(size)) return allocation;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept {
    ++deleteCalls;
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept {
    ++deleteCalls;
    std::free(value);
}

int main() {
    using vk::GenerationSlotAuthority;
    using vk::GenerationSlotPhase;

    // Host-only reserve/abort returns the slot without any accepted work.
    GenerationSlotAuthority aborted;
    assert(aborted.reserve());
    assert(!aborted.reserve());
    aborted.abort();
    assert(aborted.phase() == GenerationSlotPhase::Available);

    // A pre-enqueue failure is immediately reusable; DeviceLost is retained.
    GenerationSlotAuthority preenqueue;
    assert(preenqueue.reserve());
    preenqueue.submitFailed(false);
    assert(preenqueue.phase() == GenerationSlotPhase::Available);
    assert(preenqueue.reserve());
    preenqueue.submitFailed(true);
    assert(preenqueue.phase() == GenerationSlotPhase::Retained);

    for (const VkResult transient : {
            VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        GenerationSlotAuthority rejected;
        assert(rejected.reserve());
        assert(!vk::applyPreparedFrameSubmitFailure(rejected, false, transient));
        assert(rejected.phase() == GenerationSlotPhase::Available);
    }
    GenerationSlotAuthority lostBeforeSubmit;
    assert(lostBeforeSubmit.reserve());
    assert(vk::applyPreparedFrameSubmitFailure(
        lostBeforeSubmit, false, VK_ERROR_DEVICE_LOST));
    assert(lostBeforeSubmit.phase() == GenerationSlotPhase::Retained);

    // Any partial accepted schedule is conservatively retained.
    GenerationSlotAuthority partial;
    assert(partial.reserve());
    partial.submitAccepted(false);
    partial.submitFailed(false);
    assert(partial.phase() == GenerationSlotPhase::Retained);
    GenerationSlotAuthority partialOohm;
    assert(partialOohm.reserve());
    partialOohm.submitAccepted(false);
    assert(vk::applyPreparedFrameSubmitFailure(
        partialOohm, true, VK_ERROR_OUT_OF_HOST_MEMORY));
    assert(partialOohm.phase() == GenerationSlotPhase::Retained);

    // Terminal success needs both independent authorities.
    GenerationSlotAuthority dual;
    assert(dual.reserve());
    dual.submitAccepted(true);
    assert(!dual.retire(false, false));
    assert(!dual.retire(true, false));
    assert(!dual.retire(false, true));
    assert(dual.phase() == GenerationSlotPhase::InFlight);
    assert(dual.retire(true, true));
    assert(dual.phase() == GenerationSlotPhase::Available);

    // Two slots coexist; a third claim is unavailable without waiting.
    GenerationSlotAuthority slotA;
    GenerationSlotAuthority slotB;
    assert(slotA.reserve() && slotB.reserve());
    assert(!slotA.reserve() && !slotB.reserve());
    slotA.submitAccepted(true);
    slotB.submitAccepted(true);
    assert(slotA.phase() == GenerationSlotPhase::InFlight);
    assert(slotB.phase() == GenerationSlotPhase::InFlight);

    // The production sequencing helper emits one prepass plus N generated
    // phases, marks only accepted submits, and is allocation-free.
    size_t submits{};
    size_t accepts{};
    failAllocations = true;
    vk::executePreparedFrameSchedule(4,
        [&](vk::PreparedFrameSubmitPhase phase) noexcept {
            assert(phase.index == submits);
            assert(phase.terminal == (phase.index == 4));
            ++submits;
        },
        [&](vk::PreparedFrameSubmitPhase phase) noexcept {
            assert(phase.index + 1 == submits);
            ++accepts;
        });
    failAllocations = false;
    assert(submits == 5 && accepts == 5);

    // Every prepass/generated failure boundary stops before accepting the
    // rejected phase, preserving the exact partial-submit prefix.
    for (size_t rejected = 0; rejected < 5; ++rejected) {
        submits = 0;
        accepts = 0;
        try {
            vk::executePreparedFrameSchedule(4,
                [&](vk::PreparedFrameSubmitPhase phase) {
                    if (phase.index == rejected)
                        throw std::runtime_error("injected submit rejection");
                    ++submits;
                },
                [&](vk::PreparedFrameSubmitPhase) noexcept { ++accepts; });
            assert(false);
        } catch (const std::runtime_error&) {
            assert(submits == rejected);
            assert(accepts == rejected);
        }
    }
}
