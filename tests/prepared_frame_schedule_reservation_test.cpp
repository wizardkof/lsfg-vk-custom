/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-backend/lsfgvk.hpp"

#include <cassert>
#include <utility>

namespace {
struct Counts { size_t execute{}; size_t abort{}; };
void execute(void* value) { ++static_cast<Counts*>(value)->execute; }
void abort(void* value) { ++static_cast<Counts*>(value)->abort; }
}

int main() {
    using lsfgvk::backend::PreparedFrameScheduleReservationTestAccess;

    // Move construction transfers the only execute authority.
    Counts movedCounts;
    auto original = PreparedFrameScheduleReservationTestAccess::make(
        &movedCounts, execute, abort);
    auto moved = std::move(original);
    assert(!original.valid() && moved.valid());
    moved.execute();
    assert(movedCounts.execute == 1 && movedCounts.abort == 0);
    assert(!moved.valid());
    try { moved.execute(); assert(false); } catch (const lsfgvk::backend::error&) {}

    // Move assignment aborts the previously owned host-only reservation.
    Counts lhsCounts;
    Counts rhsCounts;
    auto lhs = PreparedFrameScheduleReservationTestAccess::make(
        &lhsCounts, execute, abort);
    auto rhs = PreparedFrameScheduleReservationTestAccess::make(
        &rhsCounts, execute, abort);
    lhs = std::move(rhs);
    assert(lhsCounts.abort == 1 && !rhs.valid() && lhs.valid());
    lhs.execute();
    assert(rhsCounts.execute == 1 && rhsCounts.abort == 0);

    // Passive destruction of an unexecuted authority performs host-only abort.
    Counts abandoned;
    {
        auto reservation = PreparedFrameScheduleReservationTestAccess::make(
            &abandoned, execute, abort);
        assert(reservation.valid());
    }
    assert(abandoned.abort == 1 && abandoned.execute == 0);
}
