/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "../lsfg-vk-layer/src/fixed_output_pacer.hpp"

#include <cassert>
#include <chrono>
#include <iostream>

using Pacer = lsfgvk::layer::FixedOutputPacer;
using namespace std::chrono_literals;

int main() {
    {
        Pacer pacer(120);
        const auto t0 = Pacer::TimePoint{};
        assert(pacer.delayUntilNext(t0) == Pacer::Duration::zero());
        pacer.markPresented(t0);

        const auto half = t0 + pacer.period() / 2;
        const auto delay = pacer.delayUntilNext(half);
        assert(delay > Pacer::Duration::zero());
        assert(delay <= pacer.period());

        const auto due = t0 + pacer.period();
        assert(pacer.delayUntilNext(due) == Pacer::Duration::zero());
        pacer.markPresented(due);
        assert(pacer.delayUntilNext(due) == pacer.period());
    }

    {
        Pacer pacer(240);
        const auto t0 = Pacer::TimePoint{};
        pacer.markPresented(t0);
        const auto oldPeriod = pacer.period();
        assert(oldPeriod > 4ms && oldPeriod < 5ms);

        // A long stall rebases instead of creating a catch-up burst.
        const auto stalled = t0 + 100ms;
        assert(pacer.delayUntilNext(stalled) == Pacer::Duration::zero());
        pacer.markPresented(stalled);
        assert(pacer.delayUntilNext(stalled) == pacer.period());
    }

    {
        Pacer pacer(60);
        const auto t0 = Pacer::TimePoint{};
        pacer.markPresented(t0);
        pacer.reset(165);
        assert(pacer.delayUntilNext(t0 + 1ms) == Pacer::Duration::zero());
        assert(pacer.period() > 6ms && pacer.period() < 7ms);
    }

    {
        Pacer pacer(0);
        const auto t0 = Pacer::TimePoint{};
        assert(pacer.delayUntilNext(t0) == Pacer::Duration::zero());
        pacer.markPresented(t0);
        assert(pacer.delayUntilNext(t0 + 1s) == Pacer::Duration::zero());
    }

    std::cout << "All FixedOutputPacer tests passed.\n";
    return 0;
}
