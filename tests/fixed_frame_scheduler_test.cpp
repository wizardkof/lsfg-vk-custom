/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "../lsfg-vk-layer/src/fixed_frame_scheduler.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using Scheduler = lsfgvk::layer::FixedFrameScheduler;
using namespace std::chrono_literals;

namespace {

struct Simulation {
    size_t real{};
    size_t generated{};
    size_t delayed{};
    size_t clamped{};
};

Scheduler::Duration period(double fps) {
    return std::chrono::duration_cast<Scheduler::Duration>(
        std::chrono::duration<double>(1.0 / fps));
}

Simulation simulate(
        double sourceFps,
        uint32_t targetFps,
        double seconds,
        size_t maxGenerated = Scheduler::DEFAULT_MAX_GENERATED_FRAMES) {
    Scheduler scheduler(targetFps, maxGenerated);
    Simulation result{};
    const size_t frames =
        static_cast<size_t>(std::llround(sourceFps * seconds));
    const auto sourcePeriod = period(sourceFps);

    for (size_t i = 0; i < frames; ++i) {
        const auto plan = scheduler.plan(sourcePeriod);
        ++result.real;
        result.generated += plan.timestamps.size();
        result.delayed += plan.sourceDelay > Scheduler::Duration::zero() ? 1U : 0U;
        result.clamped += plan.clamped ? 1U : 0U;

        float previous = 0.0F;
        for (const float timestamp : plan.timestamps) {
            assert(timestamp > 0.0F && timestamp < 1.0F);
            assert(timestamp > previous);
            previous = timestamp;
        }
    }

    return result;
}

void expectNear(double actual, double expected, double tolerance, const char* label) {
    if (std::abs(actual - expected) <= tolerance)
        return;

    std::cerr << label << ": got " << actual
              << ", expected " << expected
              << " +/- " << tolerance << '\n';
    std::abort();
}

void expectOutputRate(double source, uint32_t target, double seconds, double tolerance) {
    const auto result = simulate(source, target, seconds);
    const double output =
        static_cast<double>(result.real + result.generated) / seconds;
    expectNear(output, static_cast<double>(target), tolerance, "output rate");
    assert(result.clamped == 0);
}

}

int main() {
    expectOutputRate(30.0, 60, 20.0, 0.1);
    expectOutputRate(60.0, 120, 20.0, 0.2);
    expectOutputRate(60.0, 240, 20.0, 0.3);

    expectOutputRate(30.0, 75, 40.0, 0.1);
    expectOutputRate(30.0, 144, 40.0, 0.2);
    expectOutputRate(60.0, 165, 40.0, 0.2);
    expectOutputRate(58.0, 60, 60.0, 0.1);

    expectOutputRate(30.0, 240, 20.0, 0.3);

    {
        const auto result = simulate(120.0, 60, 5.0);
        assert(result.generated == 0);
        assert(result.delayed > 0);
    }

    {
        const auto result = simulate(20.0, 240, 5.0);
        assert(result.clamped > 0);
        assert(result.generated <= result.real * 7);
    }

    {
        Scheduler scheduler(144);
        const auto plan = scheduler.plan(1s);
        assert(plan.discontinuity);
        assert(plan.timestamps.empty());
    }

    {
        Scheduler scheduler(0);
        const auto plan = scheduler.plan(16ms);
        assert(plan.timestamps.empty());
        assert(plan.sourceDelay == Scheduler::Duration::zero());
    }

    {
        Scheduler scheduler(75);
        [[maybe_unused]] const auto beforeReset = scheduler.plan(period(30.0));
        scheduler.reset(60);
        const auto plan = scheduler.plan(period(60.0));
        assert(plan.timestamps.empty());
        assert(!plan.clamped);
    }

    {
        Scheduler monitor60(144);
        Scheduler monitor240(144);
        for (int i = 0; i < 500; ++i) {
            const auto a = monitor60.plan(period(60.0));
            const auto b = monitor240.plan(period(60.0));
            assert(a.timestamps == b.timestamps);
            assert(a.sourceDelay == b.sourceDelay);
            assert(a.clamped == b.clamped);
        }
    }

    std::cout << "All FixedFrameScheduler tests passed.\n";
    return 0;
}
