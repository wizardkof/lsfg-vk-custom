/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace lsfgvk::layer {

class FixedOutputPacer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;

    explicit FixedOutputPacer(uint32_t targetFps = 0);

    void reset(uint32_t targetFps);

    /// Return the host-side delay before the next output present. The caller
    /// should call markPresented() after a successful vkQueuePresentKHR.
    [[nodiscard]] Duration delayUntilNext(TimePoint now) const;

    /// Advance the target timeline after an output present. Long stalls are
    /// rebased instead of causing a burst of catch-up presents.
    void markPresented(TimePoint now);

    [[nodiscard]] uint32_t targetFps() const { return m_targetFps; }
    [[nodiscard]] Duration period() const { return m_period; }

private:
    uint32_t m_targetFps{};
    Duration m_period{};
    std::optional<TimePoint> m_nextDeadline;
};

}
