/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lsfgvk::layer {

class FixedFrameScheduler {
public:
    using Duration = std::chrono::nanoseconds;

    struct Plan {
        Duration sourceDelay{};
        std::vector<float> timestamps;
        double measuredSourceFps{};
        bool clamped{};
        bool discontinuity{};
    };

    static constexpr size_t DEFAULT_MAX_GENERATED_FRAMES = 7;

    explicit FixedFrameScheduler(
        uint32_t targetFps = 0,
        size_t maxGeneratedFrames = DEFAULT_MAX_GENERATED_FRAMES);

    void reset(uint32_t targetFps);

    [[nodiscard]] Plan plan(Duration sourceInterval);

    [[nodiscard]] uint32_t targetFps() const { return m_targetFps; }
    [[nodiscard]] size_t maxGeneratedFrames() const { return m_maxGeneratedFrames; }

private:
    uint32_t m_targetFps{};
    size_t m_maxGeneratedFrames{DEFAULT_MAX_GENERATED_FRAMES};
    double m_generationCredit{};
};

}
