/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "fixed_frame_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace lsfgvk::layer;

namespace {
constexpr auto MAX_SOURCE_INTERVAL = std::chrono::milliseconds(250);
constexpr double EPSILON = 1e-9;
}

FixedFrameScheduler::FixedFrameScheduler(
        uint32_t targetFps,
        size_t maxGeneratedFrames) :
    m_targetFps(targetFps),
    m_maxGeneratedFrames(std::max<size_t>(1, maxGeneratedFrames)) {}

void FixedFrameScheduler::reset(uint32_t targetFps) {
    m_targetFps = targetFps;
    m_generationCredit = 0.0;
}

FixedFrameScheduler::Plan FixedFrameScheduler::plan(Duration sourceInterval) {
    Plan result{};

    if (!m_targetFps)
        return result;

    if (sourceInterval <= Duration::zero() || sourceInterval > MAX_SOURCE_INTERVAL) {
        m_generationCredit = 0.0;
        result.discontinuity = true;
        return result;
    }

    const double targetPeriodSeconds = 1.0 / static_cast<double>(m_targetFps);
    const double sourceSeconds =
        std::chrono::duration<double>(sourceInterval).count();

    result.measuredSourceFps = 1.0 / sourceSeconds;

    if (sourceSeconds < targetPeriodSeconds) {
        result.sourceDelay = std::chrono::duration_cast<Duration>(
            std::chrono::duration<double>(targetPeriodSeconds - sourceSeconds));
        m_generationCredit = 0.0;
        return result;
    }

    const double exactGenerated =
        std::max(0.0, static_cast<double>(m_targetFps) * sourceSeconds - 1.0);

    m_generationCredit += exactGenerated;

    size_t generated =
        static_cast<size_t>(std::floor(m_generationCredit + EPSILON));

    if (generated > m_maxGeneratedFrames) {
        generated = m_maxGeneratedFrames;
        result.clamped = true;
        m_generationCredit = 0.0;
    } else {
        m_generationCredit -= static_cast<double>(generated);
        m_generationCredit = std::clamp(m_generationCredit, 0.0, 1.0);
    }

    result.timestamps.reserve(generated);
    for (size_t i = 0; i < generated; ++i) {
        result.timestamps.push_back(
            static_cast<float>(i + 1) /
            static_cast<float>(generated + 1));
    }

    return result;
}
