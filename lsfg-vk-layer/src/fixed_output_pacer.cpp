/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "fixed_output_pacer.hpp"

#include <chrono>

using namespace lsfgvk::layer;

FixedOutputPacer::FixedOutputPacer(uint32_t targetFps) {
    this->reset(targetFps);
}

void FixedOutputPacer::reset(uint32_t targetFps) {
    m_targetFps = targetFps;
    m_nextDeadline.reset();
    if (!targetFps) {
        m_period = Duration::zero();
        return;
    }

    m_period = std::chrono::duration_cast<Duration>(
        std::chrono::duration<double>(1.0 / static_cast<double>(targetFps)));
}

FixedOutputPacer::Duration FixedOutputPacer::delayUntilNext(TimePoint now) const {
    if (!m_targetFps || !m_nextDeadline.has_value() || now >= *m_nextDeadline)
        return Duration::zero();
    return std::chrono::duration_cast<Duration>(*m_nextDeadline - now);
}

void FixedOutputPacer::markPresented(TimePoint now) {
    if (!m_targetFps)
        return;

    if (!m_nextDeadline.has_value()) {
        m_nextDeadline = now + m_period;
        return;
    }

    // If the worker fell more than one whole output period behind, rebase the
    // schedule instead of immediately issuing multiple catch-up presents.
    if (now > *m_nextDeadline + m_period)
        m_nextDeadline = now + m_period;
    else
        *m_nextDeadline += m_period;
}
