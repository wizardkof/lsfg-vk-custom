/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_production_seams.hpp"

#include <cstdint>

namespace lsfgvk::layer {

enum class D3B3RuntimeFrameResult : uint8_t {
    ACCEPTED,
    TEMPORARILY_BLOCKED,
    FAILED,
    DEVICE_LOST,
    COMPLETED
};

/// Per-swapchain-generation owner of the finite D3B3 controller and every
/// operation binding needed by that controller. Construction is inert: GPU
/// work starts only when processFrame() accepts a logical application frame.
class D3B3ProductionRuntimeSession {
public:
    D3B3ProductionRuntimeSession(uint64_t swapchainGeneration,
        D3B3FiniteProductionOperations operations);

    [[nodiscard]] bool structurallyReady() const noexcept { return ready; }
    [[nodiscard]] uint64_t generation() const noexcept { return generationId; }
    [[nodiscard]] bool ownsFrame() const noexcept { return frameOwnershipAccepted; }
    [[nodiscard]] D3B3ProductionState& controller() noexcept { return state; }
    [[nodiscard]] const D3B3ProductionState& controller() const noexcept { return state; }

    D3B3RuntimeFrameResult processFrame(uint64_t frameId) noexcept;

private:
    uint64_t generationId{};
    D3B3ProductionState state;
    bool ready{};
    bool frameOwnershipAccepted{};
};

}
