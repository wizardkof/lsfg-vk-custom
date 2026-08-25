/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_production_runtime.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <utility>

namespace lsfgvk::layer {

D3B3ProductionRuntimeSession::D3B3ProductionRuntimeSession(
        uint64_t swapchainGeneration, D3B3FiniteProductionOperations operations) :
    generationId(swapchainGeneration) {
    if (generationId == 0)
        return;
    try {
        state.configureFinite(std::move(operations));
        ready = true;
    } catch (...) {
        ready = false;
    }
}

D3B3RuntimeFrameResult D3B3ProductionRuntimeSession::processFrame(
        uint64_t frameId) noexcept {
    if (!ready)
        return D3B3RuntimeFrameResult::FAILED;
    if (state.finiteStopped())
        return D3B3RuntimeFrameResult::COMPLETED;
    try {
        if (state.presentFinite(frameId)) {
            frameOwnershipAccepted = true;
            return state.finiteStopped()
                ? D3B3RuntimeFrameResult::COMPLETED
                : D3B3RuntimeFrameResult::ACCEPTED;
        }
        switch (state.finiteState()) {
        case D3B3FiniteProductionState::WAITING_REUSE:
            return D3B3RuntimeFrameResult::TEMPORARILY_BLOCKED;
        case D3B3FiniteProductionState::FAILED:
            return D3B3RuntimeFrameResult::FAILED;
        case D3B3FiniteProductionState::FINITE_STOPPED:
            return D3B3RuntimeFrameResult::COMPLETED;
        default:
            return D3B3RuntimeFrameResult::FAILED;
        }
    } catch (const ls::vulkan_error& error) {
        return error.error() == VK_ERROR_DEVICE_LOST
            ? D3B3RuntimeFrameResult::DEVICE_LOST
            : D3B3RuntimeFrameResult::FAILED;
    } catch (...) {
        return D3B3RuntimeFrameResult::FAILED;
    }
}

}
