/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command_buffer.hpp"
#include "exchange_image_sync.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace vk {
    [[nodiscard]] constexpr bool destinationReturnCardinalityValid(
            size_t destinationCount, size_t returnFdCount) noexcept {
        return destinationCount == returnFdCount;
    }

    struct DestinationReturnBackendState {
        bool firstUse{true};
        uint64_t nextGeneration{1};

        [[nodiscard]] bool requiresReturnWait() const noexcept {
            return !firstUse;
        }
        [[nodiscard]] uint64_t expectedReturnGeneration() const noexcept {
            return nextGeneration - 1;
        }
        [[nodiscard]] uint64_t pendingWriteGeneration() const noexcept {
            return nextGeneration;
        }
        void backendWriteAccepted() noexcept {
            firstUse = false;
            ++nextGeneration;
        }
    };

    struct DestinationReturnLayerState {
        uint64_t nextGeneration{1};

        [[nodiscard]] uint64_t pendingReturnGeneration() const noexcept {
            return nextGeneration;
        }
        void returnSubmitAccepted() noexcept { ++nextGeneration; }
    };

    struct BackendDestinationSubmitStorage {
        std::array<TimelineWait, 2> waits{};
        std::array<TimelineSignal, 1> signals{};
        CommandBufferSubmit submission{};
    };

    inline void prepareBackendDestinationSubmit(
            VkSemaphore prepassSemaphore, uint64_t sourceReady,
            VkSemaphore destinationReturnSemaphore,
            const DestinationReturnBackendState& returnState,
            VkSemaphore syncSemaphore, uint64_t destinationReady,
            BackendDestinationSubmitStorage& storage) noexcept {
        storage.waits[0] = TimelineWait{prepassSemaphore, sourceReady,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        size_t waitCount = 1;
        if (returnState.requiresReturnWait()) {
            storage.waits[1] = TimelineWait{destinationReturnSemaphore,
                returnState.expectedReturnGeneration(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
            waitCount = 2;
        }
        storage.signals[0] = TimelineSignal{syncSemaphore, destinationReady};
        storage.submission = CommandBufferSubmit{
            .timelineWaits = std::span<const TimelineWait>(
                storage.waits.data(), waitCount),
            .timelineSignals = storage.signals};
    }

    struct LayerDestinationReturnSubmitStorage {
        std::array<VkPipelineStageFlags, 2> binaryWaitStages{};
        std::array<TimelineWait, 1> waits{};
        std::array<TimelineSignal, 1> signals{};
        CommandBufferSubmit submission{};
    };

    inline void prepareLayerDestinationReturnSubmit(
            std::span<const VkSemaphore> binaryWaits,
            std::span<const VkSemaphore> binarySignals,
            VkSemaphore syncSemaphore, uint64_t destinationReady,
            VkSemaphore destinationReturnSemaphore,
            const DestinationReturnLayerState& returnState,
            LayerDestinationReturnSubmitStorage& storage) noexcept {
        for (size_t i = 0; i < binaryWaits.size(); ++i)
            storage.binaryWaitStages[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        storage.waits[0] = TimelineWait{syncSemaphore, destinationReady,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        storage.signals[0] = TimelineSignal{destinationReturnSemaphore,
            returnState.pendingReturnGeneration()};
        storage.submission = CommandBufferSubmit{
            .binaryWaits = binaryWaits,
            .binaryWaitStages = std::span<const VkPipelineStageFlags>(
                storage.binaryWaitStages.data(), binaryWaits.size()),
            .timelineWaits = storage.waits,
            .binarySignals = binarySignals,
            .timelineSignals = storage.signals};
    }
}
