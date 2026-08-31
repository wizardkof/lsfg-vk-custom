/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan_core.h>

namespace vk {
    struct PreparedFrameSubmitPhase {
        size_t index{}; // 0 = prepass, generated phases are 1..N
        bool terminal{};
    };

    template <typename Submit, typename Accepted>
    void executePreparedFrameSchedule(size_t generatedFrames,
            Submit&& submit, Accepted&& accepted) {
        const PreparedFrameSubmitPhase prepass{0, generatedFrames == 0};
        submit(prepass);
        accepted(prepass);
        for (size_t i = 0; i < generatedFrames; ++i) {
            const PreparedFrameSubmitPhase generated{i + 1,
                i + 1 == generatedFrames};
            submit(generated);
            accepted(generated);
        }
    }

    enum class GenerationSlotPhase : uint8_t {
        Available,
        Reserved,
        InFlight,
        Retained
    };

    class GenerationSlotAuthority {
    public:
        [[nodiscard]] GenerationSlotPhase phase() const noexcept { return state; }
        [[nodiscard]] bool reserve() noexcept {
            if (state != GenerationSlotPhase::Available) return false;
            state = GenerationSlotPhase::Reserved;
            acceptedSubmit = false;
            terminalSubmit = false;
            return true;
        }
        void abort() noexcept {
            if (state == GenerationSlotPhase::Reserved)
                state = GenerationSlotPhase::Available;
        }
        void submitAccepted(bool terminal) noexcept {
            if (state == GenerationSlotPhase::Reserved
                    || state == GenerationSlotPhase::InFlight) {
                state = GenerationSlotPhase::InFlight;
                acceptedSubmit = true;
                terminalSubmit = terminalSubmit || terminal;
            }
        }
        void submitFailed(bool deviceLost) noexcept {
            if (deviceLost || acceptedSubmit)
                state = GenerationSlotPhase::Retained;
            else
                state = GenerationSlotPhase::Available;
        }
        [[nodiscard]] bool retire(bool backendComplete,
                bool destinationReturnsComplete) noexcept {
            if (state != GenerationSlotPhase::InFlight || !terminalSubmit
                    || !backendComplete || !destinationReturnsComplete)
                return false;
            state = GenerationSlotPhase::Available;
            acceptedSubmit = false;
            terminalSubmit = false;
            return true;
        }
        [[nodiscard]] bool terminalSubmitted() const noexcept {
            return terminalSubmit;
        }
    private:
        GenerationSlotPhase state{GenerationSlotPhase::Available};
        bool acceptedSubmit{};
        bool terminalSubmit{};
    };

    [[nodiscard]] inline bool applyPreparedFrameSubmitFailure(
            GenerationSlotAuthority& slot, bool anySubmitAccepted,
            VkResult result) noexcept {
        const bool deviceLost = result == VK_ERROR_DEVICE_LOST;
        slot.submitFailed(deviceLost);
        return anySubmitAccepted || deviceLost;
    }
}
