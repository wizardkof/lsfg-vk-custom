/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <cstdint>

namespace lsfgvk::backend {
    enum class RuntimeGenerateDiagnosticAction {
        SEED_ONLY,
        GAMMA_DELTA,
        GENERATE
    };

    struct RuntimeGenerateDiagnosticStep {
        RuntimeGenerateDiagnosticAction action{};
        size_t directFrames{};
        size_t rotations{};
        size_t frameIndex{};
        size_t olderSlot{};
        size_t newerSlot{};
        size_t generateDescriptorSet{};
    };

    class RuntimeGenerateDiagnosticState {
    public:
        [[nodiscard]] RuntimeGenerateDiagnosticStep nextStep() const {
            if (failed)
                throw std::logic_error("D2 diagnostic session failed and cannot continue");
            if (directFrames >= 3)
                throw std::logic_error("D2 diagnostic session is already complete");
            if (directFrames == 0)
                return {RuntimeGenerateDiagnosticAction::SEED_ONLY, 1, 0, 0, 0, 0, 0};
            if (directFrames == 1)
                return {RuntimeGenerateDiagnosticAction::GAMMA_DELTA, 2, 1, 1, 0, 1, 1};
            return {RuntimeGenerateDiagnosticAction::GENERATE, 3, 2, 0, 1, 0, 0};
        }

        [[nodiscard]] RuntimeGenerateDiagnosticStep advance() {
            const auto step = nextStep();
            directFrames = step.directFrames;
            rotations = step.rotations;
            if (step.action == RuntimeGenerateDiagnosticAction::GENERATE)
                ++generateExecutions;
            return step;
        }

        void recordValidatedOutput() {
            if (failed)
                throw std::logic_error("D2 failed session cannot validate output");
            if (directFrames != 3 || generateExecutions != 1)
                throw std::logic_error("D2 output validation occurred outside Generate");
            outputValidated = true;
            generationPending = false;
        }

        void recordSubmittedGeneration() {
            if (failed || generateExecutions != 1 || generationPending || outputValidated)
                throw std::logic_error("invalid D2 pending-generation submission");
            generationPending = true;
        }

        [[nodiscard]] bool pendingGeneration() const noexcept { return generationPending; }

        void recordFailure() { failed = true; }

        void issueCompletionResult() {
            if (!finalPassReady())
                throw std::logic_error("D2 completion result unavailable");
            if (completionResultIssued)
                throw std::logic_error("D2 completion result was already issued");
            completionResultIssued = true;
        }

        [[nodiscard]] bool completionResultAvailable() const {
            return finalPassReady() && !completionResultIssued;
        }

        [[nodiscard]] bool finalPassReady() const {
            return !failed && directFrames == 3 && rotations == 2
                && generateExecutions == 1 && outputValidated;
        }

        [[nodiscard]] size_t generatedFrames() const { return generateExecutions; }

    private:
        size_t directFrames{};
        size_t rotations{};
        size_t generateExecutions{};
        bool outputValidated{};
        bool generationPending{};
        bool failed{};
        bool completionResultIssued{};
    };

    enum class RuntimeGenerateSerialPhase {
        READY, IN_FLIGHT, SOURCE_READS_RETIRED, OUTPUT_IN_USE, READY_NEXT, FAILED
    };

    class RuntimeGenerateSerialState {
    public:
        void submit(uint64_t generationId) {
            if (phase != RuntimeGenerateSerialPhase::READY
                    && phase != RuntimeGenerateSerialPhase::READY_NEXT)
                throw std::logic_error("serial Generate already has an active operation");
            if (generationId == 0 || generationId <= epoch)
                throw std::logic_error("stale serial Generate epoch");
            epoch = generationId;
            phase = RuntimeGenerateSerialPhase::IN_FLIGHT;
        }
        void sourceReadsRetired(uint64_t generationId) {
            requireCurrent(generationId);
            if (phase != RuntimeGenerateSerialPhase::IN_FLIGHT)
                throw std::logic_error("invalid source-read retirement");
            phase = RuntimeGenerateSerialPhase::SOURCE_READS_RETIRED;
        }
        void outputInUse(uint64_t generationId) {
            requireCurrent(generationId);
            if (phase != RuntimeGenerateSerialPhase::IN_FLIGHT
                    && phase != RuntimeGenerateSerialPhase::SOURCE_READS_RETIRED)
                throw std::logic_error("invalid generated-output transition");
            phase = RuntimeGenerateSerialPhase::OUTPUT_IN_USE;
        }
        void retireOperation(uint64_t generationId) {
            requireCurrent(generationId);
            if (phase != RuntimeGenerateSerialPhase::SOURCE_READS_RETIRED
                    && phase != RuntimeGenerateSerialPhase::OUTPUT_IN_USE)
                throw std::logic_error("operation retired before bounded completion");
            phase = RuntimeGenerateSerialPhase::READY_NEXT;
        }
        void fail() noexcept { phase = RuntimeGenerateSerialPhase::FAILED; }
        [[nodiscard]] RuntimeGenerateSerialPhase currentPhase() const noexcept { return phase; }
        [[nodiscard]] uint64_t currentEpoch() const noexcept { return epoch; }
        [[nodiscard]] bool canSubmit() const noexcept {
            return phase == RuntimeGenerateSerialPhase::READY
                || phase == RuntimeGenerateSerialPhase::READY_NEXT;
        }
    private:
        void requireCurrent(uint64_t generationId) const {
            if (generationId == 0 || generationId != epoch)
                throw std::logic_error("stale serial Generate authority");
            if (phase == RuntimeGenerateSerialPhase::FAILED)
                throw std::logic_error("failed serial Generate session");
        }
        RuntimeGenerateSerialPhase phase{RuntimeGenerateSerialPhase::READY};
        uint64_t epoch{};
    };
}
