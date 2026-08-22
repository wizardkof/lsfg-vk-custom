/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <stdexcept>

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
        [[nodiscard]] RuntimeGenerateDiagnosticStep advance() {
            if (failed)
                throw std::logic_error("D2 diagnostic session failed and cannot continue");
            if (directFrames >= 3)
                throw std::logic_error("D2 diagnostic session is already complete");

            ++directFrames;
            if (directFrames == 1)
                return {RuntimeGenerateDiagnosticAction::SEED_ONLY, 1, 0, 0, 0, 0, 0};

            ++rotations;
            if (directFrames == 2)
                return {RuntimeGenerateDiagnosticAction::GAMMA_DELTA, 2, 1, 1, 0, 1, 1};

            ++generateExecutions;
            return {RuntimeGenerateDiagnosticAction::GENERATE, 3, 2, 0, 1, 0, 0};
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
}
