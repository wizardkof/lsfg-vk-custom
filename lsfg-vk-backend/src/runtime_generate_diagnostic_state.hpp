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
        }

        void recordFailure() { failed = true; }

        [[nodiscard]] bool finalPassReady() const {
            return directFrames == 3 && rotations == 2
                && generateExecutions == 1 && outputValidated;
        }

        [[nodiscard]] size_t generatedFrames() const { return generateExecutions; }

    private:
        size_t directFrames{};
        size_t rotations{};
        size_t generateExecutions{};
        bool outputValidated{};
        bool failed{};
    };
}
