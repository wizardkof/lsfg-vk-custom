/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/configuration/config.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace lsfgvk::layer {

    /// Immutable configuration view used by one Vulkan-layer transaction.
    // This is intentionally a value object with public fields: one snapshot is
    // copied at the transaction boundary and then treated as immutable.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    struct ConfigSnapshot {
        uint64_t revision{};
        std::optional<ls::GameConf> profile;
        ls::GlobalConf global;

        [[nodiscard]] bool active() const { return this->profile.has_value(); }
        [[nodiscard]] const ls::GameConf& activeProfile() const {
            if (!this->profile.has_value())
                throw std::logic_error("inactive configuration snapshot has no profile");
            return *this->profile;
        }
        [[nodiscard]] bool fixedMode() const {
            return this->profile.has_value()
                && this->profile->frame_generation_mode == ls::FrameGenerationMode::Fixed;
        }
        [[nodiscard]] bool adaptiveBypass() const {
            return this->profile.has_value()
                && this->profile->frame_generation_mode == ls::FrameGenerationMode::Adaptive
                && this->profile->multiplier == 1;
        }
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    /// Selects effective snapshots while preserving an active profile when it
    /// disappears from a hot-reloaded file. Existing virtual swapchains cannot
    /// become passthrough without changing the VkImages known to the application.
    class ConfigSnapshotState {
    public:
        enum class Selection : std::uint8_t {
            Applied,
            Inactive,
            RetainedActiveProfile
        };

        [[nodiscard]] Selection select(
                std::optional<ls::GameConf> profile,
                ls::GlobalConf global) {
            if (!profile.has_value()) {
                if (this->current.profile.has_value())
                    return Selection::RetainedActiveProfile;

                this->current.global = std::move(global);
                return Selection::Inactive;
            }

            this->current.revision++;
            this->current.profile = std::move(profile);
            this->current.global = std::move(global);
            return Selection::Applied;
        }

        [[nodiscard]] ConfigSnapshot snapshot() const { return this->current; }

    private:
        ConfigSnapshot current;
    };

}
