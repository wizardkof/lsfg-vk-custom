/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "../lsfg-vk-layer/src/config_snapshot.hpp"

#include <cassert>
#include <optional>
#include <string>

using lsfgvk::layer::ConfigSnapshotState;

int main() {
    ConfigSnapshotState state;

    ls::GlobalConf inactiveGlobal{};
    inactiveGlobal.allow_fp16 = false;
    assert(state.select(std::nullopt, inactiveGlobal)
        == ConfigSnapshotState::Selection::Inactive);
    assert(!state.snapshot().active());
    assert(state.snapshot().revision == 0);

    ls::GameConf adaptive{};
    adaptive.name = "Adaptive 1x";
    adaptive.frame_generation_mode = ls::FrameGenerationMode::Adaptive;
    adaptive.multiplier = 1;
    ls::GlobalConf adaptiveGlobal{};
    adaptiveGlobal.allow_fp16 = true;

    assert(state.select(adaptive, adaptiveGlobal)
        == ConfigSnapshotState::Selection::Applied);
    const auto adaptiveSnapshot = state.snapshot();
    assert(adaptiveSnapshot.revision == 1);
    assert(adaptiveSnapshot.active());
    assert(!adaptiveSnapshot.fixedMode());
    assert(adaptiveSnapshot.adaptiveBypass());
    assert(adaptiveSnapshot.global.allow_fp16);

    ls::GameConf fixed{};
    fixed.name = "Fixed 60";
    fixed.frame_generation_mode = ls::FrameGenerationMode::Fixed;
    fixed.multiplier = 1;
    fixed.target_fps = 60;
    ls::GlobalConf fixedGlobal{};
    fixedGlobal.allow_fp16 = false;
    fixedGlobal.dll = "/safe/Lossless.dll";

    assert(state.select(fixed, fixedGlobal)
        == ConfigSnapshotState::Selection::Applied);
    const auto fixedSnapshot = state.snapshot();
    assert(fixedSnapshot.revision == 2);
    assert(fixedSnapshot.fixedMode());
    assert(!fixedSnapshot.adaptiveBypass());
    assert(fixedSnapshot.profile->target_fps == 60);
    assert(fixedSnapshot.global.dll == std::optional<std::string>("/safe/Lossless.dll"));

    // Removing the active profile is not a supported hot-disable operation.
    // Preserve the entire effective snapshot, including its global settings.
    ls::GlobalConf removedGlobal{};
    removedGlobal.allow_fp16 = true;
    removedGlobal.dll = "/changed/Lossless.dll";
    assert(state.select(std::nullopt, removedGlobal)
        == ConfigSnapshotState::Selection::RetainedActiveProfile);
    const auto retainedSnapshot = state.snapshot();
    assert(retainedSnapshot.revision == fixedSnapshot.revision);
    assert(retainedSnapshot.fixedMode());
    assert(retainedSnapshot.profile->name == fixedSnapshot.profile->name);
    assert(retainedSnapshot.global.dll == fixedSnapshot.global.dll);

    adaptive.multiplier = 2;
    assert(state.select(adaptive, adaptiveGlobal)
        == ConfigSnapshotState::Selection::Applied);
    const auto reactivatedSnapshot = state.snapshot();
    assert(reactivatedSnapshot.revision == 3);
    assert(!reactivatedSnapshot.fixedMode());
    assert(!reactivatedSnapshot.adaptiveBypass());

    return 0;
}
