/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/vulkan/physical_device.hpp"

#include <functional>
#include <iosfwd>
#include <vector>

namespace lsfgvk::cli::devices {

    using SnapshotProvider = std::function<std::vector<vk::PhysicalDeviceSnapshot>()>;

    /// Render a standalone Vulkan physical-device inventory.
    void render(std::ostream& output,
        const std::vector<vk::PhysicalDeviceSnapshot>& snapshots);

    /// Run with an injected snapshot provider for deterministic testing.
    int run(const SnapshotProvider& provider, std::ostream& output, std::ostream& errors);

    /// Run the standalone Vulkan inventory command.
    int run();

}
