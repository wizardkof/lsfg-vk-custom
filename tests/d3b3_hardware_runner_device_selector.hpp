/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/vulkan/physical_device.hpp"

#include <stdexcept>
#include <string>

namespace lsfgvk::tests {

[[nodiscard]] inline std::string makeHardwareRunnerDeviceSelector(
        const vk::PhysicalDeviceIdentity& selectedDevice) {
    if (selectedDevice.pci.has_value())
        return selectedDevice.pci->identifier();

    if (selectedDevice.vendorId != 0 && selectedDevice.deviceId != 0)
        return selectedDevice.vendorDeviceIdentifier();

    if (!selectedDevice.name.empty())
        return selectedDevice.name;

    throw std::invalid_argument(
        "selected hardware-runner device has no stable selector identity");
}

} // namespace lsfgvk::tests
