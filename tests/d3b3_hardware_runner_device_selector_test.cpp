/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_hardware_runner_device_selector.hpp"

#include <cassert>
#include <string>

int main() {
    vk::PhysicalDeviceIdentity selected{
        .name = "NVIDIA GeForce RTX 3060",
        .vendorId = 0x10de,
        .deviceId = 0x2487,
        .pci = vk::PciBusInfo{.domain = 0, .bus = 1, .device = 0, .function = 0}};

    const auto pciSelector =
        lsfgvk::tests::makeHardwareRunnerDeviceSelector(selected);
    assert(pciSelector == "0:1:0.0");
    assert(selected.matchesSelector(pciSelector));

    selected.pci.reset();
    const auto vendorDeviceSelector =
        lsfgvk::tests::makeHardwareRunnerDeviceSelector(selected);
    assert(vendorDeviceSelector == "0x10DE:0x2487");
    assert(selected.matchesSelector(vendorDeviceSelector));

    selected.vendorId = 0;
    selected.deviceId = 0;
    const auto nameSelector =
        lsfgvk::tests::makeHardwareRunnerDeviceSelector(selected);
    assert(nameSelector == "NVIDIA GeForce RTX 3060");
    assert(selected.matchesSelector(nameSelector));

    return 0;
}
