/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {
    vk::DeviceUuid uuid(uint8_t suffix) {
        vk::DeviceUuid value{};
        value.back() = suffix;
        return value;
    }

    vk::PhysicalDeviceIdentity identity(
            const std::string& name,
            uint8_t deviceUuid,
            uint32_t vendorId,
            uint32_t deviceId,
            uint32_t bus) {
        return {
            .name = name,
            .vendorId = vendorId,
            .deviceId = deviceId,
            .deviceUuid = uuid(deviceUuid),
            .driverUuid = uuid(static_cast<uint8_t>(deviceUuid + 32U)),
            .pci = vk::PciBusInfo{
                .domain = 0,
                .bus = bus,
                .device = 0,
                .function = 0
            }
        };
    }

    vk::PhysicalDeviceSnapshot snapshot(const vk::PhysicalDeviceIdentity& value) {
        return {.identity = value, .advertisedDeviceExtensions = {}};
    }
}

int main() {
    const auto render = identity("Render GPU", 1, 0x1002, 0x164E, 0x0C);
    const auto generation = identity("Generation GPU", 2, 0x10DE, 0x2504, 0x01);
    const std::vector<vk::PhysicalDeviceSnapshot> devices{
        snapshot(generation), snapshot(render)};

    // Default always resolves back to the application's physical GPU even
    // when backend enumeration order puts another device first.
    auto resolved = vk::resolveRuntimeDevicePair(devices, render, std::nullopt);
    assert(resolved.selection.resolution == vk::DeviceSelectionResolution::Selected);
    assert(resolved.selection.selectedIndex == 1);
    assert(resolved.selection.matchCount == 1);
    assert(resolved.pair.has_value());
    assert(!resolved.pair->crossDevice());
    assert(resolved.pair->mode == vk::RuntimeDevicePairMode::SamePhysicalDevice);
    assert(resolved.pair->render.ownership
        == vk::RuntimeDeviceOwnership::ApplicationManaged);
    assert(resolved.pair->generation.ownership
        == vk::RuntimeDeviceOwnership::BackendManaged);
    assert(resolved.pair->matchesRenderDevice(render));
    assert(resolved.pair->matchesGenerationDevice(render));
    assert(!resolved.pair->requestedGenerationSelector.has_value());

    // An explicit selector is the opt-in for a cross-physical-device pair.
    resolved = vk::resolveRuntimeDevicePair(
        devices, render, std::optional<std::string>{"Generation GPU"});
    assert(resolved.selection.resolution == vk::DeviceSelectionResolution::Selected);
    assert(resolved.selection.selectedIndex == 0);
    assert(resolved.pair.has_value());
    assert(resolved.pair->crossDevice());
    assert(resolved.pair->mode == vk::RuntimeDevicePairMode::CrossPhysicalDevice);
    assert(resolved.pair->matchesRenderDevice(render));
    assert(resolved.pair->matchesGenerationDevice(generation));
    assert(resolved.pair->requestedGenerationSelector == "Generation GPU");

    // The same stable selectors accepted by DG1 remain valid for generation.
    resolved = vk::resolveRuntimeDevicePair(
        devices, render,
        std::optional<std::string>{generation.pci->identifier()});
    assert(resolved.pair.has_value());
    assert(resolved.pair->matchesGenerationDevice(generation));
    resolved = vk::resolveRuntimeDevicePair(
        devices, render,
        std::optional<std::string>{generation.vendorDeviceIdentifier()});
    assert(resolved.pair.has_value());
    assert(resolved.pair->matchesGenerationDevice(generation));

    // Missing selections produce no pair and preserve the detailed resolution.
    resolved = vk::resolveRuntimeDevicePair(
        {snapshot(generation)}, render, std::nullopt);
    assert(resolved.selection.resolution
        == vk::DeviceSelectionResolution::ApplicationDeviceNotVisibleToBackend);
    assert(!resolved.pair.has_value());

    resolved = vk::resolveRuntimeDevicePair(
        devices, render, std::optional<std::string>{"Missing GPU"});
    assert(resolved.selection.resolution
        == vk::DeviceSelectionResolution::SelectorNotFoundInBackend);
    assert(!resolved.pair.has_value());

    // Preserve the existing explicit-selector first-match behavior while
    // surfacing ambiguity in the selection result.
    auto duplicate = generation;
    duplicate.deviceUuid = uuid(3);
    duplicate.pci = vk::PciBusInfo{
        .domain = 0,
        .bus = 2,
        .device = 0,
        .function = 0
    };
    resolved = vk::resolveRuntimeDevicePair(
        {snapshot(generation), snapshot(duplicate)}, render,
        std::optional<std::string>{"Generation GPU"});
    assert(resolved.selection.resolution
        == vk::DeviceSelectionResolution::SelectorAmbiguous);
    assert(resolved.selection.matchCount == 2);
    assert(resolved.selection.selectedIndex == 0);
    assert(resolved.pair.has_value());
    assert(resolved.pair->matchesGenerationDevice(generation));
    assert(!resolved.pair->matchesGenerationDevice(duplicate));

    // Post-creation binding validates the actually selected backend device.
    auto bound = vk::bindRuntimeDevicePair(render, render, std::nullopt);
    assert(bound.has_value());
    assert(!bound->crossDevice());

    bound = vk::bindRuntimeDevicePair(render, generation, std::nullopt);
    assert(!bound.has_value());

    bound = vk::bindRuntimeDevicePair(
        render, generation, std::optional<std::string>{"Generation GPU"});
    assert(bound.has_value());
    assert(bound->crossDevice());
    assert(bound->generation.ownership == vk::RuntimeDeviceOwnership::BackendManaged);

    bound = vk::bindRuntimeDevicePair(
        render, generation, std::optional<std::string>{"Render GPU"});
    assert(!bound.has_value());

    return 0;
}
