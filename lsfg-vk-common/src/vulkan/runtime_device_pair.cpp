/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"

#include <optional>
#include <string>
#include <vector>

using namespace vk;

namespace {
    [[nodiscard]] RuntimeDevicePair makeRuntimeDevicePair(
            const PhysicalDeviceIdentity& renderDevice,
            const PhysicalDeviceIdentity& generationDevice,
            const std::optional<std::string>& generationSelector) {
        return {
            .render = {
                .identity = renderDevice,
                .ownership = RuntimeDeviceOwnership::ApplicationManaged
            },
            .generation = {
                .identity = generationDevice,
                .ownership = RuntimeDeviceOwnership::BackendManaged
            },
            .mode = renderDevice.samePhysicalDevice(generationDevice)
                ? RuntimeDevicePairMode::SamePhysicalDevice
                : RuntimeDevicePairMode::CrossPhysicalDevice,
            .requestedGenerationSelector = generationSelector
        };
    }
}

bool RuntimeDevicePair::matchesRenderDevice(
        const PhysicalDeviceIdentity& candidate) const {
    return this->render.identity.samePhysicalDevice(candidate);
}

bool RuntimeDevicePair::matchesGenerationDevice(
        const PhysicalDeviceIdentity& candidate) const {
    return this->generation.identity.samePhysicalDevice(candidate);
}

RuntimeDevicePairResolution vk::resolveRuntimeDevicePair(
        const std::vector<PhysicalDeviceSnapshot>& backendDevices,
        const PhysicalDeviceIdentity& renderDevice,
        const std::optional<std::string>& generationSelector) {
    RuntimeDevicePairResolution result{
        .selection = resolveDeviceSelection(
            backendDevices, renderDevice, generationSelector),
        .pair = std::nullopt
    };

    if (!result.selection.selectedIndex.has_value())
        return result;

    const auto selectedIndex = *result.selection.selectedIndex;
    if (selectedIndex >= backendDevices.size())
        return result;

    result.pair = makeRuntimeDevicePair(
        renderDevice,
        backendDevices.at(selectedIndex).identity,
        generationSelector);
    return result;
}

std::optional<RuntimeDevicePair> vk::bindRuntimeDevicePair(
        const PhysicalDeviceIdentity& renderDevice,
        const PhysicalDeviceIdentity& generationDevice,
        const std::optional<std::string>& generationSelector) {
    if (generationSelector.has_value()) {
        if (!generationDevice.matchesSelector(*generationSelector))
            return std::nullopt;
    } else if (!generationDevice.samePhysicalDevice(renderDevice)) {
        return std::nullopt;
    }

    return makeRuntimeDevicePair(
        renderDevice, generationDevice, generationSelector);
}
