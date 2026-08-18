/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "physical_device.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vk {

    /// Physical-device relationship for one LSFG runtime pairing.
    enum class RuntimeDevicePairMode : uint8_t {
        SamePhysicalDevice,
        CrossPhysicalDevice
    };

    /// Ownership domain for the logical device represented by an endpoint.
    ///
    /// P3C deliberately models ownership without exposing VkDevice handles:
    /// the render logical device is supplied by the application/layer chain,
    /// while the generation logical device remains owned by backend::Instance.
    enum class RuntimeDeviceOwnership : uint8_t {
        ApplicationManaged,
        BackendManaged
    };

    struct RuntimeDeviceEndpoint {
        PhysicalDeviceIdentity identity;
        RuntimeDeviceOwnership ownership;
    };

    /// Stable role binding between the application's render GPU and the
    /// backend generation GPU. This is an identity/lifetime contract only;
    /// P3C does not connect DMA-BUF/SYNC_FD transport to the runtime yet.
    struct RuntimeDevicePair {
        RuntimeDeviceEndpoint render;
        RuntimeDeviceEndpoint generation;
        RuntimeDevicePairMode mode;
        std::optional<std::string> requestedGenerationSelector;

        [[nodiscard]] bool crossDevice() const noexcept {
            return this->mode == RuntimeDevicePairMode::CrossPhysicalDevice;
        }

        [[nodiscard]] bool matchesRenderDevice(
            const PhysicalDeviceIdentity& candidate) const;
        [[nodiscard]] bool matchesGenerationDevice(
            const PhysicalDeviceIdentity& candidate) const;
    };

    /// Result of resolving Default/an explicit selector against the complete
    /// backend-visible inventory. `pair` is populated whenever selection
    /// yields a concrete first-match device, including the legacy ambiguous
    /// selector case whose first-match semantics are intentionally preserved.
    struct RuntimeDevicePairResolution {
        DeviceSelectionResult selection;
        std::optional<RuntimeDevicePair> pair;
    };

    /// Resolve the runtime pair before backend logical-device creation.
    [[nodiscard]] RuntimeDevicePairResolution resolveRuntimeDevicePair(
        const std::vector<PhysicalDeviceSnapshot>& backendDevices,
        const PhysicalDeviceIdentity& renderDevice,
        const std::optional<std::string>& generationSelector
    );

    /// Bind a runtime pair after backend creation using the actually selected
    /// generation identity. Returns empty when the backend identity violates
    /// the requested role contract (Default must be the render physical GPU;
    /// an explicit selector must match the backend generation GPU).
    [[nodiscard]] std::optional<RuntimeDevicePair> bindRuntimeDevicePair(
        const PhysicalDeviceIdentity& renderDevice,
        const PhysicalDeviceIdentity& generationDevice,
        const std::optional<std::string>& generationSelector
    );

}
