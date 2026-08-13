/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include <vulkan/vulkan_core.h>

namespace vk {

    struct VulkanInstanceFuncs;

    using DeviceUuid = std::array<uint8_t, VK_UUID_SIZE>;

    /// Optional PCI address exposed by VK_EXT_pci_bus_info.
    // These are intentionally value objects with public fields so Vulkan query
    // results can be copied and compared without driver-owned state.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    struct PciBusInfo {
        uint32_t domain{};
        uint32_t bus{};
        uint32_t device{};
        uint32_t function{};

        /// Full domain:bus:device.function identifier.
        [[nodiscard]] std::string identifier() const;
        /// Legacy bus:device.function identifier accepted by existing profiles.
        [[nodiscard]] std::string legacyIdentifier() const;

        bool operator==(const PciBusInfo&) const = default;
    };

    /// Stable properties used to correlate a physical device across VkInstances.
    struct PhysicalDeviceIdentity {
        std::string name;
        uint32_t vendorId{};
        uint32_t deviceId{};
        DeviceUuid deviceUuid{};
        DeviceUuid driverUuid{};
        std::optional<PciBusInfo> pci;

        [[nodiscard]] bool hasValidDeviceUuid() const;
        [[nodiscard]] bool samePhysicalDevice(const PhysicalDeviceIdentity& other) const;
        [[nodiscard]] std::string vendorDeviceIdentifier() const;
        [[nodiscard]] bool matchesSelector(const std::string& selector) const;
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    /// Format a Vulkan UUID for diagnostics.
    [[nodiscard]] std::string formatUuid(const DeviceUuid& uuid);

    /// Query identity properties for one VkPhysicalDevice.
    /// @throws ls::vulkan_error when device extension enumeration fails.
    [[nodiscard]] PhysicalDeviceIdentity getPhysicalDeviceIdentity(
        const VulkanInstanceFuncs& funcs,
        VkPhysicalDevice device
    );

}
