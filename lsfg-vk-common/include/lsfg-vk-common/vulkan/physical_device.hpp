/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {

    struct VulkanInstanceFuncs;
    struct VulkanInstanceInventoryFuncs;

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

    /// Immutable device information observed through one VkInstance.
    struct PhysicalDeviceSnapshot {
        PhysicalDeviceIdentity identity;
        std::vector<std::string> advertisedDeviceExtensions;
    };

    /// Diagnostic result of resolving a profile selection against backend-visible devices.
    enum class DeviceSelectionResolution {
        Selected,
        ApplicationDeviceNotVisibleToBackend,
        SelectorNotFoundInBackend,
        SelectorAmbiguous
    };

    struct DeviceSelectionResult {
        DeviceSelectionResolution resolution;
        std::optional<size_t> selectedIndex;
        size_t matchCount{};
    };

    /// driverUUID relationship for diagnostics only; never physical-device identity.
    enum class DriverUuidRelationship {
        Same,
        Different,
        Unknown
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
    [[nodiscard]] PhysicalDeviceIdentity getPhysicalDeviceIdentity(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device
    );

    /// Enumerate and sort all device extension names advertised by a physical device.
    [[nodiscard]] std::vector<std::string> enumerateDeviceExtensionNames(
        const VulkanInstanceFuncs& funcs,
        VkPhysicalDevice device
    );
    [[nodiscard]] std::vector<std::string> enumerateDeviceExtensionNames(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device
    );

    /// Snapshot already-enumerated physical-device handles, preserving device order.
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> snapshotPhysicalDevices(
        const VulkanInstanceFuncs& funcs,
        const std::vector<VkPhysicalDevice>& devices
    );
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> snapshotPhysicalDevices(
        const VulkanInstanceInventoryFuncs& funcs,
        const std::vector<VkPhysicalDevice>& devices
    );

    /// Enumerate all physical devices visible to an instance and snapshot each one.
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> enumeratePhysicalDeviceSnapshots(
        const VulkanInstanceFuncs& funcs,
        VkInstance instance
    );
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> enumeratePhysicalDeviceSnapshots(
        const VulkanInstanceInventoryFuncs& funcs,
        VkInstance instance
    );

    [[nodiscard]] std::vector<VkPhysicalDevice> enumeratePhysicalDevices(
        const VulkanInstanceInventoryFuncs& funcs,
        VkInstance instance
    );

    /// Resolve Default or an explicit selector without changing first-match selection semantics.
    [[nodiscard]] DeviceSelectionResult resolveDeviceSelection(
        const std::vector<PhysicalDeviceSnapshot>& backendDevices,
        const PhysicalDeviceIdentity& applicationIdentity,
        const std::optional<std::string>& selector
    );

    /// Compare driver UUIDs for diagnostics, returning Unknown for invalid UUIDs.
    [[nodiscard]] DriverUuidRelationship compareDriverUuids(
        const PhysicalDeviceIdentity& first,
        const PhysicalDeviceIdentity& second
    );

}
