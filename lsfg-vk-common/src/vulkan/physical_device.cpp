/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    template<typename Funcs>
    [[nodiscard]] std::vector<std::string> enumerateDeviceExtensionNamesImpl(
        const Funcs& funcs,
        VkPhysicalDevice device
    );

    [[nodiscard]] bool hasDeviceExtension(
            const std::vector<std::string>& extensions,
            const char* extensionName) {
        return std::ranges::binary_search(extensions, std::string(extensionName));
    }

    template<typename Funcs>
    [[nodiscard]] PhysicalDeviceIdentity queryPhysicalDeviceIdentity(
            const Funcs& funcs,
            VkPhysicalDevice device,
            const std::vector<std::string>& extensions) {
        const bool hasPciBusInfo = hasDeviceExtension(
            extensions, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);

        VkPhysicalDevicePCIBusInfoPropertiesEXT pciProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT
        };
        VkPhysicalDeviceIDProperties idProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            .pNext = hasPciBusInfo ? &pciProperties : nullptr
        };
        VkPhysicalDeviceProperties2 properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &idProperties
        };
        funcs.GetPhysicalDeviceProperties2(device, &properties);

        const auto deviceName = std::to_array(properties.properties.deviceName);
        PhysicalDeviceIdentity identity{
            .name = deviceName.data(),
            .vendorId = properties.properties.vendorID,
            .deviceId = properties.properties.deviceID,
            .deviceUuid = std::to_array(idProperties.deviceUUID),
            .driverUuid = std::to_array(idProperties.driverUUID)
        };
        if (hasPciBusInfo) {
            identity.pci = PciBusInfo{
                .domain = pciProperties.pciDomain,
                .bus = pciProperties.pciBus,
                .device = pciProperties.pciDevice,
                .function = pciProperties.pciFunction
            };
        }

        return identity;
    }

    template<typename Funcs>
    [[nodiscard]] std::vector<VkPhysicalDevice> enumeratePhysicalDeviceHandles(
            const Funcs& funcs,
            VkInstance instance) {
        uint32_t count{};
        auto res = funcs.EnumeratePhysicalDevices(instance, &count, nullptr);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res,
                "vkEnumeratePhysicalDevices() failed");
        if (count == 0)
            return {};

        std::vector<VkPhysicalDevice> devices;
        while (true) {
            devices.resize(count);
            res = funcs.EnumeratePhysicalDevices(
                instance, &count, devices.data());
            if (res != VK_SUCCESS && res != VK_INCOMPLETE)
                throw ls::vulkan_error(res,
                    "vkEnumeratePhysicalDevices() failed");
            if (res == VK_SUCCESS)
                break;

            res = funcs.EnumeratePhysicalDevices(instance, &count, nullptr);
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res,
                    "vkEnumeratePhysicalDevices() failed");
        }
        devices.resize(count);
        return devices;
    }

    [[nodiscard]] std::string hexId(uint32_t id) {
        std::ostringstream stream;
        stream << "0x" << std::uppercase << std::hex
            << std::setw(4) << std::setfill('0') << id;
        return stream.str();
    }

    template<typename Funcs>
    [[nodiscard]] std::vector<std::string> enumerateDeviceExtensionNamesImpl(
            const Funcs& funcs,
            VkPhysicalDevice device) {
        uint32_t count{};
        auto res = funcs.EnumerateDeviceExtensionProperties(
            device, nullptr, &count, nullptr);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res,
                "vkEnumerateDeviceExtensionProperties() failed");
        if (count == 0)
            return {};

        std::vector<VkExtensionProperties> properties;
        while (true) {
            properties.resize(count);
            res = funcs.EnumerateDeviceExtensionProperties(
                device, nullptr, &count, properties.data());
            if (res != VK_SUCCESS && res != VK_INCOMPLETE)
                throw ls::vulkan_error(res,
                    "vkEnumerateDeviceExtensionProperties() failed");
            if (res == VK_SUCCESS)
                break;

            res = funcs.EnumerateDeviceExtensionProperties(
                device, nullptr, &count, nullptr);
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res,
                    "vkEnumerateDeviceExtensionProperties() failed");
        }
        properties.resize(count);

        std::vector<std::string> extensions;
        extensions.reserve(properties.size());
        for (const auto& property : properties)
            extensions.emplace_back(std::to_array(property.extensionName).data());
        std::ranges::sort(extensions);
        const auto duplicates = std::ranges::unique(extensions);
        extensions.erase(duplicates.begin(), duplicates.end());
        return extensions;
    }

    template<typename Funcs>
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> snapshotPhysicalDevicesImpl(
            const Funcs& funcs,
            const std::vector<VkPhysicalDevice>& devices) {
        std::vector<PhysicalDeviceSnapshot> snapshots;
        snapshots.reserve(devices.size());
        for (const auto& device : devices) {
            auto extensions = enumerateDeviceExtensionNamesImpl(funcs, device);
            snapshots.push_back({
                .identity = queryPhysicalDeviceIdentity(funcs, device, extensions),
                .advertisedDeviceExtensions = std::move(extensions)
            });
        }
        return snapshots;
    }

    template<typename Funcs>
    [[nodiscard]] std::vector<PhysicalDeviceSnapshot> enumeratePhysicalDeviceSnapshotsImpl(
            const Funcs& funcs,
            VkInstance instance) {
        return snapshotPhysicalDevicesImpl(funcs,
            enumeratePhysicalDeviceHandles(funcs, instance));
    }
}

std::string PciBusInfo::identifier() const {
    return std::to_string(this->domain) + ":" + this->legacyIdentifier();
}

std::string PciBusInfo::legacyIdentifier() const {
    return std::to_string(this->bus) + ":"
        + std::to_string(this->device) + "."
        + std::to_string(this->function);
}

bool PhysicalDeviceIdentity::hasValidDeviceUuid() const {
    return std::ranges::any_of(this->deviceUuid,
        [](uint8_t value) { return value != 0; });
}

bool PhysicalDeviceIdentity::samePhysicalDevice(
        const PhysicalDeviceIdentity& other) const {
    if (this->hasValidDeviceUuid() && other.hasValidDeviceUuid())
        return this->deviceUuid == other.deviceUuid;

    return this->pci.has_value() && other.pci.has_value()
        && *this->pci == *other.pci;
}

std::string PhysicalDeviceIdentity::vendorDeviceIdentifier() const {
    return hexId(this->vendorId) + ":" + hexId(this->deviceId);
}

bool PhysicalDeviceIdentity::matchesSelector(const std::string& selector) const {
    if (selector == this->name || selector == this->vendorDeviceIdentifier())
        return true;

    return this->pci.has_value()
        && (selector == this->pci->identifier()
            || selector == this->pci->legacyIdentifier());
}

std::string vk::formatUuid(const DeviceUuid& uuid) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < uuid.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            stream << '-';
        stream << std::setw(2) << static_cast<unsigned int>(uuid.at(i));
    }
    return stream.str();
}

PhysicalDeviceIdentity vk::getPhysicalDeviceIdentity(
        const VulkanInstanceFuncs& funcs,
        VkPhysicalDevice device) {
    const auto extensions = enumerateDeviceExtensionNamesImpl(funcs, device);
    return queryPhysicalDeviceIdentity(funcs, device, extensions);
}

PhysicalDeviceIdentity vk::getPhysicalDeviceIdentity(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device) {
    const auto extensions = enumerateDeviceExtensionNamesImpl(funcs, device);
    return queryPhysicalDeviceIdentity(funcs, device, extensions);
}

std::vector<std::string> vk::enumerateDeviceExtensionNames(
        const VulkanInstanceFuncs& funcs,
        VkPhysicalDevice device) {
    return enumerateDeviceExtensionNamesImpl(funcs, device);
}

std::vector<std::string> vk::enumerateDeviceExtensionNames(
        const VulkanInstanceInventoryFuncs& funcs,
        VkPhysicalDevice device) {
    return enumerateDeviceExtensionNamesImpl(funcs, device);
}

std::vector<PhysicalDeviceSnapshot> vk::snapshotPhysicalDevices(
        const VulkanInstanceFuncs& funcs,
        const std::vector<VkPhysicalDevice>& devices) {
    return snapshotPhysicalDevicesImpl(funcs, devices);
}

std::vector<PhysicalDeviceSnapshot> vk::snapshotPhysicalDevices(
        const VulkanInstanceInventoryFuncs& funcs,
        const std::vector<VkPhysicalDevice>& devices) {
    return snapshotPhysicalDevicesImpl(funcs, devices);
}

std::vector<PhysicalDeviceSnapshot> vk::enumeratePhysicalDeviceSnapshots(
        const VulkanInstanceFuncs& funcs,
        VkInstance instance) {
    return enumeratePhysicalDeviceSnapshotsImpl(funcs, instance);
}

std::vector<VkPhysicalDevice> vk::enumeratePhysicalDevices(
        const VulkanInstanceInventoryFuncs& funcs,
        VkInstance instance) {
    return enumeratePhysicalDeviceHandles(funcs, instance);
}

std::vector<PhysicalDeviceSnapshot> vk::enumeratePhysicalDeviceSnapshots(
        const VulkanInstanceInventoryFuncs& funcs,
        VkInstance instance) {
    return enumeratePhysicalDeviceSnapshotsImpl(funcs, instance);
}

DeviceSelectionResult vk::resolveDeviceSelection(
        const std::vector<PhysicalDeviceSnapshot>& backendDevices,
        const PhysicalDeviceIdentity& applicationIdentity,
        const std::optional<std::string>& selector) {
    DeviceSelectionResult result{
        .resolution = selector.has_value()
            ? DeviceSelectionResolution::SelectorNotFoundInBackend
            : DeviceSelectionResolution::ApplicationDeviceNotVisibleToBackend
    };

    for (size_t i = 0; i < backendDevices.size(); ++i) {
        const auto& identity = backendDevices.at(i).identity;
        const bool matches = selector.has_value()
            ? identity.matchesSelector(*selector)
            : identity.samePhysicalDevice(applicationIdentity);
        if (!matches)
            continue;

        if (!result.selectedIndex.has_value())
            result.selectedIndex = i;
        ++result.matchCount;
    }

    if (result.matchCount == 0)
        return result;

    result.resolution = selector.has_value() && result.matchCount > 1
        ? DeviceSelectionResolution::SelectorAmbiguous
        : DeviceSelectionResolution::Selected;
    return result;
}

DriverUuidRelationship vk::compareDriverUuids(
        const PhysicalDeviceIdentity& first,
        const PhysicalDeviceIdentity& second) {
    const auto valid = [](const DeviceUuid& uuid) {
        return std::ranges::any_of(uuid,
            [](uint8_t value) { return value != 0; });
    };
    if (!valid(first.driverUuid) || !valid(second.driverUuid))
        return DriverUuidRelationship::Unknown;

    return first.driverUuid == second.driverUuid
        ? DriverUuidRelationship::Same
        : DriverUuidRelationship::Different;
}
