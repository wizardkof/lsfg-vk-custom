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
#include <sstream>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    [[nodiscard]] bool hasDeviceExtension(
            const VulkanInstanceFuncs& funcs,
            VkPhysicalDevice device,
            const char* extensionName) {
        uint32_t count{};
        auto res = funcs.EnumerateDeviceExtensionProperties(
            device, nullptr, &count, nullptr);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res,
                "vkEnumerateDeviceExtensionProperties() failed");
        if (count == 0)
            return false;

        std::vector<VkExtensionProperties> extensions;
        while (true) {
            extensions.resize(count);
            res = funcs.EnumerateDeviceExtensionProperties(
                device, nullptr, &count, extensions.data());
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
        extensions.resize(count);

        return std::ranges::any_of(extensions,
            [extensionName](const VkExtensionProperties& extension) {
                const auto name = std::to_array(extension.extensionName);
                return std::string(name.data()) == extensionName;
            });
    }

    [[nodiscard]] std::string hexId(uint32_t id) {
        std::ostringstream stream;
        stream << "0x" << std::uppercase << std::hex
            << std::setw(4) << std::setfill('0') << id;
        return stream.str();
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
    const bool hasPciBusInfo = hasDeviceExtension(
        funcs, device, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);

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
