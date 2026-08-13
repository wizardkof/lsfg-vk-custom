/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {
    bool exposePciExtension{};
    vk::DeviceUuid queriedDeviceUuid{};
    vk::DeviceUuid queriedDriverUuid{};

    VKAPI_ATTR VkResult VKAPI_CALL enumerateDeviceExtensions(
            VkPhysicalDevice,
            const char*,
            uint32_t* count,
            VkExtensionProperties* properties) {
        if (!properties) {
            *count = exposePciExtension ? 1U : 0U;
            return VK_SUCCESS;
        }

        if (exposePciExtension && *count > 0) {
            std::strcpy(properties[0].extensionName,
                VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
            properties[0].specVersion = VK_EXT_PCI_BUS_INFO_SPEC_VERSION;
            *count = 1;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceProperties2(
            VkPhysicalDevice,
            VkPhysicalDeviceProperties2* properties) {
        std::strcpy(properties->properties.deviceName, "Test GPU");
        properties->properties.vendorID = 0x1002;
        properties->properties.deviceID = 0x73BF;

        auto* id = reinterpret_cast<VkPhysicalDeviceIDProperties*>(properties->pNext);
        assert(id != nullptr);
        std::ranges::copy(queriedDeviceUuid, id->deviceUUID);
        std::ranges::copy(queriedDriverUuid, id->driverUUID);

        if (id->pNext) {
            auto* pci = reinterpret_cast<VkPhysicalDevicePCIBusInfoPropertiesEXT*>(id->pNext);
            pci->pciDomain = 4;
            pci->pciBus = 3;
            pci->pciDevice = 2;
            pci->pciFunction = 1;
        }
    }

    vk::VulkanInstanceFuncs fakeFunctions() {
        return {
            .EnumerateDeviceExtensionProperties = enumerateDeviceExtensions,
            .GetPhysicalDeviceProperties2 = getPhysicalDeviceProperties2
        };
    }

    vk::DeviceUuid uuid(uint8_t suffix) {
        vk::DeviceUuid value{};
        value.back() = suffix;
        return value;
    }
}

int main() {
    const vk::PciBusInfo pci{
        .domain = 4,
        .bus = 3,
        .device = 2,
        .function = 1
    };
    assert(pci.identifier() == "4:3:2.1");
    assert(pci.legacyIdentifier() == "3:2.1");

    const vk::PhysicalDeviceIdentity identity{
        .name = "Test GPU",
        .vendorId = 0x1002,
        .deviceId = 0x73BF,
        .deviceUuid = uuid(1),
        .driverUuid = uuid(2),
        .pci = pci
    };
    assert(identity.hasValidDeviceUuid());
    assert(vk::formatUuid(identity.deviceUuid)
        == "00000000-0000-0000-0000-000000000001");
    assert(identity.vendorDeviceIdentifier() == "0x1002:0x73BF");
    assert(identity.matchesSelector("Test GPU"));
    assert(identity.matchesSelector("0x1002:0x73BF"));
    assert(identity.matchesSelector("3:2.1"));
    assert(identity.matchesSelector("4:3:2.1"));
    assert(!identity.matchesSelector("0x10DE:0x2684"));

    auto sameUuid = identity;
    sameUuid.driverUuid = uuid(9);
    sameUuid.pci = vk::PciBusInfo{.domain = 9, .bus = 9, .device = 9, .function = 0};
    assert(identity.samePhysicalDevice(sameUuid));

    auto differentUuid = identity;
    differentUuid.deviceUuid = uuid(3);
    assert(!identity.samePhysicalDevice(differentUuid));

    auto pciFallbackA = identity;
    auto pciFallbackB = identity;
    pciFallbackA.deviceUuid = {};
    pciFallbackB.deviceUuid = {};
    assert(pciFallbackA.samePhysicalDevice(pciFallbackB));
    pciFallbackB.pci = vk::PciBusInfo{
        .domain = pci.domain + 1,
        .bus = pci.bus,
        .device = pci.device,
        .function = pci.function
    };
    assert(!pciFallbackA.samePhysicalDevice(pciFallbackB));
    pciFallbackB.pci = std::nullopt;
    assert(!pciFallbackA.samePhysicalDevice(pciFallbackB));

    auto validUuidA = identity;
    auto missingUuidB = identity;
    missingUuidB.deviceUuid = {};
    assert(validUuidA.samePhysicalDevice(missingUuidB));

    auto missingUuidA = identity;
    auto validUuidB = identity;
    missingUuidA.deviceUuid = {};
    assert(missingUuidA.samePhysicalDevice(validUuidB));
    validUuidB.pci = vk::PciBusInfo{
        .domain = pci.domain + 1,
        .bus = pci.bus,
        .device = pci.device,
        .function = pci.function
    };
    assert(!missingUuidA.samePhysicalDevice(validUuidB));

    auto otherGpu = identity;
    otherGpu.name = "Other GPU";
    otherGpu.deviceUuid = uuid(4);
    const auto selectsApplicationGpu = [&identity](
            const std::vector<vk::PhysicalDeviceIdentity>& candidates) {
        const auto selected = std::ranges::find_if(candidates,
            [&identity](const vk::PhysicalDeviceIdentity& candidate) {
                return candidate.samePhysicalDevice(identity);
            });
        return selected != candidates.end()
            && selected->deviceUuid == identity.deviceUuid;
    };
    assert(selectsApplicationGpu({otherGpu, identity}));
    assert(selectsApplicationGpu({identity, otherGpu}));

    exposePciExtension = true;
    queriedDeviceUuid = uuid(7);
    queriedDriverUuid = uuid(8);
    const auto queriedWithPci = vk::getPhysicalDeviceIdentity(
        fakeFunctions(), reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}));
    assert(queriedWithPci.name == "Test GPU");
    assert(queriedWithPci.deviceUuid == queriedDeviceUuid);
    assert(queriedWithPci.driverUuid == queriedDriverUuid);
    assert(queriedWithPci.pci == pci);

    exposePciExtension = false;
    const auto queriedWithoutPci = vk::getPhysicalDeviceIdentity(
        fakeFunctions(), reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}));
    assert(!queriedWithoutPci.pci.has_value());

    return 0;
}
