/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {
    struct FakeDevice {
        VkPhysicalDevice handle;
        vk::PhysicalDeviceIdentity identity;
        std::vector<std::string> extensions;
    };

    std::vector<FakeDevice> devices;
    bool returnIncompleteOnce{};
    size_t physicalDeviceCountQueries{};

    vk::DeviceUuid uuid(uint8_t suffix) {
        vk::DeviceUuid value{};
        value.back() = suffix;
        return value;
    }

    vk::PhysicalDeviceIdentity identity(
            const std::string& name,
            uint8_t deviceUuid,
            uint8_t driverUuid,
            uint32_t deviceId) {
        return {
            .name = name,
            .vendorId = 0x1002,
            .deviceId = deviceId,
            .deviceUuid = uuid(deviceUuid),
            .driverUuid = uuid(driverUuid),
            .pci = vk::PciBusInfo{
                .domain = 0,
                .bus = deviceUuid,
                .device = 0,
                .function = 0
            }
        };
    }

    const FakeDevice& findDevice(VkPhysicalDevice handle) {
        const auto found = std::ranges::find_if(devices,
            [handle](const FakeDevice& device) { return device.handle == handle; });
        assert(found != devices.end());
        return *found;
    }

    VKAPI_ATTR VkResult VKAPI_CALL enumeratePhysicalDevices(
            VkInstance,
            uint32_t* count,
            VkPhysicalDevice* properties) {
        if (!properties) {
            ++physicalDeviceCountQueries;
            *count = returnIncompleteOnce && physicalDeviceCountQueries == 1
                ? 1U : static_cast<uint32_t>(devices.size());
            return VK_SUCCESS;
        }

        const uint32_t capacity = *count;
        const uint32_t written = std::min(capacity,
            static_cast<uint32_t>(devices.size()));
        for (uint32_t i = 0; i < written; ++i)
            properties[i] = devices.at(i).handle;
        *count = written;

        if (capacity < devices.size()) {
            returnIncompleteOnce = false;
            return VK_INCOMPLETE;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL enumerateDeviceExtensions(
            VkPhysicalDevice handle,
            const char*,
            uint32_t* count,
            VkExtensionProperties* properties) {
        const auto& extensions = findDevice(handle).extensions;
        if (!properties) {
            *count = static_cast<uint32_t>(extensions.size());
            return VK_SUCCESS;
        }

        const uint32_t capacity = *count;
        const uint32_t written = std::min(capacity,
            static_cast<uint32_t>(extensions.size()));
        for (uint32_t i = 0; i < written; ++i) {
            std::strcpy(properties[i].extensionName, extensions.at(i).c_str());
            properties[i].specVersion = 1;
        }
        *count = written;
        return capacity < extensions.size() ? VK_INCOMPLETE : VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceProperties2(
            VkPhysicalDevice handle,
            VkPhysicalDeviceProperties2* properties) {
        const auto& source = findDevice(handle).identity;
        std::strcpy(properties->properties.deviceName, source.name.c_str());
        properties->properties.vendorID = source.vendorId;
        properties->properties.deviceID = source.deviceId;

        auto* ids = reinterpret_cast<VkPhysicalDeviceIDProperties*>(properties->pNext);
        assert(ids != nullptr);
        std::ranges::copy(source.deviceUuid, ids->deviceUUID);
        std::ranges::copy(source.driverUuid, ids->driverUUID);

        if (ids->pNext && source.pci.has_value()) {
            auto* pci = reinterpret_cast<VkPhysicalDevicePCIBusInfoPropertiesEXT*>(ids->pNext);
            pci->pciDomain = source.pci->domain;
            pci->pciBus = source.pci->bus;
            pci->pciDevice = source.pci->device;
            pci->pciFunction = source.pci->function;
        }
    }

    vk::VulkanInstanceFuncs fakeFunctions() {
        return {
            .EnumeratePhysicalDevices = enumeratePhysicalDevices,
            .EnumerateDeviceExtensionProperties = enumerateDeviceExtensions,
            .GetPhysicalDeviceProperties2 = getPhysicalDeviceProperties2
        };
    }

    vk::PhysicalDeviceSnapshot snapshot(const vk::PhysicalDeviceIdentity& value) {
        return {.identity = value};
    }
}

int main() {
    const auto gpuA = identity("GPU A", 1, 7, 0x1111);
    const auto gpuB = identity("GPU B", 2, 8, 0x2222);
    const auto handleA = reinterpret_cast<VkPhysicalDevice>(uintptr_t{1});
    const auto handleB = reinterpret_cast<VkPhysicalDevice>(uintptr_t{2});

    devices = {{
        .handle = handleA,
        .identity = gpuA,
        .extensions = {
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
        }
    }};
    physicalDeviceCountQueries = 0;
    auto enumerated = vk::enumeratePhysicalDeviceSnapshots(
        fakeFunctions(), reinterpret_cast<VkInstance>(uintptr_t{1}));
    assert(enumerated.size() == 1);
    assert(enumerated.front().identity.samePhysicalDevice(gpuA));
    assert(std::ranges::is_sorted(enumerated.front().advertisedDeviceExtensions));
    assert(std::ranges::binary_search(enumerated.front().advertisedDeviceExtensions,
        std::string(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)));
    assert(!std::ranges::binary_search(enumerated.front().advertisedDeviceExtensions,
        std::string(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)));

    devices.push_back({
        .handle = handleB,
        .identity = gpuB,
        .extensions = {VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME}
    });
    physicalDeviceCountQueries = 0;
    enumerated = vk::enumeratePhysicalDeviceSnapshots(
        fakeFunctions(), reinterpret_cast<VkInstance>(uintptr_t{1}));
    assert(enumerated.size() == 2);
    assert(enumerated.at(0).identity.samePhysicalDevice(gpuA));
    assert(enumerated.at(1).identity.samePhysicalDevice(gpuB));
    assert(std::ranges::binary_search(enumerated.at(1).advertisedDeviceExtensions,
        std::string(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)));

    returnIncompleteOnce = true;
    physicalDeviceCountQueries = 0;
    enumerated = vk::enumeratePhysicalDeviceSnapshots(
        fakeFunctions(), reinterpret_cast<VkInstance>(uintptr_t{1}));
    assert(enumerated.size() == 2);
    assert(physicalDeviceCountQueries == 2);

    const std::vector<vk::PhysicalDeviceSnapshot> orderAB{
        snapshot(gpuA), snapshot(gpuB)};
    const std::vector<vk::PhysicalDeviceSnapshot> orderBA{
        snapshot(gpuB), snapshot(gpuA)};
    auto result = vk::resolveDeviceSelection(orderAB, gpuA, std::nullopt);
    assert(result.resolution == vk::DeviceSelectionResolution::Selected);
    assert(result.selectedIndex == 0);
    assert(orderAB.at(*result.selectedIndex).identity.samePhysicalDevice(gpuA));
    result = vk::resolveDeviceSelection(orderBA, gpuA, std::nullopt);
    assert(result.resolution == vk::DeviceSelectionResolution::Selected);
    assert(result.selectedIndex == 1);
    assert(orderBA.at(*result.selectedIndex).identity.samePhysicalDevice(gpuA));

    result = vk::resolveDeviceSelection({snapshot(gpuB)}, gpuA, std::nullopt);
    assert(result.resolution
        == vk::DeviceSelectionResolution::ApplicationDeviceNotVisibleToBackend);
    assert(!result.selectedIndex.has_value());

    result = vk::resolveDeviceSelection(orderAB, gpuA, std::string("GPU B"));
    assert(result.resolution == vk::DeviceSelectionResolution::Selected);
    assert(result.selectedIndex == 1);
    result = vk::resolveDeviceSelection(orderAB, gpuA, std::string("Missing GPU"));
    assert(result.resolution == vk::DeviceSelectionResolution::SelectorNotFoundInBackend);
    assert(!result.selectedIndex.has_value());

    auto duplicateName = gpuB;
    duplicateName.name = gpuA.name;
    result = vk::resolveDeviceSelection(
        {snapshot(gpuA), snapshot(duplicateName)}, gpuA, std::string("GPU A"));
    assert(result.resolution == vk::DeviceSelectionResolution::SelectorAmbiguous);
    assert(result.matchCount == 2);
    assert(result.selectedIndex == 0);

    assert(gpuA.samePhysicalDevice(gpuA));
    assert(!gpuA.samePhysicalDevice(gpuB));
    auto sameDriver = gpuB;
    sameDriver.driverUuid = gpuA.driverUuid;
    assert(vk::compareDriverUuids(gpuA, sameDriver)
        == vk::DriverUuidRelationship::Same);
    assert(vk::compareDriverUuids(gpuA, gpuB)
        == vk::DriverUuidRelationship::Different);
    auto unknownDriver = gpuB;
    unknownDriver.driverUuid = {};
    assert(vk::compareDriverUuids(gpuA, unknownDriver)
        == vk::DriverUuidRelationship::Unknown);

    return 0;
}
