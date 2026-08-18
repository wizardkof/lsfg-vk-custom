/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "interop.hpp"

#include "lsfg-vk-common/vulkan/external_capabilities.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
    const char* yesNo(bool value) { return value ? "yes" : "no"; }

    const char* deviceType(VkPhysicalDeviceType type) {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
            default: return "other";
        }
    }

    std::string formatHandleTypesImpl(VkFlags flags) {
        std::ostringstream out;
        out << "0x" << std::hex << std::uppercase << flags << std::dec;
        if (flags & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
            out << " DMA_BUF";
        if (flags & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
            out << " OPAQUE_FD";
        if (flags & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT)
            out << " SYNC_FD";
        return out.str();
    }

    struct DeviceReport {
        vk::PhysicalDeviceIdentity identity;
        VkPhysicalDeviceType type{};
        vk::ExternalBufferCapability buffer;
        vk::ExternalSemaphoreCapability syncFd;
    };
}

std::string lsfgvk::cli::interop::formatHandleTypes(
        VkFlags flags) {
    return formatHandleTypesImpl(flags);
}

bool lsfgvk::cli::interop::dmaBufImportCandidate(
        bool firstImportable, bool secondImportable) {
    return firstImportable && secondImportable;
}

bool lsfgvk::cli::interop::syncFdBridgeCandidate(
        bool firstExportable, bool secondImportable) {
    return firstExportable && secondImportable;
}

int lsfgvk::cli::interop::run() {
    try {
        const vk::VulkanInventoryInstance instance{
            "lsfg-vk-cli", vk::version{2, 0, 0},
            "lsfg-vk-cli", vk::version{2, 0, 0}
        };
        const auto devices = vk::enumeratePhysicalDevices(instance.fi(), instance.inst());
        const vk::ExternalBufferCapabilityQuery bufferQuery{
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        };
        std::vector<DeviceReport> reports;
        reports.reserve(devices.size());
        for (const auto device : devices) {
            VkPhysicalDeviceProperties2 properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
            };
            instance.fi().GetPhysicalDeviceProperties2(device, &properties);
            reports.push_back({
                .identity = vk::getPhysicalDeviceIdentity(instance.fi(), device),
                .type = properties.properties.deviceType,
                .buffer = vk::queryExternalBufferCapability(
                    instance.fi(), device, bufferQuery),
                .syncFd = vk::queryExternalSemaphoreCapability(
                    instance.fi(), device,
                    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT)
            });
        }

        for (size_t i = 0; i < reports.size(); ++i) {
            const auto& report = reports.at(i);
            const auto& id = report.identity;
            std::cout << "Physical Device " << i << "\n"
                << "  Name: " << id.name << "\n"
                << "  Type: " << deviceType(report.type) << "\n"
                << "  VendorID: 0x" << std::hex << std::uppercase << id.vendorId
                << "\n  DeviceID: 0x" << id.deviceId << std::dec << "\n"
                << "  DeviceUUID: " << vk::formatUuid(id.deviceUuid) << "\n"
                << "  DriverUUID: " << vk::formatUuid(id.driverUuid) << "\n";
            if (id.pci)
                std::cout << "  PCI: " << id.pci->identifier() << "\n";

            std::cout << "\n  DMA_BUF VkBuffer\n"
                << "    query: VK_SUCCESS\n"
                << "    importable: " << yesNo(report.buffer.importable()) << "\n"
                << "    exportable: " << yesNo(report.buffer.exportable()) << "\n"
                << "    dedicated-only: " << yesNo(report.buffer.dedicatedOnly()) << "\n"
                << "    compatible: " << yesNo(report.buffer.queriedHandleCompatible()) << "\n"
                << "    compatibleHandleTypes: ";
            std::cout << formatHandleTypes(report.buffer.compatibleHandleTypes);
            std::cout << "\n    exportFromImportedHandleTypes: ";
            std::cout << formatHandleTypes(report.buffer.exportFromImportedHandleTypes);
            std::cout << "\n\n  SYNC_FD Semaphore\n"
                << "    importable: " << yesNo(report.syncFd.importable()) << "\n"
                << "    exportable: " << yesNo(report.syncFd.exportable()) << "\n"
                << "    compatible: " << yesNo(report.syncFd.queriedHandleCompatible()) << "\n"
                << "    compatibleHandleTypes: ";
            std::cout << formatHandleTypes(report.syncFd.compatibleHandleTypes);
            std::cout << "\n\n";
        }

        for (size_t a = 0; a < reports.size(); ++a) {
            for (size_t b = 0; b < reports.size(); ++b) {
                if (a == b)
                    continue;
                const auto& first = reports.at(a);
                const auto& second = reports.at(b);
                std::cout << "Pair " << a << " -> " << b << "\n"
                    << "  DMA_BUF import/import candidate: "
                    << yesNo(dmaBufImportCandidate(
                        first.buffer.importable(), second.buffer.importable())) << "\n"
                    << "  SYNC_FD bridge candidate: "
                    << yesNo(syncFdBridgeCandidate(
                        first.syncFd.exportable(), second.syncFd.importable()))
                    << "\n\n";
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
