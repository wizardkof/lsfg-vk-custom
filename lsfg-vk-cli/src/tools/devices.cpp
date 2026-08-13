/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "devices.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

using namespace lsfgvk::cli;
using namespace lsfgvk::cli::devices;

namespace {
    constexpr std::array<const char*, 7> RELEVANT_EXTENSIONS{
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME
    };

    [[nodiscard]] const char* yesNo(bool value) {
        return value ? "YES" : "NO";
    }

    [[nodiscard]] bool extensionAdvertised(
            const vk::PhysicalDeviceSnapshot& snapshot,
            const char* extensionName) {
        return std::ranges::binary_search(
            snapshot.advertisedDeviceExtensions, std::string(extensionName));
    }

    [[nodiscard]] std::string formatId(uint32_t id) {
        std::ostringstream stream;
        stream << "0x" << std::uppercase << std::hex
            << std::setw(4) << std::setfill('0') << id;
        return stream.str();
    }

    [[nodiscard]] std::string formatPciBdf(const vk::PciBusInfo& pci) {
        std::ostringstream stream;
        stream << std::nouppercase << std::hex << std::setfill('0')
            << std::setw(4) << pci.domain << ":"
            << std::setw(2) << pci.bus << ":"
            << std::setw(2) << pci.device << "."
            << pci.function;
        return stream.str();
    }

    [[nodiscard]] std::vector<vk::PhysicalDeviceSnapshot> querySnapshots() {
        const vk::VulkanInventoryInstance instance{
            "lsfg-vk-cli", vk::version{2, 0, 0},
            "lsfg-vk-cli", vk::version{2, 0, 0}
        };
        return vk::enumeratePhysicalDeviceSnapshots(instance.fi(), instance.inst());
    }
}

void devices::render(std::ostream& output,
        const std::vector<vk::PhysicalDeviceSnapshot>& snapshots) {
    output << "Vulkan devices visible to this process: " << snapshots.size() << "\n";

    for (size_t i = 0; i < snapshots.size(); ++i) {
        const auto& snapshot = snapshots.at(i);
        const auto& identity = snapshot.identity;
        output << "\n[" << i << "]\n"
            << "  Name: " << identity.name << "\n"
            << "  Vendor ID: " << formatId(identity.vendorId) << "\n"
            << "  Device ID: " << formatId(identity.deviceId) << "\n"
            << "  Device UUID: " << vk::formatUuid(identity.deviceUuid) << "\n"
            << "  Driver UUID: " << vk::formatUuid(identity.driverUuid) << "\n"
            << "  PCI: " << (identity.pci.has_value()
                ? formatPciBdf(*identity.pci) : "<unavailable>") << "\n"
            << "\n  Advertised extensions:\n";

        for (const auto* extension : RELEVANT_EXTENSIONS)
            output << "    " << extension << ": "
                << yesNo(extensionAdvertised(snapshot, extension)) << "\n";
    }
}

int devices::run(const SnapshotProvider& provider,
        std::ostream& output, std::ostream& errors) {
    try {
        render(output, provider());
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        errors << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}

int devices::run() {
    return run(querySnapshots, std::cout, std::cerr);
}
