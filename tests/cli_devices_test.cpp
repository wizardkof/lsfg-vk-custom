/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/devices.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {
    vk::DeviceUuid uuid(uint8_t suffix) {
        vk::DeviceUuid value{};
        value.back() = suffix;
        return value;
    }

    vk::PhysicalDeviceSnapshot snapshot(
            const std::string& name,
            uint32_t vendorId,
            uint32_t deviceId,
            uint8_t deviceUuid,
            uint8_t driverUuid,
            std::optional<vk::PciBusInfo> pci,
            std::vector<std::string> extensions = {}) {
        std::ranges::sort(extensions);
        return {
            .identity = {
                .name = name,
                .vendorId = vendorId,
                .deviceId = deviceId,
                .deviceUuid = uuid(deviceUuid),
                .driverUuid = uuid(driverUuid),
                .pci = pci
            },
            .advertisedDeviceExtensions = std::move(extensions)
        };
    }
}

int main() {
    std::ostringstream output;
    lsfgvk::cli::devices::render(output, {});
    assert(output.str() == "Vulkan devices visible to this process: 0\n");

    const auto amd = snapshot(
        "AMD GPU", 0x1002, 0x73BF, 1, 2,
        vk::PciBusInfo{.domain = 0, .bus = 12, .device = 0, .function = 0},
        {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
        });
    const auto nvidia = snapshot(
        "NVIDIA GPU", 0x10DE, 0x2684, 3, 4, std::nullopt);

    output.str({});
    output.clear();
    lsfgvk::cli::devices::render(output, {nvidia});
    assert(output.str().starts_with("Vulkan devices visible to this process: 1\n"));
    assert(output.str().find("  PCI: <unavailable>\n") != std::string::npos);

    output.str({});
    output.clear();
    lsfgvk::cli::devices::render(output, {amd, nvidia});
    const std::string rendered = output.str();
    assert(rendered.starts_with("Vulkan devices visible to this process: 2\n"));
    assert(rendered.find("[0]\n  Name: AMD GPU")
        < rendered.find("[1]\n  Name: NVIDIA GPU"));
    assert(rendered.find("  Vendor ID: 0x1002\n") != std::string::npos);
    assert(rendered.find("  Device ID: 0x73BF\n") != std::string::npos);
    assert(rendered.find("  Device UUID: 00000000-0000-0000-0000-000000000001\n")
        != std::string::npos);
    assert(rendered.find("  Driver UUID: 00000000-0000-0000-0000-000000000002\n")
        != std::string::npos);
    assert(rendered.find("  PCI: 0000:0c:00.0\n") != std::string::npos);
    assert(rendered.find("    VK_KHR_external_memory: YES\n") != std::string::npos);
    assert(rendered.find("    VK_KHR_external_memory_fd: NO\n") != std::string::npos);
    assert(rendered.find("    VK_EXT_external_memory_dma_buf: YES\n") != std::string::npos);
    assert(rendered.find("    VK_KHR_external_semaphore: NO\n") != std::string::npos);
    assert(rendered.find("    VK_KHR_external_semaphore_fd: YES\n") != std::string::npos);
    assert(rendered.find("    VK_EXT_image_drm_format_modifier: NO\n") != std::string::npos);
    assert(rendered.find("    VK_EXT_queue_family_foreign: NO\n") != std::string::npos);

    std::ostringstream errors;
    output.str({});
    output.clear();
    auto result = lsfgvk::cli::devices::run(
        [] { return std::vector<vk::PhysicalDeviceSnapshot>{}; }, output, errors);
    assert(result == EXIT_SUCCESS);
    assert(output.str() == "Vulkan devices visible to this process: 0\n");
    assert(errors.str().empty());

    output.str({});
    output.clear();
    errors.str({});
    errors.clear();
    result = lsfgvk::cli::devices::run(
        []() -> std::vector<vk::PhysicalDeviceSnapshot> {
            throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                "injected inventory failure");
        }, output, errors);
    assert(result == EXIT_FAILURE);
    assert(output.str().empty());
    assert(errors.str().find("error: injected inventory failure") == 0);

    return 0;
}
