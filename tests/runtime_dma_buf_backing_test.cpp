/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "runtime_dma_buf_backing.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <string>

int main() {
    const vk::PciBusInfo pci{
        .domain = 0,
        .bus = 0x0c,
        .device = 0,
        .function = 0
    };
    assert(lsfgvk::layer::formatPciSysfsAddress(pci) == "0000:0c:00.0");

    const auto nonce = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path()
        / ("lsfg-vk-p3d-render-node-" + nonce);
    const auto pciRoot = root / "pci";
    const auto devRoot = root / "dev" / "dri";
    const auto drmRoot = pciRoot / "0000:0c:00.0" / "drm";

    std::filesystem::create_directories(drmRoot / "card2");
    std::filesystem::create_directories(drmRoot / "renderD129");
    std::filesystem::create_directories(devRoot);

    const auto resolved = lsfgvk::layer::findRenderNodeForPci(
        pci, pciRoot, devRoot);
    assert(resolved == devRoot / "renderD129");

    bool missingRejected = false;
    try {
        const vk::PciBusInfo missing{
            .domain = 0,
            .bus = 1,
            .device = 0,
            .function = 0
        };
        static_cast<void>(lsfgvk::layer::findRenderNodeForPci(
            missing, pciRoot, devRoot));
    } catch (const ls::error&) {
        missingRejected = true;
    }
    assert(missingRejected);

    std::filesystem::remove_all(root);
    return 0;
}
