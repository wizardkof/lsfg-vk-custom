/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "runtime_dma_buf_backing.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <gbm.h>
#include <unistd.h>

using namespace lsfgvk::layer;

namespace {
    constexpr uint32_t CONTROL_WIDTH = 256;
    constexpr uint32_t CONTROL_HEIGHT = 256;

    [[nodiscard]] bool isRenderNodeName(const std::string& name) {
        return name.starts_with("renderD") && name.size() > 7;
    }
}

std::string lsfgvk::layer::formatPciSysfsAddress(const vk::PciBusInfo& pci) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0')
        << std::setw(4) << pci.domain << ':'
        << std::setw(2) << pci.bus << ':'
        << std::setw(2) << pci.device << '.'
        << pci.function;
    return stream.str();
}

std::filesystem::path lsfgvk::layer::findRenderNodeForPci(
        const vk::PciBusInfo& pci,
        const std::filesystem::path& pciDevicesRoot,
        const std::filesystem::path& driRoot) {
    const auto drmDirectory = pciDevicesRoot / formatPciSysfsAddress(pci) / "drm";
    std::error_code error;
    if (!std::filesystem::is_directory(drmDirectory, error))
        throw ls::error("DRM directory unavailable for render GPU: " + drmDirectory.string());

    std::filesystem::path selected;
    for (const auto& entry : std::filesystem::directory_iterator(drmDirectory)) {
        const auto name = entry.path().filename().string();
        if (!isRenderNodeName(name))
            continue;
        if (selected.empty() || name < selected.filename().string())
            selected = driRoot / name;
    }
    if (selected.empty())
        throw ls::error("no DRM render node found for render GPU PCI address "
            + formatPciSysfsAddress(pci));
    return selected;
}

RuntimeDmaBufBacking::RuntimeDmaBufBacking(
        std::filesystem::path nodePath,
        ls::OwnedFd nodeFd,
        gbm_device* device,
        gbm_bo* bo,
        ls::OwnedFd dmaBufFd,
        VkDeviceSize size) noexcept :
    nodePath(std::move(nodePath)),
    nodeFd(std::move(nodeFd)),
    gbmDevice(device),
    gbmBo(bo),
    dmaBufFd(std::move(dmaBufFd)),
    backingSize(size) {
}

RuntimeDmaBufBacking::RuntimeDmaBufBacking(RuntimeDmaBufBacking&& other) noexcept :
    nodePath(std::move(other.nodePath)),
    nodeFd(std::move(other.nodeFd)),
    gbmDevice(std::exchange(other.gbmDevice, nullptr)),
    gbmBo(std::exchange(other.gbmBo, nullptr)),
    dmaBufFd(std::move(other.dmaBufFd)),
    backingSize(std::exchange(other.backingSize, 0)) {
}

RuntimeDmaBufBacking& RuntimeDmaBufBacking::operator=(RuntimeDmaBufBacking&& other) noexcept {
    if (this == &other)
        return *this;
    this->reset();
    this->nodePath = std::move(other.nodePath);
    this->nodeFd = std::move(other.nodeFd);
    this->gbmDevice = std::exchange(other.gbmDevice, nullptr);
    this->gbmBo = std::exchange(other.gbmBo, nullptr);
    this->dmaBufFd = std::move(other.dmaBufFd);
    this->backingSize = std::exchange(other.backingSize, 0);
    return *this;
}

RuntimeDmaBufBacking::~RuntimeDmaBufBacking() {
    this->reset();
}

void RuntimeDmaBufBacking::reset() noexcept {
    this->dmaBufFd.reset();
    if (this->gbmBo) {
        gbm_bo_destroy(this->gbmBo);
        this->gbmBo = nullptr;
    }
    if (this->gbmDevice) {
        gbm_device_destroy(this->gbmDevice);
        this->gbmDevice = nullptr;
    }
    this->nodeFd.reset();
    this->backingSize = 0;
}

RuntimeDmaBufBacking RuntimeDmaBufBacking::create(
        const vk::PhysicalDeviceIdentity& renderIdentity) {
    if (!renderIdentity.pci.has_value())
        throw ls::error("render GPU has no stable PCI address for GBM DMA-BUF allocation");

    const auto nodePath = findRenderNodeForPci(*renderIdentity.pci);
    ls::OwnedFd nodeFd(::open(nodePath.c_str(), O_RDWR | O_CLOEXEC));
    if (!nodeFd)
        throw ls::error("failed to open GBM allocator node " + nodePath.string()
            + " errno=" + std::to_string(errno));

    auto* device = gbm_create_device(nodeFd.get());
    if (!device)
        throw ls::error("gbm_create_device() failed for " + nodePath.string()
            + " errno=" + std::to_string(errno));

    auto* bo = gbm_bo_create(device,
        CONTROL_WIDTH, CONTROL_HEIGHT,
        GBM_FORMAT_R8,
        GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING);
    if (!bo) {
        gbm_device_destroy(device);
        throw ls::error("gbm_bo_create() failed for runtime exchange backing errno="
            + std::to_string(errno));
    }

    ls::OwnedFd dmaBufFd(gbm_bo_get_fd(bo));
    if (!dmaBufFd) {
        gbm_bo_destroy(bo);
        gbm_device_destroy(device);
        throw ls::error("gbm_bo_get_fd() failed for runtime exchange backing errno="
            + std::to_string(errno));
    }

    const off_t end = ::lseek(dmaBufFd.get(), 0, SEEK_END);
    if (end <= 0) {
        gbm_bo_destroy(bo);
        gbm_device_destroy(device);
        throw ls::error("unable to determine runtime DMA-BUF backing size errno="
            + std::to_string(errno));
    }
    static_cast<void>(::lseek(dmaBufFd.get(), 0, SEEK_SET));

    return RuntimeDmaBufBacking(nodePath, std::move(nodeFd), device, bo,
        std::move(dmaBufFd), static_cast<VkDeviceSize>(end));
}

ls::OwnedFd RuntimeDmaBufBacking::duplicateFd() const {
    if (!this->dmaBufFd)
        throw ls::error("runtime DMA-BUF backing has no valid file descriptor");
    const int duplicate = ::fcntl(this->dmaBufFd.get(), F_DUPFD_CLOEXEC, 0);
    if (duplicate < 0)
        throw ls::error("failed to duplicate runtime DMA-BUF fd errno="
            + std::to_string(errno));
    return ls::OwnedFd(duplicate);
}
