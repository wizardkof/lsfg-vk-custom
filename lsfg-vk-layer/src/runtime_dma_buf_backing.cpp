/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "runtime_dma_buf_backing.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <gbm.h>
#include <libdrm/drm_fourcc.h>
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
        VkDeviceSize size, uint32_t fourcc, uint64_t modifier,
        uint32_t planes, uint32_t stride, uint32_t offset) noexcept :
    nodePath(std::move(nodePath)),
    nodeFd(std::move(nodeFd)),
    gbmDevice(device),
    gbmBo(bo),
    dmaBufFd(std::move(dmaBufFd)),
    backingSize(size), boFourcc(fourcc), boModifier(modifier), boPlanes(planes),
    boStride(stride), boOffset(offset) {
}

RuntimeDmaBufBacking::RuntimeDmaBufBacking(RuntimeDmaBufBacking&& other) noexcept :
    nodePath(std::move(other.nodePath)),
    nodeFd(std::move(other.nodeFd)),
    gbmDevice(std::exchange(other.gbmDevice, nullptr)),
    gbmBo(std::exchange(other.gbmBo, nullptr)),
    dmaBufFd(std::move(other.dmaBufFd)),
    backingSize(std::exchange(other.backingSize, 0)) {
    boFourcc = std::exchange(other.boFourcc, 0);
    boModifier = std::exchange(other.boModifier, 0);
    boPlanes = std::exchange(other.boPlanes, 0);
    boStride = std::exchange(other.boStride, 0);
    boOffset = std::exchange(other.boOffset, 0);
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
    this->boFourcc = std::exchange(other.boFourcc, 0);
    this->boModifier = std::exchange(other.boModifier, 0);
    this->boPlanes = std::exchange(other.boPlanes, 0);
    this->boStride = std::exchange(other.boStride, 0);
    this->boOffset = std::exchange(other.boOffset, 0);
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

RuntimeDmaBufBacking RuntimeDmaBufBacking::createBacking(
        const vk::PhysicalDeviceIdentity& renderIdentity, bool image) {
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

    constexpr uint64_t requestedModifier = DRM_FORMAT_MOD_LINEAR;
    auto* bo = image
        ? gbm_bo_create_with_modifiers2(device, CONTROL_WIDTH, CONTROL_HEIGHT,
            GBM_FORMAT_ARGB8888, &requestedModifier, 1,
            GBM_BO_USE_RENDERING)
        : gbm_bo_create(device, CONTROL_WIDTH, CONTROL_HEIGHT, GBM_FORMAT_R8,
            GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING);
    if (!bo) {
        std::fprintf(stderr,
            "[DG2X-A3H2] Runtime %s backing request node=%s fd=%d device=%p "
            "width=%u height=%u fourcc=0x%x usage=0x%x errno=%d (%s)\n",
            image ? "image" : "legacy buffer", nodePath.c_str(), nodeFd.get(),
            static_cast<void*>(device), CONTROL_WIDTH, CONTROL_HEIGHT,
            image ? GBM_FORMAT_ARGB8888 : GBM_FORMAT_R8,
            GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING, errno, std::strerror(errno));
        gbm_device_destroy(device);
        throw ls::error("gbm backing allocation failed for runtime exchange backing errno="
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

    const auto fourcc = gbm_bo_get_format(bo);
    const auto modifier = gbm_bo_get_modifier(bo);
    const auto planes = gbm_bo_get_plane_count(bo);
    const auto stride = gbm_bo_get_stride_for_plane(bo, 0);
    const auto offset = gbm_bo_get_offset(bo, 0);
    if (image && (fourcc != DRM_FORMAT_ARGB8888 || modifier != DRM_FORMAT_MOD_LINEAR
            || planes != 1 || stride == 0)) {
        gbm_bo_destroy(bo);
        gbm_device_destroy(device);
        throw ls::error("GBM returned an incompatible runtime image backing");
    }
    return RuntimeDmaBufBacking(nodePath, std::move(nodeFd), device, bo,
        std::move(dmaBufFd), static_cast<VkDeviceSize>(end), fourcc, modifier,
        planes, stride, offset);
}

RuntimeDmaBufBacking RuntimeDmaBufBacking::create(
        const vk::PhysicalDeviceIdentity& renderIdentity) {
    return createBacking(renderIdentity, false);
}

RuntimeDmaBufBacking RuntimeDmaBufBacking::createImage(
        const vk::PhysicalDeviceIdentity& renderIdentity) {
    return createBacking(renderIdentity, true);
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

ls::OwnedFd RuntimeDmaBufBacking::duplicatePlaneFd(uint32_t plane) const {
    if (!this->gbmBo || plane >= this->boPlanes)
        throw ls::error("runtime DMA-BUF backing has no valid plane");
    const int planeFd = gbm_bo_get_fd_for_plane(this->gbmBo, plane);
    if (planeFd < 0)
        throw ls::error("failed to get runtime DMA-BUF plane fd errno="
            + std::to_string(errno));
    return ls::OwnedFd(planeFd);
}
