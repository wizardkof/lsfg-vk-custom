/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

#include <vulkan/vulkan_core.h>

struct gbm_bo;
struct gbm_device;

namespace lsfgvk::layer {

    [[nodiscard]] std::string formatPciSysfsAddress(const vk::PciBusInfo& pci);

    [[nodiscard]] std::filesystem::path findRenderNodeForPci(
        const vk::PciBusInfo& pci,
        const std::filesystem::path& pciDevicesRoot = "/sys/bus/pci/devices",
        const std::filesystem::path& driRoot = "/dev/dri");

    /// Small neutral DMA-BUF allocation used only to bootstrap/validate the P3D
    /// runtime control channel. Frame image allocation remains a later stage.
    class RuntimeDmaBufBacking {
    public:
        RuntimeDmaBufBacking() noexcept = default;
        RuntimeDmaBufBacking(const RuntimeDmaBufBacking&) = delete;
        RuntimeDmaBufBacking& operator=(const RuntimeDmaBufBacking&) = delete;
        RuntimeDmaBufBacking(RuntimeDmaBufBacking&& other) noexcept;
        RuntimeDmaBufBacking& operator=(RuntimeDmaBufBacking&& other) noexcept;
        ~RuntimeDmaBufBacking();

        [[nodiscard]] static RuntimeDmaBufBacking create(
            const vk::PhysicalDeviceIdentity& renderIdentity);
        [[nodiscard]] static RuntimeDmaBufBacking createImage(
            const vk::PhysicalDeviceIdentity& renderIdentity);

        [[nodiscard]] VkDeviceSize size() const noexcept { return this->backingSize; }
        [[nodiscard]] uint32_t fourcc() const noexcept { return this->boFourcc; }
        [[nodiscard]] uint64_t modifier() const noexcept { return this->boModifier; }
        [[nodiscard]] uint32_t planeCount() const noexcept { return this->boPlanes; }
        [[nodiscard]] uint32_t stride() const noexcept { return this->boStride; }
        [[nodiscard]] uint32_t offset() const noexcept { return this->boOffset; }
        [[nodiscard]] ls::OwnedFd duplicateFd() const;
        [[nodiscard]] ls::OwnedFd duplicatePlaneFd(uint32_t plane) const;
        [[nodiscard]] const std::filesystem::path& allocatorNode() const noexcept {
            return this->nodePath;
        }

    private:
        [[nodiscard]] static RuntimeDmaBufBacking createBacking(
            const vk::PhysicalDeviceIdentity& renderIdentity, bool image);
        RuntimeDmaBufBacking(
            std::filesystem::path nodePath,
            ls::OwnedFd nodeFd,
            gbm_device* device,
            gbm_bo* bo,
            ls::OwnedFd dmaBufFd,
            VkDeviceSize size, uint32_t fourcc, uint64_t modifier,
            uint32_t planes, uint32_t stride, uint32_t offset) noexcept;

        void reset() noexcept;

        std::filesystem::path nodePath;
        ls::OwnedFd nodeFd;
        gbm_device* gbmDevice{};
        gbm_bo* gbmBo{};
        ls::OwnedFd dmaBufFd;
        VkDeviceSize backingSize{};
        uint32_t boFourcc{};
        uint64_t boModifier{};
        uint32_t boPlanes{};
        uint32_t boStride{};
        uint32_t boOffset{};
    };

}
