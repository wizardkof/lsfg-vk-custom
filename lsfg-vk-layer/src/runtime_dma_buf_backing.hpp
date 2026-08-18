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

        [[nodiscard]] ls::OwnedFd duplicateFd() const;
        [[nodiscard]] VkDeviceSize size() const noexcept { return this->backingSize; }
        [[nodiscard]] const std::filesystem::path& allocatorNode() const noexcept {
            return this->nodePath;
        }

    private:
        RuntimeDmaBufBacking(
            std::filesystem::path nodePath,
            ls::OwnedFd nodeFd,
            gbm_device* device,
            gbm_bo* bo,
            ls::OwnedFd dmaBufFd,
            VkDeviceSize size) noexcept;

        void reset() noexcept;

        std::filesystem::path nodePath;
        ls::OwnedFd nodeFd;
        gbm_device* gbmDevice{};
        gbm_bo* gbmBo{};
        ls::OwnedFd dmaBufFd;
        VkDeviceSize backingSize{};
    };

}
