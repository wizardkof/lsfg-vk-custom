/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config_snapshot.hpp"
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "swapchain.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {
    class DeviceRetirementReactor;

    /// Layer-owned/safely-shareable queue reserved during VkDevice creation.
    /// P3D uses it for the one-shot A -> B -> A control-channel handshake instead
    /// of submitting through the application's ordinary queue without synchronization.
    struct RuntimeExchangeQueue {
        VkQueue queue{VK_NULL_HANDLE};
        uint32_t familyIndex{};
        std::shared_ptr<std::mutex> mutex;

        [[nodiscard]] bool valid() const noexcept {
            return this->queue != VK_NULL_HANDLE && this->mutex;
        }
    };

    /// root context of the lsfg-vk layer
    class Root {
    public:
        /// create the lsfg-vk root context
        /// @throws ls::error on failure
        Root();

        /// check if the layer is active
        /// @return true if active
        [[nodiscard]] bool active() const { return this->snapshot().active(); }

        /// check if the active profile uses fixed-target frame generation
        /// @return true when fixed mode is active
        [[nodiscard]] bool fixedMode() const { return this->snapshot().fixedMode(); }

        /// check if the active profile is the Adaptive 1x bypass
        /// @return true when Adaptive multiplier=1 is active
        [[nodiscard]] bool adaptiveBypass() const { return this->snapshot().adaptiveBypass(); }

        /// capture one immutable configuration view for a complete operation
        /// @return current effective configuration and its revision
        [[nodiscard]] ConfigSnapshot snapshot() const;

        /// ensure the layer is up-to-date
        /// @return new effective snapshot, or empty when nothing applicable changed
        std::optional<ConfigSnapshot> update();

        /// modify instance create info
        /// @param snapshot immutable configuration for this operation
        /// @param createInfo original create info
        /// @param finish function to call after modification
        static void modifyInstanceCreateInfo(const ConfigSnapshot& snapshot,
            VkInstanceCreateInfo& createInfo,
            const std::function<void(void)>& finish);
        /// modify device create info
        /// @param snapshot immutable configuration for this operation
        /// @param createInfo original create info
        /// @param finish function to call after modification
        static void modifyDeviceCreateInfo(const ConfigSnapshot& snapshot,
            VkDeviceCreateInfo& createInfo,
            const std::function<void(void)>& finish);

        /// modify swapchain create info
        /// @param snapshot immutable configuration for this operation
        /// @param vk vulkan instance
        /// @param createInfo original create info
        /// @param finish function to call after modification
        static void modifySwapchainCreateInfo(const ConfigSnapshot& snapshot,
            const vk::Vulkan& vk, VkSwapchainCreateInfoKHR& createInfo,
            const std::function<void(void)>& finish);
        /// create swapchain context
        /// @param snapshot immutable configuration for this operation
        /// @param vk vulkan instance
        /// @param swapchain swapchain handle
        /// @param info swapchain info
        /// @throws ls::error on failure
        void createSwapchainContext(const ConfigSnapshot& snapshot,
            const vk::Vulkan& vk, VkSwapchainKHR swapchain,
            const SwapchainInfo& info,
            std::optional<RuntimeExchangeQueue> exchangeQueue = std::nullopt);
        /// get swapchain context
        /// @param swapchain swapchain handle
        /// @return swapchain context
        /// @throws ls::error if not found
        [[nodiscard]] Swapchain& getSwapchainContext(VkSwapchainKHR swapchain) {
            const auto& it = this->swapchains.find(swapchain);
            if (it == this->swapchains.end())
                throw ls::error("swapchain context not found");

            return *it->second;
        }
        /// present through a swapchain context while protecting it from hot reload
        SwapchainPresentResult presentSwapchain(const vk::Vulkan& vk,
            VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
            VkSwapchainKHR swapchain, void* nextChain, uint32_t imageIndex,
            const std::vector<VkSemaphore>& semaphores,
            std::stop_token stopToken = {},
            std::optional<std::chrono::steady_clock::time_point> sourcePresentTime = std::nullopt,
            bool d3bSingleSwapchainEligible = false,
            const GraphicsFinalQueueInfo* graphicsFinalQueue = nullptr,
            BorrowedGraphicsQueueLease* graphicsLease = nullptr,
            bool* stopAfterCompletion = nullptr,
            std::shared_ptr<D3B3DeviceLifetimeQuarantine> deviceQuarantine = {},
            std::shared_ptr<DeviceRetirementReactor> deviceRetirementReactor = {},
            std::shared_ptr<PresentedPhysicalImageLeaseRegistry> physicalImageLeases = {},
            std::shared_ptr<BorrowedPresentFenceRegistry> borrowedPresentFences = {},
            std::unique_ptr<VirtualPresentPendingOperation>* pendingCompletion = nullptr,
            PresentedPhysicalImageIdentity* presentedIdentity = nullptr,
            std::shared_ptr<void> virtualGpuBacking = {});
        /// atomically replace a swapchain context after a configuration reload
        /// @param snapshot immutable configuration for this reload
        void recreateSwapchainContext(const ConfigSnapshot& snapshot,
            const vk::Vulkan& vk, VkSwapchainKHR swapchain,
            const SwapchainInfo& info,
            std::optional<RuntimeExchangeQueue> exchangeQueue = std::nullopt);
        /// remove swapchain context
        /// @param swapchain swapchain handle
        void removeSwapchainContext(VkSwapchainKHR swapchain);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        /// Install the concrete context used by same-module entrypoint tests.
        /// This does not exist in tests-OFF builds and does not select an
        /// alternate presentation implementation.
        [[nodiscard]] bool installSwapchainContextForTesting(
            VkSwapchainKHR swapchain, std::shared_ptr<Swapchain> context);
        void setProfileForTesting(ls::GameConf profile);
        [[nodiscard]] size_t swapchainContextCountForTesting();
#endif
    private:
        mutable std::mutex configMutex;
        ls::WatchedConfig config;
        ConfigSnapshotState configState;

        ls::lazy<backend::Instance> backend;
        std::mutex swapchainMutex;
        std::unordered_map<VkSwapchainKHR, std::shared_ptr<Swapchain>> swapchains;
    };

}
