/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    /// root context of the lsfg-vk layer
    class Root {
    public:
        /// create the lsfg-vk root context
        /// @throws ls::error on failure
        Root();

        /// check if the layer is active
        /// @return true if active
        [[nodiscard]] bool active() const { return this->active_profile.has_value(); }

        /// check if the active profile uses fixed-target frame generation
        /// @return true when fixed mode is active
        [[nodiscard]] bool fixedMode() const {
            return this->active_profile.has_value()
                && this->active_profile->frame_generation_mode == ls::FrameGenerationMode::Fixed;
        }

        /// check if the active profile is the Adaptive 1x bypass
        /// @return true when Adaptive multiplier=1 is active
        [[nodiscard]] bool adaptiveBypass() const {
            return this->active_profile.has_value()
                && this->active_profile->frame_generation_mode != ls::FrameGenerationMode::Fixed
                && this->active_profile->multiplier == 1;
        }

        /// ensure the layer is up-to-date
        /// @return true if the configuration was updated
        bool update();

        /// modify instance create info
        /// @param createInfo original create info
        /// @param finish function to call after modification
        void modifyInstanceCreateInfo(VkInstanceCreateInfo& createInfo,
            const std::function<void(void)>& finish) const;
        /// modify device create info
        /// @param createInfo original create info
        /// @param finish function to call after modification
        void modifyDeviceCreateInfo(VkDeviceCreateInfo& createInfo,
            const std::function<void(void)>& finish) const;

        /// modify swapchain create info
        /// @param vk vulkan instance
        /// @param createInfo original create info
        /// @param finish function to call after modification
        void modifySwapchainCreateInfo(const vk::Vulkan& vk, VkSwapchainCreateInfoKHR& createInfo,
            const std::function<void(void)>& finish) const;
        /// create swapchain context
        /// @param vk vulkan instance
        /// @param swapchain swapchain handle
        /// @param info swapchain info
        /// @throws ls::error on failure
        void createSwapchainContext(const vk::Vulkan& vk, VkSwapchainKHR swapchain,
            const SwapchainInfo& info);
        /// get swapchain context
        /// @param swapchain swapchain handle
        /// @return swapchain context
        /// @throws ls::error if not found
        [[nodiscard]] Swapchain& getSwapchainContext(VkSwapchainKHR swapchain) {
            const auto& it = this->swapchains.find(swapchain);
            if (it == this->swapchains.end())
                throw ls::error("swapchain context not found");

            return it->second;
        }
        /// present through a swapchain context while protecting it from hot reload
        VkResult presentSwapchain(const vk::Vulkan& vk,
            VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
            VkSwapchainKHR swapchain, void* nextChain, uint32_t imageIndex,
            const std::vector<VkSemaphore>& semaphores,
            std::stop_token stopToken = {},
            std::optional<std::chrono::steady_clock::time_point> sourcePresentTime = std::nullopt);
        /// atomically replace a swapchain context after a configuration reload
        void recreateSwapchainContext(const vk::Vulkan& vk, VkSwapchainKHR swapchain,
            const SwapchainInfo& info);
        /// remove swapchain context
        /// @param swapchain swapchain handle
        void removeSwapchainContext(VkSwapchainKHR swapchain);
    private:
        ls::WatchedConfig config;
        std::optional<ls::GameConf> active_profile;

        ls::lazy<backend::Instance> backend;
        std::mutex swapchainMutex;
        std::unordered_map<VkSwapchainKHR, Swapchain> swapchains;
    };

}
