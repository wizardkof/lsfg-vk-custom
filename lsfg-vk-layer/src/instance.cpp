/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/paths.hpp"
#include "swapchain.hpp"
#include "lsfg-vk-common/configuration/detection.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>
#include <vulkan/vulkan_core.h>

using namespace lsfgvk;
using namespace lsfgvk::layer;

namespace {
    /// helper function to add required extensions
    std::vector<const char*> add_extensions(const char* const* existingExtensions, size_t count,
            const std::vector<const char*>& requiredExtensions) {
        std::vector<const char*> extensions(count);
        std::copy_n(existingExtensions, count, extensions.data());

        for (const auto& requiredExtension : requiredExtensions) {
            auto it = std::ranges::find_if(extensions,
                [requiredExtension](const char* extension) {
                    return std::string(extension) == std::string(requiredExtension);
                });
            if (it == extensions.end())
                extensions.push_back(requiredExtension);
        }

        return extensions;
    }
}

Root::Root() {
    // find active profile
    const auto& profile = findProfile(this->config.get(), ls::identify());
    const auto selection = this->configState.select(
        profile.has_value() ? std::optional<ls::GameConf>(profile->second) : std::nullopt,
        this->config.get().global());
    if (selection != ConfigSnapshotState::Selection::Applied
            || !profile.has_value())
        return;

    const auto selected = this->configState.snapshot();

    std::cerr << "lsfg-vk: using profile with name '"
        << selected.activeProfile().name << "' ";
    switch (profile->first) {
        case ls::IdentType::OVERRIDE:
            std::cerr << "(identified via override)\n";
            break;
        case ls::IdentType::EXECUTABLE:
            std::cerr << "(identified via executable)\n";
            break;
        case ls::IdentType::WINE_EXECUTABLE:
            std::cerr << "(identified via wine executable)\n";
            break;
        case ls::IdentType::PROCESS_NAME:
            std::cerr << "(identified via process name)\n";
            break;
    }
}

ConfigSnapshot Root::snapshot() const {
    const std::scoped_lock<std::mutex> lock(this->configMutex);
    return this->configState.snapshot();
}

std::optional<ConfigSnapshot> Root::update() {
    const std::scoped_lock<std::mutex> lock(this->configMutex);
    if (!this->config.update())
        return std::nullopt;

    const auto& profile = findProfile(this->config.get(), ls::identify());
    const auto selection = this->configState.select(
        profile.has_value() ? std::optional<ls::GameConf>(profile->second) : std::nullopt,
        this->config.get().global());
    if (selection == ConfigSnapshotState::Selection::RetainedActiveProfile) {
        std::cerr << "lsfg-vk: active profile was removed from the configuration; "
            "keeping the last active profile until the process restarts\n";
        return std::nullopt;
    }
    if (selection == ConfigSnapshotState::Selection::Inactive)
        return std::nullopt;

    return this->configState.snapshot();
}

void Root::modifyInstanceCreateInfo(const ConfigSnapshot& snapshot,
        VkInstanceCreateInfo& createInfo,
        const std::function<void(void)>& finish) {
    if (!snapshot.active()) {
        finish();
        return;
    }

    auto extensions = add_extensions(
        createInfo.ppEnabledExtensionNames,
        createInfo.enabledExtensionCount,
        {
            "VK_KHR_get_physical_device_properties2",
            "VK_KHR_external_memory_capabilities",
            "VK_KHR_external_semaphore_capabilities"
        }
    );
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    finish();
}

void Root::modifyDeviceCreateInfo(const ConfigSnapshot& snapshot,
        VkDeviceCreateInfo& createInfo,
        const std::function<void(void)>& finish) {
    if (!snapshot.active()) {
        finish();
        return;
    }

    auto extensions = add_extensions(
        createInfo.ppEnabledExtensionNames,
        createInfo.enabledExtensionCount,
        {
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_fd",
            "VK_KHR_external_semaphore",
            "VK_KHR_external_semaphore_fd",
            "VK_KHR_timeline_semaphore"
        }
    );
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    bool isFeatureEnabled = false;
    auto* featureInfo = reinterpret_cast<VkBaseInStructure*>(const_cast<void*>(createInfo.pNext));
    while (featureInfo) {
        if (featureInfo->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            auto* features = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(featureInfo);
            features->timelineSemaphore = VK_TRUE;
            isFeatureEnabled = true;
        } else if (featureInfo->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
            auto* features = reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(featureInfo);
            features->timelineSemaphore = VK_TRUE;
            isFeatureEnabled = true;
        }

        featureInfo = const_cast<VkBaseInStructure*>(featureInfo->pNext);
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = const_cast<void*>(createInfo.pNext),
        .timelineSemaphore = VK_TRUE
    };
    if (!isFeatureEnabled)
        createInfo.pNext = &timelineFeatures;

    finish();
}

void Root::modifySwapchainCreateInfo(const ConfigSnapshot& snapshot,
        const vk::Vulkan& vk, VkSwapchainCreateInfoKHR& createInfo,
        const std::function<void(void)>& finish) {
    if (!snapshot.active()) {
        finish();
        return;
    }

    VkSurfaceCapabilitiesKHR caps{}; // NOLINT (enum value 0)
    auto res = vk.fi().GetPhysicalDeviceSurfaceCapabilitiesKHR(
        vk.physdev(), createInfo.surface, &caps);
    if (res != VK_SUCCESS)
        throw ls::vulkan_error(res, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR() failed");

    context_ModifySwapchainCreateInfo(
        snapshot.activeProfile(), caps.maxImageCount, createInfo);

    finish();
}

void Root::createSwapchainContext(const ConfigSnapshot& snapshot,
        const vk::Vulkan& vk,
        VkSwapchainKHR swapchain, const SwapchainInfo& info) {
    if (!snapshot.active())
        throw ls::error("attempted to create swapchain context while layer is inactive");
    const auto& profile = snapshot.activeProfile();
    const auto& global = snapshot.global;

    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);

    if (!this->backend.has_value()) { // emplace backend late, due to loader bug
        setenv("DISABLE_LSFGVK", "1", 1);

        try {
            std::string dll{};
            if (global.dll.has_value())
                dll = *global.dll;
            else
                dll = ls::findShaderDll();

            this->backend.emplace(
                [gpu = profile.gpu](
                    const std::string& deviceName,
                    std::pair<const std::string&, const std::string&> ids,
                    const std::optional<std::string>& pci
                ) {
                    if (!gpu)
                        return true;

                    return (deviceName == *gpu)
                        || (ids.first + ":" + ids.second == *gpu)
                        || (pci && *pci == *gpu);
                },
                dll, global.allow_fp16
            );
        } catch (const std::exception& e) {
            unsetenv("DISABLE_LSFGVK");
            throw ls::error("failed to create backend instance", e);
        }

        unsetenv("DISABLE_LSFGVK");
    }

    this->swapchains.emplace(swapchain,
        Swapchain(vk, this->backend.mut(), profile, info));
}

VkResult Root::presentSwapchain(const vk::Vulkan& vk,
        VkQueue queue, std::shared_ptr<std::mutex> queueMutex,
        VkSwapchainKHR swapchain, void* nextChain, uint32_t imageIndex,
        const std::vector<VkSemaphore>& semaphores,
        std::stop_token stopToken,
        std::optional<std::chrono::steady_clock::time_point> sourcePresentTime) {
    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);
    const auto it = this->swapchains.find(swapchain);
    if (it == this->swapchains.end())
        throw ls::error("swapchain context not found");

    return it->second.present(vk, queue, std::move(queueMutex), swapchain,
        nextChain, imageIndex, semaphores, std::move(stopToken), sourcePresentTime);
}

void Root::recreateSwapchainContext(const ConfigSnapshot& snapshot,
        const vk::Vulkan& vk,
        VkSwapchainKHR swapchain, const SwapchainInfo& info) {
    if (!snapshot.active())
        throw ls::error("attempted to recreate swapchain context while layer is inactive");

    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);
    if (!this->backend.has_value())
        throw ls::error("attempted to recreate swapchain context without backend");

    this->swapchains.erase(swapchain);
    this->swapchains.emplace(swapchain,
        Swapchain(vk, this->backend.mut(), snapshot.activeProfile(), info));
}

void Root::removeSwapchainContext(VkSwapchainKHR swapchain) {
    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);
    this->swapchains.erase(swapchain);
}
