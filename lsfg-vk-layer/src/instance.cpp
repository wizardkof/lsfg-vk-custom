/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/paths.hpp"
#include "swapchain.hpp"
#include "lsfg-vk-common/configuration/detection.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
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

    [[nodiscard]] const char* yesNo(bool value) {
        return value ? "YES" : "NO";
    }

    [[nodiscard]] const char* selectionResolutionName(
            vk::DeviceSelectionResolution resolution) {
        switch (resolution) {
            case vk::DeviceSelectionResolution::Selected:
                return "Selected";
            case vk::DeviceSelectionResolution::ApplicationDeviceNotVisibleToBackend:
                return "ApplicationDeviceNotVisibleToBackend";
            case vk::DeviceSelectionResolution::SelectorNotFoundInBackend:
                return "SelectorNotFoundInBackend";
            case vk::DeviceSelectionResolution::SelectorAmbiguous:
                return "SelectorAmbiguous";
        }
        return "Unknown";
    }

    [[nodiscard]] const char* driverRelationshipName(
            vk::DriverUuidRelationship relationship) {
        switch (relationship) {
            case vk::DriverUuidRelationship::Same:
                return "YES";
            case vk::DriverUuidRelationship::Different:
                return "NO";
            case vk::DriverUuidRelationship::Unknown:
                return "UNKNOWN";
        }
        return "UNKNOWN";
    }

    void logPhysicalDeviceIdentity(const vk::PhysicalDeviceIdentity& identity) {
        std::cerr << "  Name: " << identity.name << "\n"
            << "  Device UUID: " << vk::formatUuid(identity.deviceUuid) << "\n"
            << "  Driver UUID: " << vk::formatUuid(identity.driverUuid) << "\n"
            << "  PCI: " << (identity.pci.has_value()
                ? identity.pci->identifier() : "<unavailable>") << "\n";
    }

    [[nodiscard]] bool extensionAdvertised(
            const vk::PhysicalDeviceSnapshot& snapshot,
            const char* extensionName) {
        return std::ranges::binary_search(
            snapshot.advertisedDeviceExtensions, std::string(extensionName));
    }

    void logVulkanEnvironment() {
        constexpr std::array<const char*, 8> VARIABLES{
            "DRI_PRIME",
            "MESA_VK_DEVICE_SELECT",
            "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE",
            "VK_DRIVER_FILES",
            "VK_ICD_FILENAMES",
            "VK_ADD_DRIVER_FILES",
            "VK_LOADER_DRIVERS_SELECT",
            "VK_LOADER_DRIVERS_DISABLE"
        };

        std::cerr << "[DG2A] Vulkan environment\n";
        for (const auto* variable : VARIABLES) {
            const char* value = std::getenv(variable);
            std::cerr << "  " << variable << ": "
                << (value && *value != '\0' ? value : "<unset>") << "\n";
        }
    }

    void logDeviceSelectionDiagnostic(
            const vk::PhysicalDeviceIdentity& applicationIdentity,
            const std::vector<vk::PhysicalDeviceSnapshot>& backendDevices,
            bool backendEnumerationObserved,
            const std::optional<std::string>& selector,
            const std::optional<vk::PhysicalDeviceIdentity>& generationIdentity) {
        std::cerr << "[DG2A] Application GPU\n";
        logPhysicalDeviceIdentity(applicationIdentity);

        if (!backendEnumerationObserved) {
            std::cerr << "[DG2A] Backend-visible devices: <enumeration unavailable>\n"
                << "[DG2A] Selection\n"
                << "  Requested: " << (selector.has_value() ? *selector : "Default") << "\n"
                << "  Resolution: <unavailable>\n"
                << "  Same physical device: UNKNOWN\n"
                << "  Same driver UUID: UNKNOWN\n"
                << "  Cross-GPU required: UNKNOWN\n";
            logVulkanEnvironment();
            return;
        }

        std::cerr << "[DG2A] Backend-visible devices: " << backendDevices.size() << "\n";
        for (size_t i = 0; i < backendDevices.size(); ++i) {
            const auto& snapshot = backendDevices.at(i);
            std::cerr << "[" << i << "]\n";
            logPhysicalDeviceIdentity(snapshot.identity);
            constexpr std::array<const char*, 7> RELEVANT_EXTENSIONS{
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
                VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME
            };
            for (const auto* extension : RELEVANT_EXTENSIONS)
                std::cerr << "  " << extension << " advertised: "
                    << yesNo(extensionAdvertised(snapshot, extension)) << "\n";
        }

        const auto selection = vk::resolveDeviceSelection(
            backendDevices, applicationIdentity, selector);
        std::cerr << "[DG2A] Selection\n"
            << "  Requested: " << (selector.has_value() ? *selector : "Default") << "\n"
            << "  Resolution: " << selectionResolutionName(selection.resolution) << "\n";
        if (selection.resolution == vk::DeviceSelectionResolution::SelectorAmbiguous)
            std::cerr << "  Matching backend devices: " << selection.matchCount
                << " (existing first-match behavior preserved)\n";

        if (generationIdentity.has_value()) {
            const bool samePhysicalDevice = applicationIdentity.samePhysicalDevice(
                *generationIdentity);
            std::cerr << "  Same physical device: " << yesNo(samePhysicalDevice) << "\n"
                << "  Same driver UUID: " << driverRelationshipName(
                    vk::compareDriverUuids(applicationIdentity, *generationIdentity)) << "\n"
                << "  Cross-GPU required: " << yesNo(!samePhysicalDevice) << "\n";
        } else {
            std::cerr << "  Same physical device: UNKNOWN\n"
                << "  Same driver UUID: UNKNOWN\n"
                << "  Cross-GPU required: UNKNOWN\n";
        }
        logVulkanEnvironment();
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
    const auto applicationIdentity = vk::getPhysicalDeviceIdentity(
        vk.fi(), vk.physdev());

    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);

    if (!this->backend.has_value()) { // emplace backend late, due to loader bug
        setenv("DISABLE_LSFGVK", "1", 1);

        std::vector<vk::PhysicalDeviceSnapshot> backendDevices;
        bool backendEnumerationObserved{false};
        bool diagnosticLogged{false};
        std::optional<vk::PhysicalDeviceIdentity> selectedIdentity;
        try {
            std::string dll{};
            if (global.dll.has_value())
                dll = *global.dll;
            else
                dll = ls::findShaderDll();

            const backend::DevicePicker picker{
                [gpu = profile.gpu, applicationIdentity, &selectedIdentity](
                    const vk::PhysicalDeviceIdentity& candidate) {
                    const bool selected = !gpu
                        ? candidate.samePhysicalDevice(applicationIdentity)
                        : candidate.matchesSelector(*gpu);
                    if (selected && !selectedIdentity.has_value())
                        selectedIdentity = candidate;
                    return selected;
                }
            };
            const backend::DeviceEnumerationObserver observer{
                [&backendDevices, &backendEnumerationObserved](
                        const std::vector<vk::PhysicalDeviceSnapshot>& devices) {
                    backendDevices = devices;
                    backendEnumerationObserved = true;
                }
            };
            this->backend.emplace(
                picker,
                dll, global.allow_fp16,
                observer
            );

            const auto& generationIdentity = this->backend->deviceIdentity();
            logDeviceSelectionDiagnostic(applicationIdentity,
                backendDevices, backendEnumerationObserved,
                profile.gpu, generationIdentity);
            diagnosticLogged = true;
        } catch (const std::exception& e) {
            if (!diagnosticLogged)
                logDeviceSelectionDiagnostic(applicationIdentity,
                    backendDevices, backendEnumerationObserved,
                    profile.gpu, selectedIdentity);
            unsetenv("DISABLE_LSFGVK");
            throw ls::error("failed to create backend instance", e);
        }

        unsetenv("DISABLE_LSFGVK");
    }

    if (!profile.gpu
            && !this->backend->deviceIdentity().samePhysicalDevice(applicationIdentity))
        throw ls::error("default frame generation GPU does not match the application GPU");

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
