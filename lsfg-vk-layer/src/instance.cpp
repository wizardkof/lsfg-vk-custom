/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/paths.hpp"
#include "swapchain.hpp"
#include "runtime_dma_buf_backing.hpp"
#include "lsfg-vk-common/configuration/detection.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
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

        const auto pairResolution = vk::resolveRuntimeDevicePair(
            backendDevices, applicationIdentity, selector);
        const auto& selection = pairResolution.selection;
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

    [[nodiscard]] const char* runtimePairModeName(vk::RuntimeDevicePairMode mode) {
        switch (mode) {
            case vk::RuntimeDevicePairMode::SamePhysicalDevice:
                return "SAME_PHYSICAL_DEVICE";
            case vk::RuntimeDevicePairMode::CrossPhysicalDevice:
                return "CROSS_PHYSICAL_DEVICE";
        }
        return "UNKNOWN";
    }

    void logRuntimeDevicePair(const vk::RuntimeDevicePair& pair) {
        std::cerr << "[DG2X-P3C] Runtime device pair\n"
            << "  Render device: " << pair.render.identity.name << "\n"
            << "  Generation device: " << pair.generation.identity.name << "\n"
            << "  Mode: " << runtimePairModeName(pair.mode) << "\n"
            << "  Render logical-device owner: APPLICATION\n"
            << "  Generation logical-device owner: BACKEND\n"
            << "  Frame transport connected: NO\n";
    }

    void validateRuntimeExchangeChannel(const vk::Vulkan& renderVk,
            backend::Instance& backend, const vk::RuntimeDevicePair& pair,
            const RuntimeExchangeQueue& exchangeQueue) {
        constexpr VkDeviceSize CONTROL_BUFFER_SIZE = 64 * 1024;

        const auto generationSnapshot = std::ranges::find_if(
            backend.visibleDevices(),
            [&pair](const vk::PhysicalDeviceSnapshot& snapshot) {
                return snapshot.identity.samePhysicalDevice(pair.generation.identity);
            });
        if (generationSnapshot == backend.visibleDevices().end()
                || !extensionAdvertised(*generationSnapshot,
                    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME))
            throw ls::error(
                "generation GPU does not advertise VK_EXT_external_memory_dma_buf");

        std::filesystem::path allocatorNode;
        VkDeviceSize backingSize{};
        uint32_t renderMemoryType{};
        uint32_t generationMemoryType{};
        vk::RuntimeExchangeSyncDiagnostics sync{};

        {
            auto backing = RuntimeDmaBufBacking::create(pair.render.identity);
            allocatorNode = backing.allocatorNode();
            backingSize = backing.size();
            const vk::RuntimeExchangeChannelInfo info{
                .logicalSize = CONTROL_BUFFER_SIZE,
                .backingSize = backingSize,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };

            auto renderEndpoint = vk::makeRuntimeExchangeEndpoint(renderVk);
            renderEndpoint.queue = exchangeQueue.queue;
            renderEndpoint.queueFamilyIndex = exchangeQueue.familyIndex;

            auto channel = vk::createRuntimeExchangeChannel(
                pair,
                std::move(renderEndpoint),
                backend.runtimeExchangeEndpoint(),
                backing.duplicateFd(),
                backing.duplicateFd(),
                info);
            renderMemoryType = channel.renderBufferDiagnostics().memoryTypeIndex;
            generationMemoryType = channel.generationBufferDiagnostics().memoryTypeIndex;

            {
                const std::scoped_lock queueLock(*exchangeQueue.mutex);
                sync = channel.validateSyncRoundTrip();
            }
        } // channel first, then neutral GBM backing: teardown completed here.

        std::cerr << "[DG2X-P3D] Runtime cross-device exchange channel\n"
            << "  Allocator: GBM\n"
            << "  Allocator node: " << allocatorNode << "\n"
            << "  Logical size: " << CONTROL_BUFFER_SIZE << "\n"
            << "  DMA_BUF size: " << backingSize << "\n"
            << "  Render memoryType: " << renderMemoryType << "\n"
            << "  Generation memoryType: " << generationMemoryType << "\n"
            << "  Render->Generation SYNC_FD: "
            << (sync.renderToGenerationSentinel ? "SENTINEL_-1" : "FD") << "\n"
            << "  Generation->Render SYNC_FD: "
            << (sync.generationToRenderSentinel ? "SENTINEL_-1" : "FD") << "\n"
            << "  Host waits before final submit: "
            << (sync.hostWaitBeforeFinalSubmit ? "YES" : "NONE") << "\n"
            << "  Buffer import A/B: PASS\n"
            << "  SYNC_FD round trip A->B->A: PASS\n"
            << "  Safe teardown: PASS\n"
            << "  Frame transport connected: NO\n"
            << "DG2X_P3D_RUNTIME_EXCHANGE_CHANNEL_PASS\n";
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

    std::vector<const char*> requestedExtensions{
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_KHR_external_semaphore",
        "VK_KHR_external_semaphore_fd",
        "VK_KHR_timeline_semaphore"
    };
    // Cross-device transport is only reachable through an explicit generation-GPU
    // selector. Keep the Default/same-GPU device contract unchanged.
    if (snapshot.activeProfile().gpu.has_value())
        requestedExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);

    auto extensions = add_extensions(
        createInfo.ppEnabledExtensionNames,
        createInfo.enabledExtensionCount,
        requestedExtensions
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
        VkSwapchainKHR swapchain, const SwapchainInfo& info,
        std::optional<RuntimeExchangeQueue> exchangeQueue) {
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
        std::optional<vk::RuntimeDevicePairResolution> plannedPair;
        try {
            std::string dll{};
            if (global.dll.has_value())
                dll = *global.dll;
            else
                dll = ls::findShaderDll();

            const backend::DevicePicker picker{
                [&plannedPair](const vk::PhysicalDeviceIdentity& candidate) {
                    return plannedPair.has_value()
                        && plannedPair->pair.has_value()
                        && plannedPair->pair->matchesGenerationDevice(candidate);
                }
            };
            const backend::DeviceEnumerationObserver observer{
                [&backendDevices, &backendEnumerationObserved, &plannedPair,
                    applicationIdentity, generationSelector = profile.gpu](
                        const std::vector<vk::PhysicalDeviceSnapshot>& devices) {
                    backendDevices = devices;
                    backendEnumerationObserved = true;
                    plannedPair = vk::resolveRuntimeDevicePair(
                        devices, applicationIdentity, generationSelector);
                }
            };
            this->backend.emplace(
                picker,
                dll, global.allow_fp16,
                observer
            );

            const auto& generationIdentity = this->backend->deviceIdentity();
            if (!plannedPair.has_value() || !plannedPair->pair.has_value()
                    || !plannedPair->pair->matchesGenerationDevice(generationIdentity))
                throw ls::error(
                    "backend generation GPU does not match the resolved runtime device pair");

            logDeviceSelectionDiagnostic(applicationIdentity,
                backendDevices, backendEnumerationObserved,
                profile.gpu, generationIdentity);
            diagnosticLogged = true;
        } catch (const std::exception& e) {
            std::optional<vk::PhysicalDeviceIdentity> plannedGenerationIdentity;
            if (plannedPair.has_value() && plannedPair->pair.has_value())
                plannedGenerationIdentity = plannedPair->pair->generation.identity;
            if (!diagnosticLogged)
                logDeviceSelectionDiagnostic(applicationIdentity,
                    backendDevices, backendEnumerationObserved,
                    profile.gpu, plannedGenerationIdentity);
            unsetenv("DISABLE_LSFGVK");
            throw ls::error("failed to create backend instance", e);
        }

        unsetenv("DISABLE_LSFGVK");
    }

    const auto runtimePair = vk::bindRuntimeDevicePair(
        applicationIdentity, this->backend->deviceIdentity(), profile.gpu);
    if (!runtimePair.has_value())
        throw ls::error("runtime render/generation device pair contract mismatch");

    logRuntimeDevicePair(*runtimePair);
    if (runtimePair->crossDevice()) {
        if (!exchangeQueue.has_value() || !exchangeQueue->valid())
            throw ls::error(
                "cross-device runtime exchange requires a layer-managed offload queue");
        validateRuntimeExchangeChannel(
            vk, this->backend.mut(), *runtimePair, *exchangeQueue);
        throw ls::error(
            "cross-device runtime exchange channel validated, but frame transport is not connected yet");
    }

    this->swapchains.emplace(swapchain,
        Swapchain(vk, this->backend.mut(), *runtimePair, profile, info));
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
        VkSwapchainKHR swapchain, const SwapchainInfo& info,
        std::optional<RuntimeExchangeQueue> exchangeQueue) {
    if (!snapshot.active())
        throw ls::error("attempted to recreate swapchain context while layer is inactive");

    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);
    if (!this->backend.has_value())
        throw ls::error("attempted to recreate swapchain context without backend");

    const auto applicationIdentity = vk::getPhysicalDeviceIdentity(
        vk.fi(), vk.physdev());
    const auto& profile = snapshot.activeProfile();
    const auto runtimePair = vk::bindRuntimeDevicePair(
        applicationIdentity, this->backend->deviceIdentity(), profile.gpu);
    if (!runtimePair.has_value())
        throw ls::error("runtime render/generation device pair contract mismatch");
    if (runtimePair->crossDevice()) {
        if (!exchangeQueue.has_value() || !exchangeQueue->valid())
            throw ls::error(
                "cross-device runtime exchange requires a layer-managed offload queue");
        validateRuntimeExchangeChannel(
            vk, this->backend.mut(), *runtimePair, *exchangeQueue);
        throw ls::error(
            "cross-device runtime exchange channel validated, but frame transport is not connected yet");
    }

    this->swapchains.erase(swapchain);
    this->swapchains.emplace(swapchain,
        Swapchain(vk, this->backend.mut(), *runtimePair, profile, info));
}

void Root::removeSwapchainContext(VkSwapchainKHR swapchain) {
    const std::scoped_lock<std::mutex> lock(this->swapchainMutex);
    this->swapchains.erase(swapchain);
}
