/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "adaptive_1x_preparation_reservation.hpp"
#include "batch_application_present_bridge_authority.hpp"
#include "batch_present_transaction.hpp"
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
#include "entrypoint_test_seam.hpp"
#endif
#include "present_batch_projection.hpp"
#include "aborted_present_semantics.hpp"
#include "borrowed_present_fence_registry.hpp"
#include "device_retirement_reactor.hpp"
#include "d2_vulkan_shadow_runtime.hpp"
#include "fixed_present_mode.hpp"
#include "lsfg-vk-common/configuration/config.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_extension_contract.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "swapchain.hpp"
#include "virtual_swapchain_image_spec.hpp"
#include "virtual_swapchain_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

namespace {
    // global layer info initialized at layer negotiation
    struct LayerInfo {
        std::unordered_map<std::string, PFN_vkVoidFunction> map; //!< function pointer override map
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr;

        Root root;
    }* layer_info; // NOLINT (global variable)

    struct OffloadQueueInfo {
        VkQueue queue{VK_NULL_HANDLE};
        uint32_t familyIndex{};
        uint32_t queueIndex{};
        uint32_t applicationQueueIndex{};
        bool sharedWithApplication{};
        std::shared_ptr<std::mutex> mutex{std::make_shared<std::mutex>()};
    };

    enum class QueueReservationMode {
        None,
        DedicatedSpare,
        DedicatedAuxiliary,
        SharedInternallySynchronized
    };

    enum class SwapchainMaintenanceFamily {
        None,
        KHR,
        EXT
    };

    constexpr const char* SURFACE_MAINTENANCE_KHR = "VK_KHR_surface_maintenance1";
    constexpr const char* SURFACE_MAINTENANCE_EXT = "VK_EXT_surface_maintenance1";
    constexpr const char* SWAPCHAIN_MAINTENANCE_KHR = "VK_KHR_swapchain_maintenance1";
    constexpr const char* SWAPCHAIN_MAINTENANCE_EXT = "VK_EXT_swapchain_maintenance1";
    constexpr const char* GET_SURFACE_CAPABILITIES_2_KHR = "VK_KHR_get_surface_capabilities2";
    constexpr const char* SURFACE_KHR = "VK_KHR_surface";
    constexpr const char* GET_MEMORY_REQUIREMENTS_2_KHR = "VK_KHR_get_memory_requirements2";
    constexpr const char* DEDICATED_ALLOCATION_KHR = "VK_KHR_dedicated_allocation";
    constexpr const char* GET_PHYSICAL_DEVICE_PROPERTIES_2_KHR =
        "VK_KHR_get_physical_device_properties2";
    constexpr const char* IMAGE_DRM_FORMAT_MODIFIER_EXT =
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME;

    [[nodiscard]] const char* maintenanceFamilyName(SwapchainMaintenanceFamily family) {
        switch (family) {
            case SwapchainMaintenanceFamily::KHR:
                return "KHR";
            case SwapchainMaintenanceFamily::EXT:
                return "EXT";
            case SwapchainMaintenanceFamily::None:
                return "none";
        }
        return "none";
    }

    [[nodiscard]] const char* surfaceMaintenanceExtension(
            SwapchainMaintenanceFamily family) {
        return family == SwapchainMaintenanceFamily::KHR
            ? SURFACE_MAINTENANCE_KHR
            : SURFACE_MAINTENANCE_EXT;
    }

    [[nodiscard]] const char* swapchainMaintenanceExtension(
            SwapchainMaintenanceFamily family) {
        return family == SwapchainMaintenanceFamily::KHR
            ? SWAPCHAIN_MAINTENANCE_KHR
            : SWAPCHAIN_MAINTENANCE_EXT;
    }

    struct QueueReservation {
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        std::vector<float> priorities;
        std::vector<const char*> enabledExtensions;
        std::optional<uint32_t> queueInfoIndex;
        std::optional<uint32_t> internalQueueInfoIndex;
        uint32_t familyIndex{};
        uint32_t queueIndex{};
        uint32_t applicationQueueIndex{};
        float internalPriority{1.0F};
        QueueReservationMode mode{QueueReservationMode::None};
        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internalSyncFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR,
            .pNext = nullptr,
            .internallySynchronizedQueues = VK_TRUE
        };

        [[nodiscard]] bool valid() const {
            return this->mode != QueueReservationMode::None
                && this->queueInfoIndex.has_value();
        }
        [[nodiscard]] bool shared() const {
            return this->mode == QueueReservationMode::SharedInternallySynchronized;
        }

        void apply(VkDeviceCreateInfo& info) {
            if (!this->valid())
                return;

            if (this->mode == QueueReservationMode::DedicatedSpare) {
                this->queueInfos.at(*this->queueInfoIndex).pQueuePriorities =
                    this->priorities.data();
            } else if (this->mode == QueueReservationMode::DedicatedAuxiliary) {
                this->queueInfos.at(*this->queueInfoIndex).pQueuePriorities =
                    &this->internalPriority;
            } else if (this->shared()) {
                this->queueInfos.at(*this->internalQueueInfoIndex).pQueuePriorities =
                    &this->internalPriority;

                this->enabledExtensions.clear();
                if (info.enabledExtensionCount && info.ppEnabledExtensionNames) {
                    this->enabledExtensions.assign(
                        info.ppEnabledExtensionNames,
                        info.ppEnabledExtensionNames + info.enabledExtensionCount);
                }
                const auto ext = std::ranges::find_if(this->enabledExtensions,
                    [](const char* name) {
                        return name && std::string(name)
                            == VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME;
                    });
                if (ext == this->enabledExtensions.end())
                    this->enabledExtensions.push_back(
                        VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
                info.enabledExtensionCount =
                    static_cast<uint32_t>(this->enabledExtensions.size());
                info.ppEnabledExtensionNames = this->enabledExtensions.data();

                bool featureAlreadyPresent{};
                auto* current = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
                while (current) {
                    if (current->sType
                            == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR) {
                        featureAlreadyPresent = true;
                        break;
                    }
                    current = current->pNext;
                }
                if (!featureAlreadyPresent) {
                    this->internalSyncFeatures.pNext = const_cast<void*>(info.pNext);
                    info.pNext = &this->internalSyncFeatures;
                }
            }

            info.queueCreateInfoCount = static_cast<uint32_t>(this->queueInfos.size());
            info.pQueueCreateInfos = this->queueInfos.data();
        }
    };

    [[nodiscard]] bool extensionEnabled(
            const char* const* names, uint32_t count, const char* extensionName) {
        if (!count || !names)
            return false;
        for (uint32_t i = 0; i < count; ++i) {
            if (names[i] && std::string(names[i]) == extensionName)
                return true;
        }
        return false;
    }

    void appendExtension(std::vector<const char*>& extensions, const char* name) {
        if (!extensionEnabled(extensions.data(),
                static_cast<uint32_t>(extensions.size()), name))
            extensions.push_back(name);
    }

    [[nodiscard]] bool hasInstanceExtension(
            PFN_vkGetInstanceProcAddr getInstanceProcAddr, const char* extensionName) {
        if (!getInstanceProcAddr)
            return false;
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
        if (!enumerate)
            return false;

        uint32_t count{};
        auto res = enumerate(nullptr, &count, nullptr);
        if (res != VK_SUCCESS || !count)
            return false;

        std::vector<VkExtensionProperties> extensions(count);
        res = enumerate(nullptr, &count, extensions.data());
        if (res != VK_SUCCESS && res != VK_INCOMPLETE)
            return false;
        extensions.resize(count);

        return std::ranges::any_of(extensions,
            [extensionName](const VkExtensionProperties& extension) {
                return std::string(extension.extensionName) == extensionName;
            });
    }

    [[nodiscard]] SwapchainMaintenanceFamily selectSurfaceMaintenanceFamily(
            PFN_vkGetInstanceProcAddr getInstanceProcAddr,
            const VkInstanceCreateInfo& info,
            bool layerActive) {
        if (!layerActive
                || !extensionEnabled(info.ppEnabledExtensionNames,
                    info.enabledExtensionCount, SURFACE_KHR)
                || !hasInstanceExtension(getInstanceProcAddr,
                    GET_SURFACE_CAPABILITIES_2_KHR))
            return SwapchainMaintenanceFamily::None;

        if (hasInstanceExtension(getInstanceProcAddr, SURFACE_MAINTENANCE_KHR))
            return SwapchainMaintenanceFamily::KHR;
        if (hasInstanceExtension(getInstanceProcAddr, SURFACE_MAINTENANCE_EXT))
            return SwapchainMaintenanceFamily::EXT;
        return SwapchainMaintenanceFamily::None;
    }

    [[nodiscard]] bool hasDeviceExtension(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
            const char* extensionName) {
        uint32_t count{};
        auto res = funcs.EnumerateDeviceExtensionProperties(
            physdev, nullptr, &count, nullptr);
        if (res != VK_SUCCESS || !count)
            return false;

        std::vector<VkExtensionProperties> extensions(count);
        res = funcs.EnumerateDeviceExtensionProperties(
            physdev, nullptr, &count, extensions.data());
        if (res != VK_SUCCESS && res != VK_INCOMPLETE)
            return false;
        extensions.resize(count);

        return std::ranges::any_of(extensions,
            [extensionName](const VkExtensionProperties& extension) {
                return std::string(extension.extensionName) == extensionName;
            });
    }

    [[nodiscard]] bool supportsSwapchainMaintenance(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
            PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2,
            SwapchainMaintenanceFamily family) {
        if (family == SwapchainMaintenanceFamily::None
                || !getPhysicalDeviceFeatures2
                || !hasDeviceExtension(physdev, funcs,
                    swapchainMaintenanceExtension(family)))
            return false;

        VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenanceFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &maintenanceFeatures
        };
        getPhysicalDeviceFeatures2(physdev, &features);
        return maintenanceFeatures.swapchainMaintenance1 == VK_TRUE;
    }

    [[nodiscard]] bool enableSwapchainMaintenanceFeature(
            VkDeviceCreateInfo& info,
            VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR& storage) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
        while (current) {
            if (current->sType
                    == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR) {
                return reinterpret_cast<const
                    VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR*>(current)
                    ->swapchainMaintenance1 == VK_TRUE;
            }
            current = current->pNext;
        }

        storage.pNext = const_cast<void*>(info.pNext);
        storage.swapchainMaintenance1 = VK_TRUE;
        info.pNext = &storage;
        return true;
    }

    [[nodiscard]] bool timelineSemaphoreUsable(const VkDeviceCreateInfo& info) {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
        while (current) {
            if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                return reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(current)
                    ->timelineSemaphore == VK_TRUE;
            if (current->sType
                    == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)
                return reinterpret_cast<const
                    VkPhysicalDeviceTimelineSemaphoreFeatures*>(current)
                    ->timelineSemaphore == VK_TRUE;
            current = current->pNext;
        }
        return true;
    }

    [[nodiscard]] bool supportsInternallySynchronizedQueues(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
            PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2) {
        if (!getPhysicalDeviceFeatures2
                || !hasDeviceExtension(physdev, funcs,
                    VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME))
            return false;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR syncFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &syncFeatures
        };
        getPhysicalDeviceFeatures2(physdev, &features);
        return syncFeatures.internallySynchronizedQueues == VK_TRUE;
    }

    QueueReservation reserveAuxiliaryTransferQueue(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
            PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2,
            uint32_t applicationApiVersion,
            const VkDeviceCreateInfo& info) {
        QueueReservation result{};
        if (!info.queueCreateInfoCount || !info.pQueueCreateInfos)
            return result;

        uint32_t familyCount{};
        funcs.GetPhysicalDeviceQueueFamilyProperties(physdev, &familyCount, nullptr);
        if (!familyCount)
            return result;

        std::vector<VkQueueFamilyProperties> families(familyCount);
        funcs.GetPhysicalDeviceQueueFamilyProperties(physdev, &familyCount, families.data());

        std::optional<uint32_t> graphicsFamily;
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families.at(i).queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsFamily = i;
                break;
            }
        }
        if (!graphicsFamily.has_value())
            return result;

        result.queueInfos.assign(
            info.pQueueCreateInfos,
            info.pQueueCreateInfos + info.queueCreateInfoCount);

        for (size_t i = 0; i < result.queueInfos.size(); ++i) {
            auto& queueInfo = result.queueInfos.at(i);
            if (queueInfo.queueFamilyIndex != *graphicsFamily)
                continue;
            // The application's logical queue indices are defined by the
            // flags==0 request. Non-zero flag groups use vkGetDeviceQueue2().
            if (queueInfo.flags != 0)
                continue;

            const uint32_t available = families.at(*graphicsFamily).queueCount;
            uint32_t requested{};
            for (const auto& candidate : result.queueInfos) {
                if (candidate.queueFamilyIndex == *graphicsFamily)
                    requested += candidate.queueCount;
            }

            result.familyIndex = *graphicsFamily;
            result.queueInfoIndex = static_cast<uint32_t>(i);

            if (requested < available) {
                result.priorities.reserve(queueInfo.queueCount + 1);
                if (queueInfo.pQueuePriorities) {
                    result.priorities.assign(
                        queueInfo.pQueuePriorities,
                        queueInfo.pQueuePriorities + queueInfo.queueCount);
                } else {
                    result.priorities.assign(queueInfo.queueCount, 1.0F);
                }
                result.priorities.push_back(1.0F);

                result.queueIndex = queueInfo.queueCount;
                result.applicationQueueIndex = result.queueIndex;
                result.mode = QueueReservationMode::DedicatedSpare;
                queueInfo.queueCount++;
                return result;
            }

            // If the graphics family is full, prefer a queue from an entirely
            // unrequested auxiliary family. P3D only needs queue submission for
            // semaphore hand-off, so a compute/transfer-capable family is enough
            // and does not require changing the application's graphics queue.
            for (uint32_t auxiliaryFamily = 0; auxiliaryFamily < familyCount; ++auxiliaryFamily) {
                if (auxiliaryFamily == *graphicsFamily)
                    continue;

                const auto& family = families.at(auxiliaryFamily);
                if (family.queueCount == 0
                        || !(family.queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)))
                    continue;

                const bool alreadyRequested = std::ranges::any_of(
                    result.queueInfos,
                    [auxiliaryFamily](const VkDeviceQueueCreateInfo& candidate) {
                        return candidate.queueFamilyIndex == auxiliaryFamily;
                    });
                if (alreadyRequested)
                    continue;

                result.queueInfos.push_back(VkDeviceQueueCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queueFamilyIndex = auxiliaryFamily,
                    .queueCount = 1,
                    .pQueuePriorities = nullptr
                });
                result.queueInfoIndex =
                    static_cast<uint32_t>(result.queueInfos.size() - 1);
                result.familyIndex = auxiliaryFamily;
                result.queueIndex = 0;
                result.applicationQueueIndex = 0;
                result.internalPriority = 1.0F;
                result.mode = QueueReservationMode::DedicatedAuxiliary;
                return result;
            }

            // If the application already consumes the full graphics family,
            // split its final flags==0 logical queue into a separate internally
            // synchronized queue. The layer remaps the application's original
            // logical index to that same VkQueue, allowing the Fixed worker and
            // application to share it without external queue synchronization.
            if (applicationApiVersion >= VK_API_VERSION_1_1
                    && requested == available
                    && queueInfo.queueCount > 0
                    && supportsInternallySynchronizedQueues(
                        physdev, funcs, getPhysicalDeviceFeatures2)) {
                const auto* feature = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
                while (feature && feature->sType
                        != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR)
                    feature = feature->pNext;
                if (feature && reinterpret_cast<const
                        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR*>(feature)
                        ->internallySynchronizedQueues != VK_TRUE)
                    return QueueReservation{};
                const bool duplicateInternalGroup = std::ranges::any_of(
                    result.queueInfos,
                    [graphicsFamily](const VkDeviceQueueCreateInfo& candidate) {
                        return candidate.queueFamilyIndex == *graphicsFamily
                            && candidate.flags
                                == VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR;
                    });
                if (duplicateInternalGroup)
                    return QueueReservation{};

                const uint32_t logicalIndex = queueInfo.queueCount - 1;
                result.internalPriority = queueInfo.pQueuePriorities
                    ? queueInfo.pQueuePriorities[logicalIndex]
                    : 1.0F;
                result.applicationQueueIndex = logicalIndex;
                result.queueIndex = 0; // index within the internally-synchronized flags group
                result.mode = QueueReservationMode::SharedInternallySynchronized;

                if (queueInfo.queueCount == 1) {
                    queueInfo.flags =
                        VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR;
                    result.internalQueueInfoIndex = static_cast<uint32_t>(i);
                } else {
                    const auto originalQueueInfo = queueInfo;
                    queueInfo.queueCount--;
                    VkDeviceQueueCreateInfo internalQueueInfo{
                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        .pNext = originalQueueInfo.pNext,
                        .flags = VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR,
                        .queueFamilyIndex = *graphicsFamily,
                        .queueCount = 1,
                        .pQueuePriorities = nullptr
                    };
                    result.queueInfos.push_back(internalQueueInfo);
                    result.internalQueueInfoIndex =
                        static_cast<uint32_t>(result.queueInfos.size() - 1);
                }
                return result;
            }

            return QueueReservation{};
        }

        return QueueReservation{};
    }

    [[nodiscard]] bool offloadQueueSupportsSurface(
            const vk::Vulkan& vk,
            const OffloadQueueInfo& offload,
            VkSurfaceKHR surface) {
        if (!vk.fi().GetPhysicalDeviceSurfaceSupportKHR)
            return false;

        VkBool32 supported{VK_FALSE};
        const auto res = vk.fi().GetPhysicalDeviceSurfaceSupportKHR(
            vk.physdev(), offload.familyIndex, surface, &supported);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res,
                "vkGetPhysicalDeviceSurfaceSupportKHR() failed");

        return supported == VK_TRUE;
    }

    std::vector<VkPresentModeKHR> querySurfacePresentModes(
            const vk::Vulkan& vk, VkSurfaceKHR surface) {
        if (!vk.fi().GetPhysicalDeviceSurfacePresentModesKHR)
            return {VK_PRESENT_MODE_FIFO_KHR};

        uint32_t count{};
        auto res = vk.fi().GetPhysicalDeviceSurfacePresentModesKHR(
            vk.physdev(), surface, &count, nullptr);
        if (res != VK_SUCCESS)
            throw ls::vulkan_error(res,
                "vkGetPhysicalDeviceSurfacePresentModesKHR() failed");

        while (count) {
            std::vector<VkPresentModeKHR> modes(count);
            uint32_t written = count;
            res = vk.fi().GetPhysicalDeviceSurfacePresentModesKHR(
                vk.physdev(), surface, &written, modes.data());
            if (res == VK_SUCCESS) {
                modes.resize(written);
                return modes;
            }
            if (res != VK_INCOMPLETE)
                throw ls::vulkan_error(res,
                    "vkGetPhysicalDeviceSurfacePresentModesKHR() failed");

            // The set may have grown between the count and data queries. Ask
            // for a fresh count instead of assuming 'written' is the total.
            res = vk.fi().GetPhysicalDeviceSurfacePresentModesKHR(
                vk.physdev(), surface, &count, nullptr);
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res,
                    "vkGetPhysicalDeviceSurfacePresentModesKHR() failed");
        }
        return {VK_PRESENT_MODE_FIFO_KHR};
    }

    std::vector<VkPresentModeKHR> swapchainCreationAllowedPresentModes(
            const VkSwapchainCreateInfoKHR& info) {
        auto* next = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
        while (next) {
            if (next->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT) {
                const auto* modes =
                    reinterpret_cast<const VkSwapchainPresentModesCreateInfoEXT*>(next);
                if (!modes->presentModeCount || !modes->pPresentModes)
                    return {};
                return std::vector<VkPresentModeKHR>(
                    modes->pPresentModes,
                    modes->pPresentModes + modes->presentModeCount);
            }
            next = next->pNext;
        }
        return {};
    }

    [[nodiscard]] bool hasSwapchainPresentModesCreateInfo(
            const VkSwapchainCreateInfoKHR& info) {
        auto* next = reinterpret_cast<const VkBaseInStructure*>(info.pNext);
        while (next) {
            if (next->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR)
                return true;
            next = next->pNext;
        }
        return false;
    }

    struct SurfacePresentModeCaps {
        uint32_t minImageCount{};
        uint32_t maxImageCount{};
        VkImageUsageFlags supportedUsageFlags{};
        std::vector<VkPresentModeKHR> compatibleModes;
    };

    [[nodiscard]] std::optional<SurfacePresentModeCaps> querySurfacePresentModeCaps(
            const vk::Vulkan& vk,
            PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR getCapabilities2,
            VkSurfaceKHR surface, VkPresentModeKHR mode) {
        if (!getCapabilities2)
            return std::nullopt;

        VkSurfacePresentModeKHR surfaceMode{
            .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR,
            .presentMode = mode
        };
        VkPhysicalDeviceSurfaceInfo2KHR surfaceInfo{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
            .pNext = &surfaceMode,
            .surface = surface
        };
        VkSurfacePresentModeCompatibilityKHR compatibility{
            .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR
        };
        VkSurfaceCapabilities2KHR capabilities{
            .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
            .pNext = &compatibility
        };

        for (int attempt = 0; attempt < 4; ++attempt) {
            compatibility.presentModeCount = 0;
            compatibility.pPresentModes = nullptr;
            auto res = getCapabilities2(vk.physdev(), &surfaceInfo, &capabilities);
            if (res != VK_SUCCESS)
                return std::nullopt;

            std::vector<VkPresentModeKHR> compatible(compatibility.presentModeCount);
            if (compatible.empty()) {
                return SurfacePresentModeCaps {
                    .minImageCount = capabilities.surfaceCapabilities.minImageCount,
                    .maxImageCount = capabilities.surfaceCapabilities.maxImageCount,
                    .supportedUsageFlags = capabilities.surfaceCapabilities.supportedUsageFlags,
                    .compatibleModes = {}
                };
            }

            compatibility.pPresentModes = compatible.data();
            res = getCapabilities2(vk.physdev(), &surfaceInfo, &capabilities);
            if (res == VK_SUCCESS) {
                compatible.resize(compatibility.presentModeCount);
                return SurfacePresentModeCaps {
                    .minImageCount = capabilities.surfaceCapabilities.minImageCount,
                    .maxImageCount = capabilities.surfaceCapabilities.maxImageCount,
                    .supportedUsageFlags = capabilities.surfaceCapabilities.supportedUsageFlags,
                    .compatibleModes = std::move(compatible)
                };
            }
            if (res != VK_INCOMPLETE)
                return std::nullopt;
        }

        return std::nullopt;
    }

    [[nodiscard]] bool containsPresentMode(
            const std::vector<VkPresentModeKHR>& modes, VkPresentModeKHR mode) {
        return std::ranges::find(modes, mode) != modes.end();
    }

    [[nodiscard]] uint32_t commonMaximumImageCount(
            uint32_t first, uint32_t second) {
        if (!first)
            return second;
        if (!second)
            return first;
        return std::min(first, second);
    }

    [[nodiscard]] bool adoptApplicationDualPresentModeDeclaration(
            const vk::Vulkan& vk,
            PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR getCapabilities2,
            SwapchainMaintenanceFamily family,
            VkSwapchainCreateInfoKHR& info,
            uint32_t minimumImageCountFloor,
            std::vector<VkPresentModeKHR>& modeStorage) {
        if (family == SwapchainMaintenanceFamily::None
                || !hasSwapchainPresentModesCreateInfo(info))
            return false;

        const auto declared = swapchainCreationAllowedPresentModes(info);
        const auto fixedMode = selectApplicationDualPresentMode(
            declared, info.presentMode);
        if (!fixedMode.has_value())
            return false;

        const auto adaptiveCaps = querySurfacePresentModeCaps(
            vk, getCapabilities2, info.surface, VK_PRESENT_MODE_FIFO_KHR);
        const auto fixedCaps = querySurfacePresentModeCaps(
            vk, getCapabilities2, info.surface, *fixedMode);
        if (!adaptiveCaps.has_value() || !fixedCaps.has_value())
            return false;

        if (!containsPresentMode(adaptiveCaps->compatibleModes, *fixedMode)
                || !containsPresentMode(
                    fixedCaps->compatibleModes, VK_PRESENT_MODE_FIFO_KHR))
            return false;

        const auto effectiveUsage = effectiveSwapchainImageUsage(info);
        if ((effectiveUsage & ~VkImageUsageFlags2KHR{UINT32_MAX}) != 0)
            return false;
        const auto requiredUsage = static_cast<VkImageUsageFlags>(effectiveUsage);
        if ((adaptiveCaps->supportedUsageFlags & requiredUsage) != requiredUsage
                || (fixedCaps->supportedUsageFlags & requiredUsage) != requiredUsage)
            return false;

        const uint32_t commonMin = std::max(
            std::max(info.minImageCount, minimumImageCountFloor),
            std::max(adaptiveCaps->minImageCount, fixedCaps->minImageCount));
        const uint32_t commonMax = commonMaximumImageCount(
            adaptiveCaps->maxImageCount, fixedCaps->maxImageCount);
        if (commonMax && commonMin > commonMax)
            return false;

        info.minImageCount = commonMin;
        modeStorage = { VK_PRESENT_MODE_FIFO_KHR, *fixedMode };

        std::cerr << "lsfg-vk: adopted application dual present modes: "
            "Adaptive=FIFO Fixed=" << fixedPresentModeName(*fixedMode)
            << " family=" << maintenanceFamilyName(family)
            << " min-images=" << info.minImageCount << "\n";
        return true;
    }

    [[nodiscard]] bool prepareDualPresentModeDeclaration(
            const vk::Vulkan& vk,
            PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR getCapabilities2,
            SwapchainMaintenanceFamily family,
            VkSwapchainCreateInfoKHR& info,
            uint32_t minimumImageCountFloor,
            std::vector<VkPresentModeKHR>& modeStorage,
            VkSwapchainPresentModesCreateInfoKHR& createInfoStorage) {
        if (family == SwapchainMaintenanceFamily::None
                || hasSwapchainPresentModesCreateInfo(info))
            return false;

        const auto supported = querySurfacePresentModes(vk, info.surface);
        const auto initialCaps = querySurfacePresentModeCaps(
            vk, getCapabilities2, info.surface, info.presentMode);
        if (!initialCaps.has_value())
            return false;

        std::vector<VkPresentModeKHR> alternatives;
        if (info.presentMode != VK_PRESENT_MODE_FIFO_KHR)
            alternatives.push_back(VK_PRESENT_MODE_FIFO_KHR);
        if (info.presentMode != VK_PRESENT_MODE_MAILBOX_KHR)
            alternatives.push_back(VK_PRESENT_MODE_MAILBOX_KHR);
        if (info.presentMode != VK_PRESENT_MODE_IMMEDIATE_KHR)
            alternatives.push_back(VK_PRESENT_MODE_IMMEDIATE_KHR);

        const auto effectiveUsage = effectiveSwapchainImageUsage(info);
        if ((effectiveUsage & ~VkImageUsageFlags2KHR{UINT32_MAX}) != 0)
            return false;
        const auto requiredUsage = static_cast<VkImageUsageFlags>(effectiveUsage);
        if ((initialCaps->supportedUsageFlags & requiredUsage) != requiredUsage)
            return false;

        for (const auto alternative : alternatives) {
            if (!containsPresentMode(supported, alternative)
                    || !containsPresentMode(initialCaps->compatibleModes, alternative))
                continue;

            const auto alternativeCaps = querySurfacePresentModeCaps(
                vk, getCapabilities2, info.surface, alternative);
            if (!alternativeCaps.has_value()
                    || !containsPresentMode(alternativeCaps->compatibleModes, info.presentMode)
                    || (alternativeCaps->supportedUsageFlags & requiredUsage) != requiredUsage)
                continue;

            const uint32_t commonMin = std::max(
                std::max(info.minImageCount, minimumImageCountFloor),
                std::max(initialCaps->minImageCount, alternativeCaps->minImageCount));
            const uint32_t commonMax = commonMaximumImageCount(
                initialCaps->maxImageCount, alternativeCaps->maxImageCount);
            if (commonMax && commonMin > commonMax)
                continue;

            info.minImageCount = commonMin;
            modeStorage = { info.presentMode, alternative };
            createInfoStorage = VkSwapchainPresentModesCreateInfoKHR {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
                .pNext = info.pNext,
                .presentModeCount = static_cast<uint32_t>(modeStorage.size()),
                .pPresentModes = modeStorage.data()
            };
            info.pNext = &createInfoStorage;

            std::cerr << "lsfg-vk: dual present modes declared: initial="
                << fixedPresentModeName(modeStorage.at(0))
                << " alternate=" << fixedPresentModeName(modeStorage.at(1))
                << " family=" << maintenanceFamilyName(family)
                << " min-images=" << info.minImageCount << "\n";
            return true;
        }

        return false;
    }

    // instance-wide info initialized at instance creation(s)
    struct InstanceInfo {
        struct QueueMetadata {
            VkDevice device{VK_NULL_HANDLE};
            uint32_t family{VK_QUEUE_FAMILY_IGNORED};
            uint32_t index{};
            VkQueueFlags flags{};
            // Every layer submit to this queue, including the batch bridge and
            // each reserved Adaptive producer, shares this serialization
            // authority.
            std::shared_ptr<std::mutex> mutex{std::make_shared<std::mutex>()};
        };
        std::vector<VkInstance> handles; // there may be several instances
        uint32_t applicationApiVersion{VK_API_VERSION_1_0};
        vk::VulkanInstanceFuncs funcs;

        std::unordered_map<VkDevice, vk::Vulkan> devices;
        std::unordered_map<VkDevice, std::shared_ptr<DeviceRetirementReactor>>
            deviceRetirementReactors;
        std::unordered_map<VkDevice, std::shared_ptr<D3B3DeviceLifetimeQuarantine>>
            deviceQuarantines;
        std::unordered_map<VkDevice,
            std::shared_ptr<PresentedPhysicalImageLeaseRegistry>>
            devicePresentedPhysicalImages;
        std::unordered_map<VkDevice,
            std::shared_ptr<BorrowedPresentFenceRegistry>>
            deviceBorrowedPresentFences;
        std::unordered_map<VkDevice,
            std::shared_ptr<AcquireFenceProxyRegistry>> deviceAcquireFenceProxies;
        std::unordered_map<VkDevice,
            std::shared_ptr<DeferredVirtualRetirementOwner>>
            deviceDeferredVirtualRetirements;
        SwapchainMaintenanceFamily surfaceMaintenanceFamily{
            SwapchainMaintenanceFamily::None
        };
        std::unordered_map<VkDevice, SwapchainMaintenanceFamily>
            swapchainMaintenanceFamilies;
        std::unordered_map<VkDevice, bool> deviceTimelineSemaphoreAvailable;
        std::unordered_map<VkDevice, bool> deviceD2SyncFdAvailable;
        std::unordered_map<VkDevice, bool> deviceD2SinglePhysicalDevice;
        std::unordered_map<VkDevice, OffloadQueueInfo> offloadQueues;
        std::unordered_map<VkQueue, QueueMetadata> queues;
        std::unordered_map<VkSwapchainKHR, ls::R<vk::Vulkan>> swapchains;
        std::unordered_map<VkSwapchainKHR, SwapchainInfo> swapchainInfos;
        std::unordered_map<VkDevice,
            std::vector<std::shared_ptr<D2VulkanShadowRuntime>>>
            deviceD2VulkanRuntimeOwners;
        std::unordered_map<VkSwapchainKHR, std::shared_ptr<VirtualSwapchainRuntime>>
            virtualSwapchains;
    }* instance_info; // NOLINT (global variable)

    // create instance
    VkResult myvkCreateInstance(
            const VkInstanceCreateInfo* info,
            const VkAllocationCallbacks* alloc,
            VkInstance* instance) {
        // apply layer chaining
        auto* layerInfo = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
                || layerInfo->function != VK_LAYER_LINK_INFO)) {
            layerInfo = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer info found in pNext chain, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* linkInfo = layerInfo->u.pLayerInfo;
        if (!linkInfo) {
            std::cerr << "lsfg-vk: link info is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layer_info->GetInstanceProcAddr = linkInfo->pfnNextGetInstanceProcAddr;
        if (!layer_info->GetInstanceProcAddr) {
            std::cerr << "lsfg-vk: next layer's vkGetInstanceProcAddr is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layerInfo->u.pLayerInfo = linkInfo->pNext; // advance for next layer

        // create instance
        auto* vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
            layer_info->GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
        if (!vkCreateInstance) {
            std::cerr << "lsfg-vk: failed to get next layer's vkCreateInstance, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        const uint32_t applicationApiVersion = info->pApplicationInfo
            && info->pApplicationInfo->apiVersion
            ? info->pApplicationInfo->apiVersion
            : VK_API_VERSION_1_0;

        try {
            const auto configSnapshot = layer_info->root.snapshot();
            VkInstanceCreateInfo newInfo = *info;
            std::vector<const char*> compatibilityInstanceExtensions;
            if (configSnapshot.active() && applicationApiVersion < VK_API_VERSION_1_1) {
                if (!hasInstanceExtension(layer_info->GetInstanceProcAddr,
                        GET_PHYSICAL_DEVICE_PROPERTIES_2_KHR)) {
                    std::cerr << "lsfg-vk: P4B Vulkan 1.0 compatibility unavailable: "
                        "VK_KHR_get_physical_device_properties2 not available\n";
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
                }
                compatibilityInstanceExtensions.assign(
                    newInfo.ppEnabledExtensionNames ? newInfo.ppEnabledExtensionNames : nullptr,
                    newInfo.ppEnabledExtensionNames
                        ? newInfo.ppEnabledExtensionNames + newInfo.enabledExtensionCount
                        : nullptr);
                appendExtension(compatibilityInstanceExtensions,
                    GET_PHYSICAL_DEVICE_PROPERTIES_2_KHR);
                if (configSnapshot.activeProfile().gpu.has_value()) {
                    if (!hasInstanceExtension(layer_info->GetInstanceProcAddr,
                            VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME)) {
                        std::cerr << "lsfg-vk: event retirement unavailable: "
                            << VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME
                            << " not available\n";
                        return VK_ERROR_EXTENSION_NOT_PRESENT;
                    }
                    appendExtension(compatibilityInstanceExtensions,
                        VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
                }
                newInfo.enabledExtensionCount =
                    static_cast<uint32_t>(compatibilityInstanceExtensions.size());
                newInfo.ppEnabledExtensionNames = compatibilityInstanceExtensions.data();
            }
            const auto maintenanceFamily = selectSurfaceMaintenanceFamily(
                layer_info->GetInstanceProcAddr, newInfo, configSnapshot.active());
            const bool usesCompatibilityInstanceExtensions =
                applicationApiVersion < VK_API_VERSION_1_1
                && extensionEnabled(newInfo.ppEnabledExtensionNames,
                    newInfo.enabledExtensionCount,
                    GET_PHYSICAL_DEVICE_PROPERTIES_2_KHR);
            std::vector<const char*> maintenanceExtensions;
            if (maintenanceFamily != SwapchainMaintenanceFamily::None) {
                if (newInfo.enabledExtensionCount && newInfo.ppEnabledExtensionNames) {
                    maintenanceExtensions.assign(
                        newInfo.ppEnabledExtensionNames,
                        newInfo.ppEnabledExtensionNames + newInfo.enabledExtensionCount);
                }
                appendExtension(maintenanceExtensions, GET_SURFACE_CAPABILITIES_2_KHR);
                appendExtension(maintenanceExtensions,
                    surfaceMaintenanceExtension(maintenanceFamily));
                newInfo.enabledExtensionCount =
                    static_cast<uint32_t>(maintenanceExtensions.size());
                newInfo.ppEnabledExtensionNames = maintenanceExtensions.data();
            }

            layer_info->root.modifyInstanceCreateInfo(configSnapshot, newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = vkCreateInstance(newInfo, alloc, instance);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateInstance() failed");
                }
            );

            if (!instance_info)
                instance_info = new InstanceInfo{ // NOLINT (memory management)
                        .applicationApiVersion = applicationApiVersion,
                        .funcs = vk::initVulkanInstanceFuncs(*instance,
                        layer_info->GetInstanceProcAddr, true,
                        usesCompatibilityInstanceExtensions),
                };
            else
                instance_info->applicationApiVersion = std::min(
                    instance_info->applicationApiVersion, applicationApiVersion);

            if (instance_info->handles.empty())
                instance_info->surfaceMaintenanceFamily = maintenanceFamily;
            instance_info->handles.push_back(*instance);

            if (maintenanceFamily != SwapchainMaintenanceFamily::None)
                std::cerr << "lsfg-vk: dual-ready surface maintenance enabled: "
                    << maintenanceFamilyName(maintenanceFamily) << "\n";
            else if (configSnapshot.active())
                std::cerr << "lsfg-vk: surface maintenance unavailable; "
                    "dual present-mode declaration will use legacy fallback\n";

            return VK_SUCCESS;
        } catch (const ls::vulkan_error& e) {
            if (e.error() == VK_ERROR_EXTENSION_NOT_PRESENT)
                std::cerr << "lsfg-vk: required Vulkan instance extensions are not present. "
                    "Your GPU driver is not supported.\n";
            return e.error();
        }
    }

    // create device
    VkResult myvkCreateDevice(
            VkPhysicalDevice physdev,
            const VkDeviceCreateInfo* info,
            const VkAllocationCallbacks* alloc,
            VkDevice* device) {
        // apply layer chaining
        auto* layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                || layerInfo->function != VK_LAYER_LINK_INFO)) {
            layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer info found in pNext chain, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* linkInfo = layerInfo->u.pLayerInfo;
        if (!linkInfo) {
            std::cerr << "lsfg-vk: link info is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        instance_info->funcs.GetDeviceProcAddr = linkInfo->pfnNextGetDeviceProcAddr;
        if (!linkInfo->pfnNextGetDeviceProcAddr) {
            std::cerr << "lsfg-vk: next layer's vkGetDeviceProcAddr is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layerInfo->u.pLayerInfo = linkInfo->pNext; // advance for next layer

        // fetch device loader functions
        layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                || layerInfo->function != VK_LOADER_DATA_CALLBACK)) {
            layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer loader data found in pNext chain.\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* setLoaderData = layerInfo->u.pfnSetDeviceLoaderData;
        if (!setLoaderData) {
            std::cerr << "lsfg-vk: instance loader data function is null.\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        // Prepare the offload graphics queue for every active LSFG profile, not
        // only when the process starts in Fixed mode. This keeps the VkDevice
        // capable of entering the asynchronous Fixed path after a future mode
        // switch while leaving worker/swapchain activation gated by fixedMode().
        QueueReservation offloadReservation{};
        const bool applicationTimelineSemaphoreUsable = timelineSemaphoreUsable(*info);
        SwapchainMaintenanceFamily deviceMaintenanceFamily{
            SwapchainMaintenanceFamily::None
        };
        bool d2SyncFdExtensionsEnabled{};
        std::vector<const char*> maintenanceDeviceExtensions;
        VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenanceFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR
        };
        const auto configSnapshot = layer_info->root.snapshot();
        bool d2SinglePhysicalDevice = true;
        for (auto* node = reinterpret_cast<const VkBaseInStructure*>(info->pNext);
                node; node = node->pNext)
            if (node->sType == VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO) {
                const auto* group = reinterpret_cast<
                    const VkDeviceGroupDeviceCreateInfo*>(node);
                d2SinglePhysicalDevice = group->physicalDeviceCount <= 1;
                break;
            }

        // create device
        try {
            VkDeviceCreateInfo newInfo = *info;
            std::vector<const char*> compatibilityDeviceExtensions;
            const bool vulkan10 = instance_info->applicationApiVersion < VK_API_VERSION_1_1;
            const bool needsMemoryRequirements2 = configSnapshot.active() && vulkan10
                && hasDeviceExtension(physdev, instance_info->funcs,
                    GET_MEMORY_REQUIREMENTS_2_KHR);
            const bool needsDedicatedAllocation = needsMemoryRequirements2
                && hasDeviceExtension(physdev, instance_info->funcs,
                    DEDICATED_ALLOCATION_KHR);
            if (configSnapshot.active() && vulkan10) {
                if (!needsMemoryRequirements2) {
                    std::cerr << "lsfg-vk: P4B Vulkan 1.0 compatibility unavailable: "
                        "VK_KHR_get_memory_requirements2 not available\n";
                } else if (!needsDedicatedAllocation) {
                    std::cerr << "lsfg-vk: P4B Vulkan 1.0 compatibility unavailable: "
                        "VK_KHR_dedicated_allocation not available\n";
                } else {
                    compatibilityDeviceExtensions.assign(
                        newInfo.ppEnabledExtensionNames ? newInfo.ppEnabledExtensionNames : nullptr,
                        newInfo.ppEnabledExtensionNames
                            ? newInfo.ppEnabledExtensionNames + newInfo.enabledExtensionCount
                            : nullptr);
                    appendExtension(compatibilityDeviceExtensions, GET_MEMORY_REQUIREMENTS_2_KHR);
                    appendExtension(compatibilityDeviceExtensions, DEDICATED_ALLOCATION_KHR);
                    newInfo.enabledExtensionCount =
                        static_cast<uint32_t>(compatibilityDeviceExtensions.size());
                    newInfo.ppEnabledExtensionNames = compatibilityDeviceExtensions.data();
                }
                if (!needsMemoryRequirements2 || !needsDedicatedAllocation)
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            if (configSnapshot.active()) {
                if (!hasDeviceExtension(physdev, instance_info->funcs,
                        IMAGE_DRM_FORMAT_MODIFIER_EXT)) {
                    std::cerr << "lsfg-vk: runtime image compatibility unavailable: "
                        << IMAGE_DRM_FORMAT_MODIFIER_EXT << " not available\n";
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
                }
                if (compatibilityDeviceExtensions.empty()) {
                    compatibilityDeviceExtensions.assign(
                        newInfo.ppEnabledExtensionNames ? newInfo.ppEnabledExtensionNames : nullptr,
                        newInfo.ppEnabledExtensionNames
                            ? newInfo.ppEnabledExtensionNames + newInfo.enabledExtensionCount
                            : nullptr);
                }
                appendExtension(compatibilityDeviceExtensions, IMAGE_DRM_FORMAT_MODIFIER_EXT);
                newInfo.enabledExtensionCount =
                    static_cast<uint32_t>(compatibilityDeviceExtensions.size());
                newInfo.ppEnabledExtensionNames = compatibilityDeviceExtensions.data();
            }
            if (configSnapshot.active()
                    && configSnapshot.activeProfile().gpu.has_value()) {
                for (const auto* extension : vk::RUNTIME_CROSS_DEVICE_DEVICE_EXTENSIONS) {
                    if (!hasDeviceExtension(physdev, instance_info->funcs, extension)) {
                        std::cerr << "lsfg-vk: cross-device runtime compatibility unavailable: "
                            << extension << " not available\n";
                        return VK_ERROR_EXTENSION_NOT_PRESENT;
                    }
                }
                if (!hasDeviceExtension(physdev, instance_info->funcs,
                        VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME)
                        || (vulkan10 && !hasDeviceExtension(physdev,
                            instance_info->funcs,
                            VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME))) {
                    std::cerr << "lsfg-vk: event retirement unavailable: external fence "
                        "SYNC_FD contract is not advertised\n";
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
                }
                if (compatibilityDeviceExtensions.empty()) {
                    compatibilityDeviceExtensions.assign(
                        newInfo.ppEnabledExtensionNames
                            ? newInfo.ppEnabledExtensionNames : nullptr,
                        newInfo.ppEnabledExtensionNames
                            ? newInfo.ppEnabledExtensionNames
                                + newInfo.enabledExtensionCount : nullptr);
                }
                if (vulkan10)
                    appendExtension(compatibilityDeviceExtensions,
                        VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
                appendExtension(compatibilityDeviceExtensions,
                    VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
                newInfo.enabledExtensionCount = static_cast<uint32_t>(
                    compatibilityDeviceExtensions.size());
                newInfo.ppEnabledExtensionNames =
                    compatibilityDeviceExtensions.data();
            }
            if (configSnapshot.active()) {
                const auto getPhysicalDeviceFeatures2 = reinterpret_cast<
                    PFN_vkGetPhysicalDeviceFeatures2>(layer_info->GetInstanceProcAddr(
                        instance_info->handles.front(), "vkGetPhysicalDeviceFeatures2"));
                offloadReservation = reserveAuxiliaryTransferQueue(
                    physdev, instance_info->funcs, getPhysicalDeviceFeatures2,
                    instance_info->applicationApiVersion, newInfo);
                offloadReservation.apply(newInfo);

                const bool semaphoreFdAvailable = hasDeviceExtension(
                    physdev, instance_info->funcs,
                    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
                const bool fenceFdAvailable = hasDeviceExtension(
                    physdev, instance_info->funcs,
                    VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
                const bool baseSemaphoreAvailable = !vulkan10
                    || hasDeviceExtension(physdev, instance_info->funcs,
                        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
                const bool baseFenceAvailable = !vulkan10
                    || hasDeviceExtension(physdev, instance_info->funcs,
                        VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
                if (semaphoreFdAvailable && fenceFdAvailable
                        && baseSemaphoreAvailable && baseFenceAvailable) {
                    if (compatibilityDeviceExtensions.empty())
                        compatibilityDeviceExtensions.assign(
                            newInfo.ppEnabledExtensionNames,
                            newInfo.ppEnabledExtensionNames
                                + newInfo.enabledExtensionCount);
                    if (vulkan10) {
                        appendExtension(compatibilityDeviceExtensions,
                            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
                        appendExtension(compatibilityDeviceExtensions,
                            VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
                    }
                    appendExtension(compatibilityDeviceExtensions,
                        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
                    appendExtension(compatibilityDeviceExtensions,
                        VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
                    newInfo.enabledExtensionCount = static_cast<uint32_t>(
                        compatibilityDeviceExtensions.size());
                    newInfo.ppEnabledExtensionNames =
                        compatibilityDeviceExtensions.data();
                    d2SyncFdExtensionsEnabled = true;
                }

                const auto requestedFamily = instance_info->surfaceMaintenanceFamily;
                if (supportsSwapchainMaintenance(
                        physdev, instance_info->funcs,
                        getPhysicalDeviceFeatures2, requestedFamily)) {
                    if (enableSwapchainMaintenanceFeature(newInfo, maintenanceFeatures)) {
                        deviceMaintenanceFamily = requestedFamily;
                        if (newInfo.enabledExtensionCount && newInfo.ppEnabledExtensionNames) {
                            maintenanceDeviceExtensions.assign(
                                newInfo.ppEnabledExtensionNames,
                                newInfo.ppEnabledExtensionNames
                                    + newInfo.enabledExtensionCount);
                        }
                        appendExtension(maintenanceDeviceExtensions,
                            swapchainMaintenanceExtension(deviceMaintenanceFamily));
                        newInfo.enabledExtensionCount =
                            static_cast<uint32_t>(maintenanceDeviceExtensions.size());
                        newInfo.ppEnabledExtensionNames = maintenanceDeviceExtensions.data();
                    }
                }
            }

            layer_info->root.modifyDeviceCreateInfo(configSnapshot, newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = instance_info->funcs.CreateDevice(physdev, newInfo, alloc, device);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateDevice() failed");
                }
            );
        } catch (const ls::vulkan_error& e) {
            if (e.error() == VK_ERROR_EXTENSION_NOT_PRESENT)
                std::cerr << "lsfg-vk: required Vulkan device extensions are not present. "
                    "Your GPU driver is not supported.\n";
            return e.error();
        }

        // create layer instance
        try {
            instance_info->devices.emplace(
                *device,
                vk::Vulkan(
                    instance_info->handles.front(), *device, physdev,
                    instance_info->funcs, vk::initVulkanDeviceFuncs(instance_info->funcs, *device,
                        true,
                        instance_info->applicationApiVersion < VK_API_VERSION_1_1
                            && hasDeviceExtension(physdev, instance_info->funcs,
                                GET_MEMORY_REQUIREMENTS_2_KHR)
                            && hasDeviceExtension(physdev, instance_info->funcs,
                                DEDICATED_ALLOCATION_KHR)),
                    true, setLoaderData, std::nullopt,
                    offloadReservation.shared()
                        && offloadReservation.queueInfoIndex
                            == offloadReservation.internalQueueInfoIndex
                        ? VkDeviceQueueCreateFlags{
                            VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR}
                        : VkDeviceQueueCreateFlags{0}
                )
            );
            instance_info->deviceQuarantines.emplace(
                *device, std::make_shared<D3B3DeviceLifetimeQuarantine>());
            instance_info->deviceRetirementReactors.emplace(
                *device, std::make_shared<DeviceRetirementReactor>());
            instance_info->devicePresentedPhysicalImages.emplace(
                *device, std::make_shared<PresentedPhysicalImageLeaseRegistry>());
            instance_info->deviceBorrowedPresentFences.emplace(
                *device, std::make_shared<BorrowedPresentFenceRegistry>(
                    instance_info->deviceQuarantines.at(*device)->identity()));
            instance_info->deviceAcquireFenceProxies.emplace(*device,
                std::make_shared<AcquireFenceProxyRegistry>(
                    instance_info->deviceBorrowedPresentFences.at(*device)));
            instance_info->deviceDeferredVirtualRetirements.emplace(
                *device, std::make_shared<DeferredVirtualRetirementOwner>());
            instance_info->deviceTimelineSemaphoreAvailable.emplace(
                *device, applicationTimelineSemaphoreUsable);
            instance_info->deviceD2SyncFdAvailable.emplace(
                *device, d2SyncFdExtensionsEnabled);
            instance_info->deviceD2SinglePhysicalDevice.emplace(
                *device, d2SinglePhysicalDevice);
            instance_info->deviceD2VulkanRuntimeOwners.emplace(*device,
                std::vector<std::shared_ptr<D2VulkanShadowRuntime>>{});
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk initialization:\n";
            std::cerr << "- " << e.what() << '\n';
        }

        if (deviceMaintenanceFamily != SwapchainMaintenanceFamily::None
                && instance_info->devices.contains(*device)) {
            instance_info->swapchainMaintenanceFamilies.emplace(
                *device, deviceMaintenanceFamily);
            std::cerr << "lsfg-vk: dual-ready swapchain maintenance enabled: "
                << maintenanceFamilyName(deviceMaintenanceFamily) << "\n";
        } else if (configSnapshot.active()) {
            std::cerr << "lsfg-vk: swapchain maintenance feature unavailable; "
                "dual present-mode declaration will use legacy fallback\n";
        }

        if (offloadReservation.valid()) {
            const auto it = instance_info->devices.find(*device);
            if (it != instance_info->devices.end()) {
                VkQueue queue{VK_NULL_HANDLE};
                if (offloadReservation.shared()) {
                    const VkDeviceQueueInfo2 queueInfo{
                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
                        .flags = VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR,
                        .queueFamilyIndex = offloadReservation.familyIndex,
                        .queueIndex = offloadReservation.queueIndex
                    };
                    const auto getDeviceQueue2 = reinterpret_cast<PFN_vkGetDeviceQueue2>(
                        instance_info->funcs.GetDeviceProcAddr(*device, "vkGetDeviceQueue2"));
                    if (getDeviceQueue2)
                        getDeviceQueue2(*device, &queueInfo, &queue);
                } else {
                    it->second.df().GetDeviceQueue(
                        *device,
                        offloadReservation.familyIndex,
                        offloadReservation.queueIndex,
                        &queue);
                }

                if (queue != VK_NULL_HANDLE) {
                    uint32_t familyCount{};
                    it->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        it->second.physdev(), &familyCount, nullptr);
                    std::vector<VkQueueFamilyProperties> families(familyCount);
                    it->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        it->second.physdev(), &familyCount, families.data());
                    instance_info->queues[queue] = {
                        *device, offloadReservation.familyIndex,
                        offloadReservation.queueIndex,
                        families.at(offloadReservation.familyIndex).queueFlags};
                    auto res = setLoaderData(*device, queue);
                    if (res != VK_SUCCESS) {
                        std::cerr << "lsfg-vk: failed to attach loader data to fixed-pacing queue\n";
                    } else {
                        instance_info->offloadQueues.emplace(*device, OffloadQueueInfo {
                            .queue = queue,
                            .familyIndex = offloadReservation.familyIndex,
                            .queueIndex = offloadReservation.queueIndex,
                            .applicationQueueIndex = offloadReservation.applicationQueueIndex,
                            .sharedWithApplication = offloadReservation.shared()
                        });

                        std::cerr << "lsfg-vk: dual-ready offload queue prepared: "
                            << (offloadReservation.shared()
                                ? "shared-internally-synchronized"
                                : offloadReservation.mode == QueueReservationMode::DedicatedAuxiliary
                                    ? "dedicated-auxiliary"
                                    : "dedicated")
                            << " family=" << offloadReservation.familyIndex
                            << " queue=" << offloadReservation.queueIndex
                            << " app-index=" << offloadReservation.applicationQueueIndex
                            << " initial-mode="
                            << (configSnapshot.fixedMode() ? "Fixed" : "Adaptive")
                            << "\n";

                        if (offloadReservation.shared() && configSnapshot.fixedMode())
                            std::cerr << "lsfg-vk: Fixed worker sharing internally synchronized graphics queue "
                                << offloadReservation.familyIndex << ':'
                                << offloadReservation.applicationQueueIndex << "\n";
                    }
                }
            }
        } else if (configSnapshot.fixedMode()) {
            std::cerr << "lsfg-vk: no dedicated or internally synchronized graphics queue "
                "is available for asynchronous fixed pacing; the current synchronous Fixed "
                "path will remain in use\n";
        } else if (configSnapshot.active()) {
            std::cerr << "lsfg-vk: no dual-ready offload graphics queue is available; "
                "Adaptive remains unchanged and a future hot switch to asynchronous Fixed "
                "will require the legacy fallback path\n";
        }

        return VK_SUCCESS;
    }

    // destroy device
    void myvkDestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
        // A well-behaved application destroys swapchains first, but stop any
        // remaining workers defensively before their VkDevice/queue disappears.
        std::vector<VkSwapchainKHR> deviceSwapchains;
        for (auto& [swapchain, runtime] : instance_info->virtualSwapchains) {
            const auto swapchainIt = instance_info->swapchains.find(swapchain);
            if (swapchainIt != instance_info->swapchains.end()
                    && swapchainIt->second.get().dev() == device) {
                if (!runtime->stopAndDetach(
                        instance_info->deviceDeferredVirtualRetirements.at(device)))
                    std::terminate();
                deviceSwapchains.push_back(swapchain);
            }
        }

        // Keep the reactor alive while internal present SYNC_FDs are drained.
        // Erasing it below stops registrations only after exact WSI authority
        // has been resolved.
        const auto reactorIt = instance_info->deviceRetirementReactors.find(device);

        // Device destruction is a terminal synchronization boundary for
        // device-owned work. It does not fabricate presentation completion:
        // every pending authority was already spliced into the deferred owner.
        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt != instance_info->devices.end()) {
            const auto idle = deviceIt->second.df().DeviceWaitIdle(device);
            if (idle == VK_SUCCESS || idle == VK_ERROR_DEVICE_LOST) {
                instance_info->deviceDeferredVirtualRetirements.at(device)
                    ->notifyDeviceRetirement();
            } else {
                std::cerr << "lsfg-vk: final vkDestroyDevice idle boundary failed: "
                    << idle << '\n';
            }

            // Device destruction is the application's terminal fence-use
            // boundary. Observe/drain borrowed present fences explicitly;
            // DeviceWaitIdle above is not presentation-engine authority.
            const auto borrowed =
                instance_info->deviceBorrowedPresentFences.find(device);
            if (borrowed != instance_info->deviceBorrowedPresentFences.end()) {
                static_cast<void>(borrowed->second->drainForTeardown(
                    [&](VkFence fence) {
                        return deviceIt->second.df().GetFenceStatus(device, fence);
                    }, [&](VkFence fence) {
                        return deviceIt->second.df().WaitForFences(
                            device, 1, &fence, VK_TRUE, UINT64_MAX);
                    }));
            }
            const auto leases =
                instance_info->devicePresentedPhysicalImages.find(device);
            if (leases != instance_info->devicePresentedPhysicalImages.end()) {
                // Internal present fences, rather than DeviceWaitIdle, retire
                // committed WSI leases. Only uncommitted producer-only
                // reservations are released by the successful idle boundary.
                static_cast<void>(leases->second->drainInternalFencesForTeardown());
                if (idle == VK_SUCCESS || idle == VK_ERROR_DEVICE_LOST)
                    static_cast<void>(leases->second
                        ->releasePreparedProducerBackingsAfterDeviceIdle());
                if (idle == VK_SUCCESS && leases->second->size() != 0) {
                    std::cerr << "lsfg-vk: refusing to destroy a healthy device "
                        "with unresolved WSI present leases\n";
                    std::terminate();
                }
            }
        }
        for (const auto swapchain : deviceSwapchains) {
            instance_info->virtualSwapchains.erase(swapchain);
            instance_info->swapchainInfos.erase(swapchain);
            instance_info->swapchains.erase(swapchain);
            layer_info->root.removeSwapchainContext(swapchain);
        }

        instance_info->offloadQueues.erase(device);
        instance_info->swapchainMaintenanceFamilies.erase(device);

        // Device teardown is the explicit boundary where a final device-wide
        // synchronization is permitted. Stop new reactor registrations first,
        // then make healthy accepted work safe before releasing child objects.
        if (reactorIt != instance_info->deviceRetirementReactors.end()) {
            instance_info->deviceRetirementReactors.erase(reactorIt);
        }

        // Release accepted device-lost backings at the actual device lifetime
        // boundary, while vkDestroySemaphore can still target a live device.
        instance_info->deviceQuarantines.erase(device);
        instance_info->devicePresentedPhysicalImages.erase(device);
        instance_info->deviceAcquireFenceProxies.erase(device);
        instance_info->deviceBorrowedPresentFences.erase(device);
        instance_info->deviceTimelineSemaphoreAvailable.erase(device);
        instance_info->deviceD2SyncFdAvailable.erase(device);
        instance_info->deviceD2SinglePhysicalDevice.erase(device);
        instance_info->deviceD2VulkanRuntimeOwners.erase(device);
        if (const auto deferred = instance_info->deviceDeferredVirtualRetirements.find(device);
                deferred != instance_info->deviceDeferredVirtualRetirements.end()) {
            deferred->second->notifyDeviceRetirement();
            instance_info->deviceDeferredVirtualRetirements.erase(deferred);
        }

        // destroy layer instance
        auto it = instance_info->devices.find(device);
        if (it != instance_info->devices.end())
            instance_info->devices.erase(it);

        // destroy device
        auto vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
            instance_info->funcs.GetDeviceProcAddr(device, "vkDestroyDevice"));
        if (!vkDestroyDevice) {
            std::cerr << "lsfg-vk: failed to get next layer's vkDestroyDevice, "
                "the previous layer does not follow spec\n";
            return;
        }

        vkDestroyDevice(device, alloc);
    }

    // destroy instance
    void myvkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc) {
        // remove instance handle
        auto it = std::ranges::find(instance_info->handles, instance);
        if (it != instance_info->handles.end())
            instance_info->handles.erase(it);

        // destroy instance info if no handles remain
        if (instance_info->handles.empty()) {
            delete instance_info; // NOLINT (memory management)
            instance_info = nullptr;
        }

        // destroy instance
        auto vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
            layer_info->GetInstanceProcAddr(instance, "vkDestroyInstance"));
        if (!vkDestroyInstance) {
            std::cerr << "lsfg-vk: failed to get next layer's vkDestroyInstance, "
                "the previous layer does not follow spec\n";
            return;
        }

        vkDestroyInstance(instance, alloc);
    }

    // Preserve the application's original flags==0 logical queue view when
    // Fixed had to split the final queue into an internally synchronized group.
    void myvkGetDeviceQueue(
            VkDevice device,
            uint32_t queueFamilyIndex,
            uint32_t queueIndex,
            VkQueue* queue) {
        if (!queue)
            return;

        if (instance_info) {
            const auto offload = instance_info->offloadQueues.find(device);
            if (offload != instance_info->offloadQueues.end()
                    && offload->second.sharedWithApplication
                    && offload->second.familyIndex == queueFamilyIndex
                    && offload->second.applicationQueueIndex == queueIndex) {
                *queue = offload->second.queue;
                const auto deviceIt = instance_info->devices.find(device);
                if (deviceIt != instance_info->devices.end()) {
                    uint32_t count{};
                    deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        deviceIt->second.physdev(), &count, nullptr);
                    std::vector<VkQueueFamilyProperties> families(count);
                    deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        deviceIt->second.physdev(), &count, families.data());
                    instance_info->queues[*queue] = {device, queueFamilyIndex, queueIndex,
                        families.at(queueFamilyIndex).queueFlags};
                }
                return;
            }

            const auto it = instance_info->devices.find(device);
            if (it != instance_info->devices.end()) {
                it->second.df().GetDeviceQueue(
                    device, queueFamilyIndex, queueIndex, queue);
                if (*queue != VK_NULL_HANDLE) {
                    uint32_t count{};
                    it->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        it->second.physdev(), &count, nullptr);
                    std::vector<VkQueueFamilyProperties> families(count);
                    it->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        it->second.physdev(), &count, families.data());
                    instance_info->queues[*queue] = {device, queueFamilyIndex, queueIndex,
                        families.at(queueFamilyIndex).queueFlags};
                }
                return;
            }

            if (instance_info->funcs.GetDeviceProcAddr) {
                const auto next = reinterpret_cast<PFN_vkGetDeviceQueue>(
                    instance_info->funcs.GetDeviceProcAddr(device, "vkGetDeviceQueue"));
                if (next) {
                    next(device, queueFamilyIndex, queueIndex, queue);
                    if (*queue != VK_NULL_HANDLE) {
                        const auto deviceIt = instance_info->devices.find(device);
                        if (deviceIt != instance_info->devices.end()) {
                            uint32_t count{};
                            deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                                deviceIt->second.physdev(), &count, nullptr);
                            std::vector<VkQueueFamilyProperties> families(count);
                            deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                                deviceIt->second.physdev(), &count, families.data());
                            instance_info->queues[*queue] = {device, queueFamilyIndex, queueIndex,
                                families.at(queueFamilyIndex).queueFlags};
                        }
                    }
                    return;
                }
            }
        }

        *queue = VK_NULL_HANDLE;
    }

    void myvkGetDeviceQueue2(
            VkDevice device,
            const VkDeviceQueueInfo2* queueInfo,
            VkQueue* queue) {
        if (!queue)
            return;

        if (instance_info && queueInfo) {
            const auto offload = instance_info->offloadQueues.find(device);
            if (offload != instance_info->offloadQueues.end()
                    && offload->second.sharedWithApplication
                    && queueInfo->flags == 0
                    && offload->second.familyIndex == queueInfo->queueFamilyIndex
                    && offload->second.applicationQueueIndex == queueInfo->queueIndex) {
                *queue = offload->second.queue;
                const auto deviceIt = instance_info->devices.find(device);
                if (deviceIt != instance_info->devices.end()) {
                    uint32_t count{};
                    deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        deviceIt->second.physdev(), &count, nullptr);
                    std::vector<VkQueueFamilyProperties> families(count);
                    deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                        deviceIt->second.physdev(), &count, families.data());
                    instance_info->queues[*queue] = {device, queueInfo->queueFamilyIndex,
                        queueInfo->queueIndex,
                        families.at(queueInfo->queueFamilyIndex).queueFlags};
                }
                return;
            }

            if (instance_info->funcs.GetDeviceProcAddr) {
                const auto next = reinterpret_cast<PFN_vkGetDeviceQueue2>(
                    instance_info->funcs.GetDeviceProcAddr(device, "vkGetDeviceQueue2"));
                if (next) {
                    next(device, queueInfo, queue);
                    if (*queue != VK_NULL_HANDLE) {
                        const auto deviceIt = instance_info->devices.find(device);
                        if (deviceIt != instance_info->devices.end()) {
                            uint32_t count{};
                            deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                                deviceIt->second.physdev(), &count, nullptr);
                            std::vector<VkQueueFamilyProperties> families(count);
                            deviceIt->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                                deviceIt->second.physdev(), &count, families.data());
                            instance_info->queues[*queue] = {device,
                                queueInfo->queueFamilyIndex, queueInfo->queueIndex,
                                families.at(queueInfo->queueFamilyIndex).queueFlags};
                        }
                    }
                    return;
                }
            }
        }

        *queue = VK_NULL_HANDLE;
    }

    // get optional function pointer override
    PFN_vkVoidFunction getProcAddr(const std::string& name) {
        auto it = layer_info->map.find(name);
        if (it != layer_info->map.end())
            return it->second;
        return nullptr;
    }

    // get instance-level function pointers
    PFN_vkVoidFunction myvkGetInstanceProcAddr(VkInstance instance, const char* name) {
        if (!name) return nullptr;

        auto func = getProcAddr(name);
        if (func) return func;

        if (!layer_info->GetInstanceProcAddr) return nullptr;
        return layer_info->GetInstanceProcAddr(instance, name);
    }

    // get device-level function pointers
    PFN_vkVoidFunction myvkGetDeviceProcAddr(VkDevice device, const char* name) {
        if (!name) return nullptr;

        auto func = getProcAddr(name);
        if (func) return func;

        if (!instance_info->funcs.GetDeviceProcAddr) return nullptr;
        return instance_info->funcs.GetDeviceProcAddr(device, name);
    }
}

namespace {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    struct DeviceEntrypointTestContext {
        VkDevice device{VK_NULL_HANDLE};
        vk::VulkanDeviceFuncs downstream{};
        std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases;
        std::shared_ptr<BorrowedPresentFenceRegistry> borrowed;
        std::shared_ptr<AcquireFenceProxyRegistry> proxies;
    };
    std::unique_ptr<DeviceEntrypointTestContext> deviceEntrypointTestContext;
    bool queuePresentHarnessInstalled{};
    PresentedPhysicalImageIdentity queuePresentHarnessLastIdentity{};
    std::shared_ptr<DeferredVirtualRetirementOwner>
        queuePresentHarnessDeferredOwner;
    std::shared_ptr<std::atomic_bool> queuePresentHarnessSeededDone;
    std::weak_ptr<void> queuePresentHarnessSeededBacking;
    uint32_t queuePresentHarnessSeededDestroys{};
    std::atomic<VkResult> queuePresentHarnessWorkerInternalFailure{VK_SUCCESS};
    std::atomic_bool queuePresentHarnessFailD2Bookkeeping{};
    std::atomic_bool queuePresentHarnessFailD2BatchReservation{};
    std::weak_ptr<D2RealWsiState> queuePresentHarnessLastD2State;
#endif

    VkResult myvkCreateFence(VkDevice device, const VkFenceCreateInfo* info,
            const VkAllocationCallbacks* alloc, VkFence* fence) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            const auto result = deviceEntrypointTestContext->downstream.CreateFence(
                device, info, alloc, fence);
            if (result == VK_SUCCESS && fence && *fence != VK_NULL_HANDLE)
                static_cast<void>(deviceEntrypointTestContext->borrowed->created(*fence));
            return result;
        }
#endif
        const auto it = instance_info->devices.find(device);
        if (it == instance_info->devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        const auto result = it->second.df().CreateFence(device, info, alloc, fence);
        if (result == VK_SUCCESS && fence && *fence != VK_NULL_HANDLE) {
            const auto registry = instance_info->deviceBorrowedPresentFences.find(device);
            if (registry == instance_info->deviceBorrowedPresentFences.end()
                    || !registry->second->created(*fence).valid()) {
                // Observation metadata is optional. Never replace a successful
                // application create, substitute its handle, or add a hidden
                // destroy. A later present using an untracked fence retains its
                // WSI lease conservatively instead.
                std::cerr << "lsfg-vk: application fence lifecycle could not be "
                    "tracked; preserving downstream create result\n";
            }
        }
        return result;
    }

    void myvkDestroyFence(VkDevice device, VkFence fence,
            const VkAllocationCallbacks* alloc) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            deviceEntrypointTestContext->proxies->destroy(fence);
            observeBorrowedDestroyFence(*deviceEntrypointTestContext->borrowed,
                fence, [&] { deviceEntrypointTestContext->downstream.DestroyFence(
                    device, fence, alloc); });
            return;
        }
#endif
        const auto registry = instance_info->deviceBorrowedPresentFences.find(device);
        if (const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
                proxies != instance_info->deviceAcquireFenceProxies.end())
            proxies->second->destroy(fence);
        const auto vulkan = instance_info->devices.find(device);
        bool forwarded{};
        if (registry != instance_info->deviceBorrowedPresentFences.end()
                && vulkan != instance_info->devices.end()) {
            observeBorrowedDestroyFence(*registry->second, fence, [&] {
                vulkan->second.df().DestroyFence(device, fence, alloc);
            });
            forwarded = true;
        }
        if (const auto deferred = instance_info->deviceDeferredVirtualRetirements.find(device);
                deferred != instance_info->deviceDeferredVirtualRetirements.end())
            deferred->second->notifyDeviceRetirement();
        if (!forwarded && vulkan != instance_info->devices.end())
            vulkan->second.df().DestroyFence(device, fence, alloc);
    }

    VkResult myvkGetFenceStatus(VkDevice device, VkFence fence) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            if (!deviceEntrypointTestContext->proxies
                    ->applicationOperationAllowed(fence))
                return VK_ERROR_DEVICE_LOST;
            const auto projected = deviceEntrypointTestContext->proxies->project(fence);
            if (projected != fence)
                return deviceEntrypointTestContext->downstream.GetFenceStatus(
                    device, projected);
            return observeBorrowedGetFenceStatus(
                *deviceEntrypointTestContext->borrowed, fence, [&] {
                    return deviceEntrypointTestContext->downstream.GetFenceStatus(
                        device, deviceEntrypointTestContext->proxies->project(fence)); });
        }
#endif
        const auto it = instance_info->devices.find(device);
        if (it == instance_info->devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        const auto registry = instance_info->deviceBorrowedPresentFences.find(device);
        const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
        if (proxies != instance_info->deviceAcquireFenceProxies.end()
                && !proxies->second->applicationOperationAllowed(fence))
            return VK_ERROR_DEVICE_LOST;
        const auto projected = proxies == instance_info->deviceAcquireFenceProxies.end()
            ? fence : proxies->second->project(fence);
        if (projected != fence)
            return it->second.df().GetFenceStatus(device, projected);
        const auto result = registry == instance_info->deviceBorrowedPresentFences.end()
            ? it->second.df().GetFenceStatus(device, projected)
            : observeBorrowedGetFenceStatus(*registry->second, fence, [&] {
                return it->second.df().GetFenceStatus(device, projected);
            });
        if (result == VK_SUCCESS) {
            if (const auto deferred = instance_info->deviceDeferredVirtualRetirements.find(device);
                    deferred != instance_info->deviceDeferredVirtualRetirements.end())
                deferred->second->notifyDeviceRetirement();
        }
        return result;
    }

    VkResult myvkWaitForFences(VkDevice device, uint32_t fenceCount,
            const VkFence* fences, VkBool32 waitAll, uint64_t timeout) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            for (uint32_t i = 0; fences && i < fenceCount; ++i)
                if (!deviceEntrypointTestContext->proxies
                        ->applicationOperationAllowed(fences[i]))
                    return VK_ERROR_DEVICE_LOST;
            std::vector<VkFence> projected;
            try {
                projected = deviceEntrypointTestContext->proxies->project(
                    fenceCount, fences);
            } catch (const std::bad_alloc&) {
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            bool hasProjection{};
            for (uint32_t i = 0; i < fenceCount; ++i)
                hasProjection = hasProjection || projected[i] != fences[i];
            return observeBorrowedWaitForFences(
                *deviceEntrypointTestContext->borrowed,
                fenceCount, fences, hasProjection ? VK_FALSE : waitAll, [&] {
                    return deviceEntrypointTestContext->downstream.WaitForFences(
                        device, fenceCount, projected.data(), waitAll, timeout); },
                [&](VkFence fence) {
                    for (uint32_t i = 0; i < fenceCount; ++i)
                        if (fences[i] == fence && projected[i] != fence)
                            return VK_NOT_READY;
                    return deviceEntrypointTestContext->downstream.GetFenceStatus(
                        device, fence);
                });
        }
#endif
        const auto it = instance_info->devices.find(device);
        if (it == instance_info->devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        const auto registry = instance_info->deviceBorrowedPresentFences.find(device);
        std::vector<VkFence> projected;
        if (const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
                proxies != instance_info->deviceAcquireFenceProxies.end() && fences) {
            for (uint32_t i = 0; i < fenceCount; ++i)
                if (!proxies->second->applicationOperationAllowed(fences[i]))
                    return VK_ERROR_DEVICE_LOST;
            try {
                projected = proxies->second->project(fenceCount, fences);
            } catch (const std::bad_alloc&) {
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
        }
        const auto* downstreamFences = projected.empty() ? fences : projected.data();
        bool hasProjection{};
        for (uint32_t i = 0; i < projected.size(); ++i)
            hasProjection = hasProjection || projected[i] != fences[i];
        const auto downstream = [&] { return it->second.df().WaitForFences(
            device, fenceCount, downstreamFences, waitAll, timeout); };
        const auto result = registry == instance_info->deviceBorrowedPresentFences.end()
            ? downstream() : observeBorrowedWaitForFences(*registry->second,
                fenceCount, fences, hasProjection ? VK_FALSE : waitAll,
                downstream, [&](VkFence fence) {
                    for (uint32_t i = 0; i < projected.size(); ++i)
                        if (fences[i] == fence && projected[i] != fence)
                            return VK_NOT_READY;
                    return it->second.df().GetFenceStatus(device, fence);
                });
        if (result == VK_SUCCESS && fences && (waitAll || fenceCount == 1)) {
            if (const auto deferred = instance_info->deviceDeferredVirtualRetirements.find(device);
                    deferred != instance_info->deviceDeferredVirtualRetirements.end())
                deferred->second->notifyDeviceRetirement();
        }
        return result;
    }

    VkResult myvkResetFences(VkDevice device, uint32_t fenceCount,
            const VkFence* fences) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            for (uint32_t i = 0; fences && i < fenceCount; ++i)
                if (!deviceEntrypointTestContext->proxies
                        ->applicationOperationAllowed(fences[i]))
                    return VK_ERROR_DEVICE_LOST;
            const auto result = observeBorrowedResetFences(
                *deviceEntrypointTestContext->borrowed,
                fenceCount, fences, [&] {
                    return deviceEntrypointTestContext->downstream.ResetFences(
                        device, fenceCount, fences); }, [&](VkFence fence) {
                    return deviceEntrypointTestContext->downstream.GetFenceStatus(
                        device, fence);
                });
            if (result == VK_SUCCESS)
                deviceEntrypointTestContext->proxies->resetSuccess(
                    fenceCount, fences);
            return result;
        }
#endif
        const auto it = instance_info->devices.find(device);
        if (it == instance_info->devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        if (const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
                proxies != instance_info->deviceAcquireFenceProxies.end())
            for (uint32_t i = 0; fences && i < fenceCount; ++i)
                if (!proxies->second->applicationOperationAllowed(fences[i]))
                    return VK_ERROR_DEVICE_LOST;
        const auto registry = instance_info->deviceBorrowedPresentFences.find(device);
        const auto downstream = [&] {
            return it->second.df().ResetFences(device, fenceCount, fences); };
        const auto result = registry == instance_info->deviceBorrowedPresentFences.end()
            ? downstream() : observeBorrowedResetFences(*registry->second,
                fenceCount, fences, downstream, [&](VkFence fence) {
                    return it->second.df().GetFenceStatus(device, fence);
                });
        if (result == VK_SUCCESS)
            if (const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
                    proxies != instance_info->deviceAcquireFenceProxies.end())
                proxies->second->resetSuccess(fenceCount, fences);
        if (fences) {
            if (const auto deferred = instance_info->deviceDeferredVirtualRetirements.find(device);
                    deferred != instance_info->deviceDeferredVirtualRetirements.end())
                deferred->second->notifyDeviceRetirement();
        }
        return result;
    }

    VkResult myvkImportFenceFdKHR(VkDevice device,
            const VkImportFenceFdInfoKHR* info) {
        if (!info) return VK_ERROR_INITIALIZATION_FAILED;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        if (deviceEntrypointTestContext
                && deviceEntrypointTestContext->device == device) {
            if (!deviceEntrypointTestContext->proxies
                    ->applicationOperationAllowed(info->fence))
                return VK_ERROR_DEVICE_LOST;
            if (!deviceEntrypointTestContext->downstream.ImportFenceFdKHR)
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            const auto result = deviceEntrypointTestContext->downstream.ImportFenceFdKHR(
                device, info);
            if (result == VK_SUCCESS) {
                static_cast<void>(deviceEntrypointTestContext->borrowed
                    ->importBoundary(info->fence));
                deviceEntrypointTestContext->proxies->importSuccess(info->fence);
            }
            return result;
        }
#endif
        const auto it = instance_info->devices.find(device);
        if (it == instance_info->devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        if (!it->second.df().ImportFenceFdKHR) return VK_ERROR_EXTENSION_NOT_PRESENT;
        const auto proxies = instance_info->deviceAcquireFenceProxies.find(device);
        if (proxies != instance_info->deviceAcquireFenceProxies.end()
                && !proxies->second->applicationOperationAllowed(info->fence))
            return VK_ERROR_DEVICE_LOST;
        const auto result = it->second.df().ImportFenceFdKHR(device, info);
        if (result == VK_SUCCESS) {
            if (const auto borrowed = instance_info->deviceBorrowedPresentFences.find(device);
                    borrowed != instance_info->deviceBorrowedPresentFences.end())
                static_cast<void>(borrowed->second->importBoundary(info->fence));
            if (proxies != instance_info->deviceAcquireFenceProxies.end())
                proxies->second->importSuccess(info->fence);
        }
        return result;
    }

    VkResult myvkCreateSwapchainKHR(
            VkDevice device,
            const VkSwapchainCreateInfoKHR* info,
            const VkAllocationCallbacks* alloc,
            VkSwapchainKHR* swapchain) {
        const auto& it = instance_info->devices.find(device);
        if (it == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;

        bool d2DownstreamSwapchainCreated{};
        const auto cleanupFailedD2Create = [&]() noexcept {
            if (!d2DownstreamSwapchainCreated || !swapchain
                    || *swapchain == VK_NULL_HANDLE)
                return;
            const auto failed = *swapchain;
            try { layer_info->root.removeSwapchainContext(failed); } catch (...) {}
            instance_info->virtualSwapchains.erase(failed);
            instance_info->swapchainInfos.erase(failed);
            instance_info->swapchains.erase(failed);
            it->second.df().DestroySwapchainKHR(device, failed, alloc);
            *swapchain = VK_NULL_HANDLE;
            d2DownstreamSwapchainCreated = false;
        };

        try {
            // vkCreateSwapchainKHR retires oldSwapchain immediately. Retired
            // swapchains cannot acquire new real WSI images, so stop/join our
            // asynchronous worker before the driver sees oldSwapchain. Keep
            // the runtime and application-visible virtual VkImages alive until
            // vkDestroySwapchainKHR, but drain hidden queue work first.
            if (info && info->oldSwapchain != VK_NULL_HANDLE) {
                const auto oldInfo =
                    instance_info->swapchainInfos.find(info->oldSwapchain);
                if (oldInfo != instance_info->swapchainInfos.end()
                        && oldInfo->second.d2Foundation
                        && oldInfo->second.d2State)
                    oldInfo->second.d2State->retireForLsfgWork();
                const auto oldRuntime =
                    instance_info->virtualSwapchains.find(info->oldSwapchain);
                if (oldRuntime != instance_info->virtualSwapchains.end()) {
                    if (!oldRuntime->second->stopAndDetach(
                            instance_info->deviceDeferredVirtualRetirements.at(device)))
                        throw ls::vulkan_error(VK_ERROR_OUT_OF_HOST_MEMORY,
                            "failed to detach old virtual swapchain retirement");
                    std::cerr << "lsfg-vk: retired old virtual presentation worker before swapchain recreation\n";
                }
            }

            // Capture one immutable configuration revision for the complete
            // create transaction. No later mode/profile queries may observe a
            // different hot-reload revision while this swapchain is built.
            const auto updatedSnapshot = layer_info->root.update();
            const auto configSnapshot = updatedSnapshot.has_value()
                ? *updatedSnapshot
                : layer_info->root.snapshot();

            constexpr bool d2Foundation = true;
            VkSwapchainCreateInfoKHR newInfo = *info;
            const VkPresentModeKHR applicationPresentMode = newInfo.presentMode;
            const uint32_t applicationMinImageCount = newInfo.minImageCount;
            bool fixedAsyncPresentModeEligible{};
            bool dualPresentModeDeclared{};
            std::vector<VkPresentModeKHR> dualPresentModes;
            VkSwapchainPresentModesCreateInfoKHR dualPresentModesInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR
            };
            if (d2Foundation) {
                const auto createResult = it->second.df().CreateSwapchainKHR(
                    device, info, alloc, swapchain);
                if (createResult != VK_SUCCESS)
                    throw ls::vulkan_error(
                        createResult, "vkCreateSwapchainKHR() failed");
                d2DownstreamSwapchainCreated = true;
            } else layer_info->root.modifySwapchainCreateInfo(
                configSnapshot, it->second, newInfo,
                [&, newInfo = &newInfo]() {
                    // Only the asynchronous Fixed path changes WSI mode. The
                    // Adaptive and synchronous 3B paths retain their existing
                    // FIFO behavior. Eligibility is checked before creation so
                    // an unavailable dedicated presentation queue cannot alter
                    // legacy swapchain semantics.
                    if (configSnapshot.fixedMode()) {
                        const auto queueIt = instance_info->offloadQueues.find(device);
                        const auto spec = makeVirtualSwapchainImageSpec(*newInfo);
                        if (queueIt != instance_info->offloadQueues.end()
                                && spec.supported()
                                && offloadQueueSupportsSurface(
                                    it->second, queueIt->second, newInfo->surface)) {
                            const auto supported = querySurfacePresentModes(
                                it->second, newInfo->surface);
                            const auto allowed = swapchainCreationAllowedPresentModes(*newInfo);
                            const auto selected = selectFixedPresentMode(
                                supported, allowed, applicationPresentMode);
                            newInfo->presentMode = selected;
                            fixedAsyncPresentModeEligible = true;

                            std::cerr << "lsfg-vk: Fixed WSI present mode: "
                                << fixedPresentModeName(selected);
                            if (selected == VK_PRESENT_MODE_FIFO_KHR)
                                std::cerr << " fallback";
                            std::cerr << '\n';
                        }
                    }

                    const auto maintenanceIt =
                        instance_info->swapchainMaintenanceFamilies.find(device);
                    if (maintenanceIt != instance_info->swapchainMaintenanceFamilies.end()) {
                        const auto getCapabilities2 = reinterpret_cast<
                            PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
                                layer_info->GetInstanceProcAddr(
                                    instance_info->handles.front(),
                                    "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
                        try {
                            const uint32_t stableAdaptiveImageFloor =
                                configSnapshot.fixedMode()
                                    ? 0U
                                    : applicationMinImageCount
                                        + static_cast<uint32_t>(
                                            ls::GameConf::MAX_ADAPTIVE_MULTIPLIER);

                            if (hasSwapchainPresentModesCreateInfo(*newInfo)) {
                                if (adoptApplicationDualPresentModeDeclaration(
                                        it->second, getCapabilities2, maintenanceIt->second,
                                        *newInfo, stableAdaptiveImageFloor,
                                        dualPresentModes)) {
                                    dualPresentModeDeclared = true;
                                } else {
                                    std::cerr << "lsfg-vk: application present-mode declaration "
                                        "is not usable for LSFG dual mode; keeping legacy WSI "
                                        "swapchain\n";
                                }
                            } else if (prepareDualPresentModeDeclaration(
                                    it->second, getCapabilities2, maintenanceIt->second,
                                    *newInfo, stableAdaptiveImageFloor,
                                    dualPresentModes, dualPresentModesInfo)) {
                                dualPresentModeDeclared = true;
                            } else {
                                std::cerr << "lsfg-vk: compatible dual present modes unavailable; "
                                    "keeping legacy WSI swapchain\n";
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "lsfg-vk: dual present-mode query failed; "
                                "keeping legacy WSI swapchain:\n- " << e.what() << '\n';
                        }
                    }

                    auto res = it->second.df().CreateSwapchainKHR(
                        device, newInfo, alloc, swapchain);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateSwapchainKHR() failed");
                }
            );

            uint32_t imageCount{};
            auto res = it->second.df().GetSwapchainImagesKHR(device, *swapchain,
                &imageCount, VK_NULL_HANDLE);
            if (res != VK_SUCCESS || imageCount == 0)
                throw ls::vulkan_error(res, "vkGetSwapchainImagesKHR() failed");

            std::vector<VkImage> realImages(imageCount);
            res = it->second.df().GetSwapchainImagesKHR(device, *swapchain,
                &imageCount, realImages.data());
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res, "vkGetSwapchainImagesKHR() failed");

            std::shared_ptr<VirtualSwapchainRuntime> virtualRuntime;
            std::vector<VkImage> applicationImages = realImages;
            bool virtualized{};
            // Pass 4A D2 foundation: expose the downstream WSI image set
            // directly. Shadow/carrier transformation is reserved for 4B.
            bool dynamicPresentModeEligible{};

            const bool fixedMode = configSnapshot.fixedMode();
            const bool adaptiveMode = !fixedMode;

            VkPresentModeKHR adaptivePresentMode = VK_PRESENT_MODE_FIFO_KHR;
            VkPresentModeKHR fixedPresentMode = newInfo.presentMode;
            if (dualPresentModeDeclared
                    && containsPresentMode(dualPresentModes, VK_PRESENT_MODE_FIFO_KHR)) {
                if (containsPresentMode(dualPresentModes, VK_PRESENT_MODE_MAILBOX_KHR)) {
                    fixedPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                    dynamicPresentModeEligible = true;
                } else if (containsPresentMode(
                        dualPresentModes, VK_PRESENT_MODE_IMMEDIATE_KHR)) {
                    fixedPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    dynamicPresentModeEligible = true;
                }
            }

            const auto queueIt = instance_info->offloadQueues.find(device);
            const auto spec = makeVirtualSwapchainImageSpec(newInfo);
            const bool queueAvailable =
                queueIt != instance_info->offloadQueues.end();
            const bool queuePresentSupported = !d2Foundation && queueAvailable
                && offloadQueueSupportsSurface(
                    it->second, queueIt->second, newInfo.surface);
            std::vector<uint32_t> surfacePresentFamilies;
            if (!d2Foundation) {
                uint32_t surfaceFamilyCount{};
                it->second.fi().GetPhysicalDeviceQueueFamilyProperties(
                    it->second.physdev(), &surfaceFamilyCount, nullptr);
                for (uint32_t family = 0; family < surfaceFamilyCount; ++family) {
                    VkBool32 supported{};
                    if (it->second.fi().GetPhysicalDeviceSurfaceSupportKHR(
                            it->second.physdev(), family, newInfo.surface, &supported)
                            == VK_SUCCESS && supported)
                        surfacePresentFamilies.push_back(family);
                }
            }
            const bool topologyEligible = fixedMode
                ? fixedAsyncPresentModeEligible
                : dualPresentModeDeclared;
            const bool timelineAvailable = !d2Foundation && instance_info
                ->deviceTimelineSemaphoreAvailable.at(device);
            const auto maintenanceIt =
                instance_info->swapchainMaintenanceFamilies.find(device);
            const bool presentFenceAvailable = maintenanceIt
                    != instance_info->swapchainMaintenanceFamilies.end()
                && maintenanceIt->second != SwapchainMaintenanceFamily::None;

            if (d2Foundation) {
                if (configSnapshot.activeProfile().multiplier > 1)
                    std::cerr << "lsfg-vk: effective multiplier=1 reason="
                        "D2_FOUNDATION_NO_CARRIER\n";
            } else if (topologyEligible
                    && queueAvailable
                    && queuePresentSupported
                    && timelineAvailable
                    && presentFenceAvailable
                    && spec.supported()) {
                try {
                    virtualRuntime = std::make_shared<VirtualSwapchainRuntime>(
                        it->second,
                        queueIt->second.queue,
                        queueIt->second.mutex,
                        realImages.size(),
                        spec,
                        [&]() {
                            const auto maintenanceIt = instance_info->swapchainMaintenanceFamilies.find(device);
                            if (maintenanceIt == instance_info->swapchainMaintenanceFamilies.end())
                                return false;
                            const auto& funcs = it->second.df();
                            return (maintenanceIt->second == SwapchainMaintenanceFamily::KHR
                                    && funcs.ReleaseSwapchainImagesKHR)
                                || (maintenanceIt->second == SwapchainMaintenanceFamily::EXT
                                    && funcs.ReleaseSwapchainImagesEXT);
                        }());
                    applicationImages = virtualRuntime->imageHandles();
                    virtualized = true;
                    // The same runtime is valid for Adaptive and Fixed.
                } catch (const std::exception& e) {
                    std::cerr << "lsfg-vk: virtual swapchain setup failed; "
                        "keeping the legacy presentation path:\n";
                    std::cerr << "- " << e.what() << '\n';
                }
            } else if (!spec.supported()) {
                std::cerr << "lsfg-vk: swapchain flags are not supported by the "
                    "virtual bridge; keeping the legacy presentation path\n";
            } else if (!presentFenceAvailable) {
                std::cerr << "lsfg-vk: swapchain maintenance1 present-fence "
                    "authority unavailable; transformed WSI path is ineligible\n";
            } else if (adaptiveMode) {
                std::cerr << "lsfg-vk: Adaptive virtual topology prerequisites "
                    "unavailable; keeping the legacy Adaptive path\n";
            } else if (!fixedAsyncPresentModeEligible) {
                std::cerr << "lsfg-vk: asynchronous Fixed WSI prerequisites unavailable; "
                    "keeping FIFO synchronous 3B path\n";
            }

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
            if (queuePresentHarnessInstalled
                    && queuePresentHarnessFailD2Bookkeeping.exchange(false))
                throw std::bad_alloc();
#endif
            auto [infoIt, inserted] = instance_info->swapchainInfos.emplace(
                *swapchain,
                SwapchainInfo {
                    .images = std::move(applicationImages),
                    .realImages = std::move(realImages),
                    .format = newInfo.imageFormat,
                    .colorSpace = newInfo.imageColorSpace,
                    .extent = newInfo.imageExtent,
                    .arrayLayers = newInfo.imageArrayLayers,
                    .usage = spec.effectiveUsage,
                    .timelineSemaphoreAvailable = timelineAvailable,
                    .sharingMode = newInfo.imageSharingMode,
                    .queueFamilyIndices = newInfo.imageSharingMode == VK_SHARING_MODE_CONCURRENT
                        && newInfo.queueFamilyIndexCount && newInfo.pQueueFamilyIndices
                        ? std::vector<uint32_t>(newInfo.pQueueFamilyIndices,
                            newInfo.pQueueFamilyIndices + newInfo.queueFamilyIndexCount)
                        : std::vector<uint32_t>{},
                    .surfacePresentFamilies = std::move(surfacePresentFamilies),
                    .surfaceSupportsTransferSrc = [&]() {
                        if (d2Foundation) return false;
                        VkSurfaceCapabilitiesKHR capabilities{};
                        const auto query = it->second.fi().GetPhysicalDeviceSurfaceCapabilitiesKHR(
                            it->second.physdev(), newInfo.surface, &capabilities);
                        return query == VK_SUCCESS
                            && (capabilities.supportedUsageFlags
                                & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
                    }(),
                    .presentMode = newInfo.presentMode,
                    .adaptivePresentMode = adaptivePresentMode,
                    .fixedPresentMode = fixedPresentMode,
                    .virtualized = virtualized,
                    .d2Foundation = d2Foundation,
                    .d2State = std::make_shared<D2RealWsiState>(imageCount),
                    .requestedMultiplier = static_cast<uint32_t>(
                        configSnapshot.activeProfile().multiplier),
                    .effectiveMultiplier = 1,
                    .multiplierReason = configSnapshot.activeProfile().multiplier > 1
                        ? D2MultiplierReason::FoundationNoCarrier
                        : D2MultiplierReason::None,
                    .dynamicPresentModeEligible =
                        virtualized && dynamicPresentModeEligible,
                    .fixedContext = fixedMode,
                    .presentFenceAvailable = presentFenceAvailable,
                    .releaseBackend = [&]() {
                        const auto maintenanceIt = instance_info->swapchainMaintenanceFamilies.find(device);
                        if (maintenanceIt == instance_info->swapchainMaintenanceFamilies.end())
                            return SwapchainReleaseBackend::None;
                        const auto& funcs = it->second.df();
                        if (maintenanceIt->second == SwapchainMaintenanceFamily::KHR && funcs.ReleaseSwapchainImagesKHR)
                            return SwapchainReleaseBackend::Khr;
                        if (maintenanceIt->second == SwapchainMaintenanceFamily::EXT && funcs.ReleaseSwapchainImagesEXT)
                            return SwapchainReleaseBackend::Ext;
                        return SwapchainReleaseBackend::None;
                    }()
                });
            if (!inserted)
                throw ls::error("swapchain info already exists");

            auto& swapchainInfo = infoIt->second;

            try {
                if (d2Foundation) {
                    const auto& df = it->second.df();
                    VkFormatProperties2 formatProperties{
                        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
                    if (it->second.fi().GetPhysicalDeviceFormatProperties2)
                        it->second.fi().GetPhysicalDeviceFormatProperties2(
                            it->second.physdev(), newInfo.imageFormat,
                            &formatProperties);
                    const auto transferFeatures =
                        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
                        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
                    const bool usageEligible =
                        (spec.effectiveUsage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
                        == (VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    const bool formatEligible =
                        (formatProperties.formatProperties.optimalTilingFeatures
                            & transferFeatures)
                        == transferFeatures;
                    VkExternalSemaphoreProperties semaphoreProperties{
                        .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
                    const VkPhysicalDeviceExternalSemaphoreInfo semaphoreInfo{
                        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
                        .handleType =
                            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
                    if (it->second.fi()
                            .GetPhysicalDeviceExternalSemaphoreProperties)
                        it->second.fi().GetPhysicalDeviceExternalSemaphoreProperties(
                            it->second.physdev(), &semaphoreInfo,
                            &semaphoreProperties);
                    VkExternalFenceProperties fenceProperties{
                        .sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES};
                    const VkPhysicalDeviceExternalFenceInfo fenceInfo{
                        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
                        .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
                    if (it->second.fi().GetPhysicalDeviceExternalFenceProperties)
                        it->second.fi().GetPhysicalDeviceExternalFenceProperties(
                            it->second.physdev(), &fenceInfo, &fenceProperties);
                    const bool externalSyncFdEligible =
                        d2ExternalSyncFdFeaturesEligible(
                            semaphoreProperties.externalSemaphoreFeatures,
                            fenceProperties.externalFenceFeatures);
                    const bool dispatchEligible = df.CmdCopyImage
                        && df.QueueSubmit && df.CreateSemaphore
                        && df.DestroySemaphore && df.CreateFence
                        && df.DestroyFence && df.GetSemaphoreFdKHR
                        && df.ImportSemaphoreFdKHR && df.GetFenceFdKHR;
                    const bool syncFdEnabled = instance_info
                        ->deviceD2SyncFdAvailable.contains(device)
                        && instance_info->deviceD2SyncFdAvailable.at(device);
                    const bool singlePhysicalDevice = instance_info
                        ->deviceD2SinglePhysicalDevice.contains(device)
                        && instance_info->deviceD2SinglePhysicalDevice.at(device);
                    const bool queueEligible = queueAvailable
                        && queueIt->second.familyIndex
                            == it->second.queueFamilyIndex()
                        && offloadQueueSupportsSurface(
                            it->second, queueIt->second, newInfo.surface);
                    const bool carrierEligible = usageEligible
                        && formatEligible && newInfo.imageArrayLayers == 1
                        && !(newInfo.flags & VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR)
                        && presentFenceAvailable
                        && swapchainInfo.releaseBackend
                            != SwapchainReleaseBackend::None
                        && dispatchEligible
                        && syncFdEnabled && singlePhysicalDevice
                        && externalSyncFdEligible
                        && queueEligible;
                    if (carrierEligible) {
                        try {
                            swapchainInfo.d2VulkanRuntime =
                                std::make_shared<D2VulkanShadowRuntime>(
                                    it->second, queueIt->second.queue,
                                    queueIt->second.mutex,
                                    swapchainInfo.realImages,
                                    newInfo.imageFormat, newInfo.imageExtent);
                            instance_info->deviceD2VulkanRuntimeOwners.at(device)
                                .push_back(swapchainInfo.d2VulkanRuntime);
                            swapchainInfo.d2State->configureCarrierCapability(
                                true, true, queueIt->second.familyIndex);
                        } catch (const std::exception& e) {
                            swapchainInfo.d2VulkanRuntime.reset();
                            swapchainInfo.d2State->configureCarrierCapability(
                                false, presentFenceAvailable);
                            std::cerr << "lsfg-vk: D2 shadow allocation unavailable; "
                                "keeping native 1x: " << e.what() << '\n';
                        }
                    } else {
                        swapchainInfo.d2State->configureCarrierCapability(
                            false, presentFenceAvailable);
                    }
                }
                if (!d2Foundation) {
                std::optional<RuntimeExchangeQueue> runtimeExchangeQueue;
                if (queueIt != instance_info->offloadQueues.end()) {
                    runtimeExchangeQueue = RuntimeExchangeQueue{
                        .queue = queueIt->second.queue,
                        .familyIndex = queueIt->second.familyIndex,
                        .mutex = queueIt->second.mutex
                    };
                }
                layer_info->root.createSwapchainContext(
                    configSnapshot, it->second, *swapchain, swapchainInfo,
                    std::move(runtimeExchangeQueue));
                }

                if (virtualRuntime) {
                    const auto workerSwapchain = *swapchain;
                    auto* root = &layer_info->root;
                    virtualRuntime->setPrePresentGate([root, workerSwapchain]() {
                        return root->getSwapchainContext(workerSwapchain).prePresentGate();
                    });
                    const auto borrowedQueueMutex = std::make_shared<std::mutex>();
                    virtualRuntime->startWorker(
                        [workerSwapchain, borrowedQueueMutex, device](
                                uint32_t imageIndex,
                                VkSemaphore readySemaphore,
                                VkSemaphore originalReadySemaphore,
                                void* nextChain,
                                std::stop_token stopToken,
                                std::chrono::steady_clock::time_point sourcePresentTime,
                                bool d3bSingleSwapchainEligible,
                                const GraphicsFinalQueueInfo& graphicsFinalQueue,
                                BorrowedGraphicsQueueLease& graphicsLease,
                                bool& stopAfterCompletion,
                                PresentedPhysicalImageIdentity* presentedIdentity,
                                std::shared_ptr<void> virtualGpuBacking)
                                -> PresentExecutionResult {
                            const auto deviceIt = instance_info->devices.find(device);
                            if (deviceIt == instance_info->devices.end())
                                return PresentExecutionResult::failed(VK_ERROR_DEVICE_LOST);

                            try {
                                std::unique_ptr<VirtualPresentPendingOperation> pending;
                                const auto result = layer_info->root.presentSwapchain(
                                    deviceIt->second,
                                    graphicsFinalQueue.queue,
                                    borrowedQueueMutex,
                                    workerSwapchain,
                                    nextChain,
                                    imageIndex,
                                    (std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC")
                                        && std::strcmp(std::getenv("LSFGVK_D3B2_INSERTION_DIAGNOSTIC"), "1") == 0)
                                        ? std::vector<VkSemaphore>{readySemaphore, originalReadySemaphore}
                                        : std::vector<VkSemaphore>{readySemaphore},
                                    stopToken,
                                    sourcePresentTime,
                                    d3bSingleSwapchainEligible,
                                    &graphicsFinalQueue,
                                    &graphicsLease,
                                    &stopAfterCompletion,
                                    instance_info->deviceQuarantines.at(device),
                                    instance_info->deviceRetirementReactors.at(device),
                                    instance_info->devicePresentedPhysicalImages.at(device),
                                    instance_info->deviceBorrowedPresentFences.at(device),
                                    &pending, presentedIdentity,
                                    std::move(virtualGpuBacking));
                                if (pending)
                                    return PresentExecutionResult::pendingCompletion(
                                        result.result, std::move(pending),
                                        result.origin == SwapchainPresentResultOrigin::LOGICAL_DOWNSTREAM
                                            ? PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT
                                            : PresentResultOrigin::INTERNAL);
                                return PresentExecutionResult::completed(result.result,
                                    result.origin == SwapchainPresentResultOrigin::LOGICAL_DOWNSTREAM
                                        ? PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT
                                        : PresentResultOrigin::INTERNAL);
                            } catch (const LogicalPresentError& e) {
                                if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                                    std::cerr << "lsfg-vk: asynchronous virtual presentation failed:\n";
                                    std::cerr << "- " << e.what() << '\n';
                                }
                                return PresentExecutionResult::failed(e.error(),
                                    PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT);
                            } catch (const ls::vulkan_error& e) {
                                std::cerr << "lsfg-vk: internal asynchronous virtual presentation failed:\n";
                                std::cerr << "- " << e.what() << '\n';
                                return PresentExecutionResult::failed(e.error());
                            } catch (const std::exception& e) {
                                std::cerr << "lsfg-vk: asynchronous virtual presentation failed:\n";
                                std::cerr << "- " << e.what() << '\n';
                                return PresentExecutionResult::failed(VK_ERROR_UNKNOWN);
                            }
                        });
                    if (swapchainInfo.dynamicPresentModeEligible) {
                        std::cerr << "lsfg-vk: dual-mode virtual presentation worker enabled "
                            "(Adaptive=FIFO, Fixed="
                            << fixedPresentModeName(swapchainInfo.fixedPresentMode)
                            << ", initial=" << (fixedMode ? "Fixed" : "Adaptive") << ")\n";
                    } else if (fixedMode) {
                        std::cerr << "lsfg-vk: asynchronous Fixed presentation worker enabled\n";
                    } else {
                        std::cerr << "lsfg-vk: Adaptive virtual presentation worker enabled "
                            "(FIFO; 1x bypass or 2x-5x frame generation)\n";
                    }
                }
            } catch (...) {
                layer_info->root.removeSwapchainContext(*swapchain);
                instance_info->swapchainInfos.erase(*swapchain);
                throw;
            }

            if (virtualRuntime)
                instance_info->virtualSwapchains.emplace(
                    *swapchain, std::move(virtualRuntime));

            instance_info->swapchains.emplace(*swapchain,
                ls::R<vk::Vulkan>(it->second));

            d2DownstreamSwapchainCreated = false;
            return res;
        } catch (const ls::vulkan_error& e) {
            cleanupFailedD2Create();
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain creation:\n";
            std::cerr << "- " << e.what() << '\n';
            return e.error();
        } catch (const std::exception& e) {
            cleanupFailedD2Create();
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain creation:\n";
            std::cerr << "- " << e.what() << '\n';
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }



    VkResult myvkGetSwapchainImagesKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            uint32_t* count,
            VkImage* images) {
        const auto runtime = instance_info->virtualSwapchains.find(swapchain);
        if (runtime != instance_info->virtualSwapchains.end())
            return runtime->second->getImages(count, images);

        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;
        return deviceIt->second.df().GetSwapchainImagesKHR(
            device, swapchain, count, images);
    }

    VkResult releaseD2PhysicalImage(VkDevice device,
            VkSwapchainKHR swapchain, uint32_t index) {
        const auto deviceIt = instance_info->devices.find(device);
        const auto swapchainIt = instance_info->swapchainInfos.find(swapchain);
        if (deviceIt == instance_info->devices.end()
                || swapchainIt == instance_info->swapchainInfos.end())
            return VK_ERROR_DEVICE_LOST;
        const VkReleaseSwapchainImagesInfoKHR releaseInfo{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .swapchain = swapchain, .imageIndexCount = 1,
            .pImageIndices = &index};
        if (swapchainIt->second.releaseBackend == SwapchainReleaseBackend::Khr
                && deviceIt->second.df().ReleaseSwapchainImagesKHR)
            return deviceIt->second.df().ReleaseSwapchainImagesKHR(
                device, &releaseInfo);
        if (swapchainIt->second.releaseBackend == SwapchainReleaseBackend::Ext
                && deviceIt->second.df().ReleaseSwapchainImagesEXT)
            return deviceIt->second.df().ReleaseSwapchainImagesEXT(
                device, &releaseInfo);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VkResult myvkAcquireNextImageKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            uint64_t timeout,
            VkSemaphore semaphore,
            VkFence fence,
            uint32_t* imageIndex) {
        const auto runtime = instance_info->virtualSwapchains.find(swapchain);
        if (runtime != instance_info->virtualSwapchains.end())
            return runtime->second->acquire(timeout, semaphore, fence, imageIndex);

        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;
        const auto d2 = instance_info->swapchainInfos.find(swapchain);
        if (d2 != instance_info->swapchainInfos.end()
                && d2->second.d2Foundation && d2->second.d2State
                && d2->second.d2VulkanRuntime
                && d2->second.d2State->publicAcquireRelayRequired()) {
            return d2->second.d2VulkanRuntime->publicAcquire(
                *d2->second.d2State, semaphore, fence, imageIndex,
                [&](VkSemaphore internalSemaphore, VkFence internalFence,
                        uint32_t* internalIndex) {
                    return deviceIt->second.df().AcquireNextImageKHR(
                        device, swapchain, timeout, internalSemaphore,
                        internalFence, internalIndex);
                }, [&](uint32_t index) {
                    return releaseD2PhysicalImage(
                        device, swapchain, index);
                }, instance_info->deviceAcquireFenceProxies.at(device).get());
        }
        const auto result = deviceIt->second.df().AcquireNextImageKHR(
            device, swapchain, timeout, semaphore, fence, imageIndex);
        if ((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) && imageIndex) {
            const auto state = instance_info->swapchainInfos.find(swapchain);
            if (state != instance_info->swapchainInfos.end()
                    && state->second.d2Foundation && state->second.d2State
                    && !state->second.d2State->acquired(*imageIndex))
                return VK_ERROR_DEVICE_LOST;
        }
        return result;
    }

    VkResult myvkAcquireNextImage2KHR(
            VkDevice device,
            const VkAcquireNextImageInfoKHR* info,
            uint32_t* imageIndex) {
        if (!info)
            return VK_ERROR_INITIALIZATION_FAILED;
        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;

        const auto runtime = instance_info->virtualSwapchains.find(info->swapchain);
        if (runtime != instance_info->virtualSwapchains.end()) {
            return runtime->second->acquire(
                info->timeout, info->semaphore, info->fence, imageIndex);
        }

        auto next = reinterpret_cast<PFN_vkAcquireNextImage2KHR>(
            instance_info->funcs.GetDeviceProcAddr(device, "vkAcquireNextImage2KHR"));
        if (!next)
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        const auto d2 = instance_info->swapchainInfos.find(info->swapchain);
        if (d2 != instance_info->swapchainInfos.end()
                && d2->second.d2Foundation && d2->second.d2State
                && d2->second.d2VulkanRuntime && info->deviceMask == 1
                && d2->second.d2State->publicAcquireRelayRequired()) {
            return d2->second.d2VulkanRuntime->publicAcquire(
                *d2->second.d2State, info->semaphore, info->fence, imageIndex,
                [&](VkSemaphore internalSemaphore, VkFence internalFence,
                        uint32_t* internalIndex) {
                    auto downstreamInfo = *info;
                    downstreamInfo.semaphore = internalSemaphore;
                    downstreamInfo.fence = internalFence;
                    return next(device, &downstreamInfo, internalIndex);
                }, [&](uint32_t index) {
                    return releaseD2PhysicalImage(
                        device, info->swapchain, index);
                }, instance_info->deviceAcquireFenceProxies.at(device).get());
        }
        const auto result = next(device, info, imageIndex);
        if ((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) && imageIndex) {
            const auto state = instance_info->swapchainInfos.find(info->swapchain);
            if (state != instance_info->swapchainInfos.end()
                    && state->second.d2Foundation && state->second.d2State
                    && !state->second.d2State->acquired(*imageIndex))
                return VK_ERROR_DEVICE_LOST;
        }
        return result;
    }

    template<class Downstream>
    VkResult releaseD2SwapchainImages(VkDevice device,
            const VkReleaseSwapchainImagesInfoKHR* info,
            Downstream&& downstream) {
        if (!info || info->swapchain == VK_NULL_HANDLE
                || (info->imageIndexCount != 0 && !info->pImageIndices))
            return VK_ERROR_UNKNOWN;
        const auto state = instance_info->swapchainInfos.find(info->swapchain);
        if (state == instance_info->swapchainInfos.end()
                || !state->second.d2Foundation || !state->second.d2State)
            return downstream();
        const std::span<const uint32_t> indices(
            info->pImageIndices, info->imageIndexCount);
        if (!state->second.d2State->prepareRelease(indices))
            return VK_ERROR_UNKNOWN;
        const auto result = downstream();
        state->second.d2State->finishRelease(indices, result == VK_SUCCESS);
        return result;
    }

    VkResult myvkReleaseSwapchainImagesKHR(VkDevice device,
            const VkReleaseSwapchainImagesInfoKHR* info) {
        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt == instance_info->devices.end()
                || !deviceIt->second.df().ReleaseSwapchainImagesKHR)
            return VK_ERROR_UNKNOWN;
        return releaseD2SwapchainImages(device, info, [&] {
            return deviceIt->second.df().ReleaseSwapchainImagesKHR(device, info);
        });
    }

    VkResult myvkReleaseSwapchainImagesEXT(VkDevice device,
            const VkReleaseSwapchainImagesInfoKHR* info) {
        const auto deviceIt = instance_info->devices.find(device);
        if (deviceIt == instance_info->devices.end()
                || !deviceIt->second.df().ReleaseSwapchainImagesEXT)
            return VK_ERROR_UNKNOWN;
        return releaseD2SwapchainImages(device, info, [&] {
            return deviceIt->second.df().ReleaseSwapchainImagesEXT(device, info);
        });
    }

    VkResult myvkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"

        // ensure layer config is up to date
        std::optional<ConfigSnapshot> reload;
        try {
            reload = layer_info->root.update();
        } catch (const std::exception&) {
            reload = std::nullopt; // ignore parse errors
        }

        if (reload.has_value()) {
            try {
                const auto& configSnapshot = *reload;
                for (const auto& [swapchain, vk] : instance_info->swapchains) {
                    auto& swapchainInfo = instance_info->swapchainInfos.at(swapchain);

                    // Virtual VkImage handles are stable for the life of the
                    // application's swapchain. If LSFG declared compatible
                    // FIFO + MAILBOX/IMMEDIATE modes at creation, a mode reload
                    // can select the new WSI mode per present and replace only
                    // the Root-owned Swapchain context.
                    const bool requestedFixed = configSnapshot.fixedMode();
                    const bool modeChanged =
                        swapchainInfo.fixedContext != requestedFixed;

                    if (swapchainInfo.d2Foundation) {
                        swapchainInfo.requestedMultiplier = static_cast<uint32_t>(
                            configSnapshot.activeProfile().multiplier);
                        swapchainInfo.effectiveMultiplier = 1;
                        swapchainInfo.multiplierReason =
                            swapchainInfo.requestedMultiplier > 1
                                ? D2MultiplierReason::FoundationNoCarrier
                                : D2MultiplierReason::None;
                        swapchainInfo.fixedContext = requestedFixed;
                        continue;
                    }

                    if (swapchainInfo.virtualized && modeChanged
                            && !swapchainInfo.dynamicPresentModeEligible) {
                        std::cerr << "lsfg-vk: frame_generation_mode change requires "
                            "a dual-mode virtual swapchain; keeping current "
                            << (swapchainInfo.fixedContext ? "Fixed" : "Adaptive")
                            << " context\n";
                        continue;
                    }

                    if (swapchainInfo.dynamicPresentModeEligible) {
                        swapchainInfo.presentMode = requestedFixed
                            ? swapchainInfo.fixedPresentMode
                            : swapchainInfo.adaptivePresentMode;
                    }

                    std::optional<RuntimeExchangeQueue> runtimeExchangeQueue;
                    const auto queueIt = instance_info->offloadQueues.find(vk.get().dev());
                    if (queueIt != instance_info->offloadQueues.end()) {
                        runtimeExchangeQueue = RuntimeExchangeQueue{
                            .queue = queueIt->second.queue,
                            .familyIndex = queueIt->second.familyIndex,
                            .mutex = queueIt->second.mutex
                        };
                    }
                    layer_info->root.recreateSwapchainContext(
                        configSnapshot, vk, swapchain, swapchainInfo,
                        std::move(runtimeExchangeQueue));

                    if (swapchainInfo.virtualized) {
                        if (modeChanged && swapchainInfo.dynamicPresentModeEligible) {
                            std::cerr << "lsfg-vk: dual-mode hot switch "
                                << (swapchainInfo.fixedContext ? "Fixed" : "Adaptive")
                                << " -> " << (requestedFixed ? "Fixed" : "Adaptive")
                                << " present="
                                << fixedPresentModeName(swapchainInfo.presentMode)
                                << "\n";
                        } else if (!requestedFixed) {
                            std::cerr << "lsfg-vk: Adaptive virtual context hot-reloaded\n";
                        } else {
                            std::cerr << "lsfg-vk: Fixed virtual context hot-reloaded\n";
                        }
                    }

                    swapchainInfo.fixedContext = requestedFixed;
                }

                std::cerr << "lsfg-vk: updated lsfg-vk configuration\n";
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk configuration update:\n";
                std::cerr << "- " << e.what() << '\n';
            }
        }

        if (!info || info->swapchainCount == 0 || !info->pSwapchains
                || !info->pImageIndices)
            return VK_ERROR_INITIALIZATION_FAILED;

        bool allD2 = true;
        for (uint32_t i = 0; i < info->swapchainCount; ++i) {
            const auto state = instance_info->swapchainInfos.find(info->pSwapchains[i]);
            if (state == instance_info->swapchainInfos.end()
                    || !state->second.d2Foundation || !state->second.d2State) {
                allD2 = false;
                break;
            }
        }
        if (allD2) {
            for (uint32_t i = 0; i < info->swapchainCount; ++i)
                if (!instance_info->swapchainInfos.at(info->pSwapchains[i])
                        .d2State->canPresent(info->pImageIndices[i]))
                    return VK_ERROR_DEVICE_LOST;
            const auto first = instance_info->swapchains.find(info->pSwapchains[0]);
            if (first == instance_info->swapchains.end())
                return VK_ERROR_DEVICE_LOST;
            const auto result = first->second.get().df().QueuePresentKHR(queue, info);
            const auto aggregateClass = classifyPresentResult(result);
            if (aggregateClass == PresentResultClass::Normal
                    || aggregateClass == PresentResultClass::EnqueuedRejection) {
                const auto queueMetadata = instance_info->queues.find(queue);
                const std::optional<uint32_t> presentFamily =
                    queueMetadata == instance_info->queues.end()
                        ? std::nullopt
                        : std::optional<uint32_t>{queueMetadata->second.family};
                for (uint32_t i = 0; i < info->swapchainCount; ++i) {
                    const auto perResult = info->pResults ? info->pResults[i] : result;
                    const auto classification = classifyPresentResult(perResult);
                    if (classification == PresentResultClass::Normal
                            || classification == PresentResultClass::EnqueuedRejection)
                        instance_info->swapchainInfos.at(info->pSwapchains[i])
                            .d2State->presented(
                                info->pImageIndices[i], presentFamily);
                }
            }
            return result;
        }

        bool anyVirtual{};
        for (uint32_t i = 0; i < info->swapchainCount; ++i)
            anyVirtual = anyVirtual
                || instance_info->virtualSwapchains.contains(info->pSwapchains[i]);

        // A multi-swapchain VkPresentInfoKHR is one application queue operation:
        // it owns one binary wait set, may carry per-swapchain pNext arrays, and
        // can have platform atomicity semantics.  The current Root presenter is
        // still per-swapchain and cannot reproduce that transaction without a
        // batch-aware wait authority.  Preserve native application semantics by
        // bypassing LSFG transformation for non-virtual batches.  Virtual image
        // indices cannot be forwarded to the real swapchain, so those batches
        // fail before any queue side effect until a batch authority exists.
        if (info->swapchainCount > 1) {
            if (anyVirtual) {
                const auto failBeforeBridge = [&](VkResult result) noexcept {
                    if (info->pResults)
                        for (uint32_t i = 0; i < info->swapchainCount; ++i)
                            info->pResults[i] = result;
                    return result;
                };
                const auto queueMetadata = instance_info->queues.find(queue);
                if (queueMetadata == instance_info->queues.end())
                    return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                const auto device = queueMetadata->second.device;
                auto deviceVk = instance_info->devices.find(device);
                auto reactorIt = instance_info->deviceRetirementReactors.find(device);
                auto quarantineIt = instance_info->deviceQuarantines.find(device);
                auto borrowedIt = instance_info->deviceBorrowedPresentFences.find(device);
                if (deviceVk == instance_info->devices.end()
                        || reactorIt == instance_info->deviceRetirementReactors.end()
                        || quarantineIt == instance_info->deviceQuarantines.end()
                        || borrowedIt == instance_info->deviceBorrowedPresentFences.end())
                    return failBeforeBridge(VK_ERROR_DEVICE_LOST);

                struct VirtualBatchEntry final {
                    uint32_t logicalIndex{};
                    std::shared_ptr<VirtualSwapchainRuntime> runtime;
                    std::optional<VirtualSwapchainRuntime::BatchPresentReservation>
                        runtimeReservation;
                    std::optional<Adaptive1xPreparationReservation> preparation;
                    std::optional<Adaptive1xPreparedLogicalFinal> prepared;
                };
                struct D2BatchEntry final {
                    uint32_t logicalIndex{};
                    uint32_t imageIndex{};
                    uint64_t generation{};
                    std::shared_ptr<D2RealWsiState> state;
                    bool reservationActive{true};

                    D2BatchEntry() = default;
                    D2BatchEntry(uint32_t logical, uint32_t image,
                            std::shared_ptr<D2RealWsiState> value) noexcept
                        : logicalIndex(logical), imageIndex(image),
                          state(std::move(value)) {}
                    D2BatchEntry(const D2BatchEntry&) = delete;
                    D2BatchEntry& operator=(const D2BatchEntry&) = delete;
                    D2BatchEntry(D2BatchEntry&&) noexcept = default;
                    D2BatchEntry& operator=(D2BatchEntry&&) noexcept = default;
                    ~D2BatchEntry() {
                        if (state && reservationActive)
                            static_cast<void>(state->abortBatchPresent(
                                imageIndex, generation));
                    }
                    [[nodiscard]] bool commit(
                            std::optional<uint32_t> family) noexcept {
                        if (!state || !reservationActive) return false;
                        reservationActive = false;
                        return state->commitBatchPresent(
                            imageIndex, generation, family);
                    }
                    [[nodiscard]] bool abort() noexcept {
                        if (!state || !reservationActive) return false;
                        reservationActive = false;
                        return state->abortBatchPresent(imageIndex, generation);
                    }
                    void retainIndeterminate() noexcept {
                        reservationActive = false;
                    }
                };
                struct BatchSemaphoreBacking final {
                    explicit BatchSemaphoreBacking(const vk::Vulkan& vk) : value(vk) {}
                    vk::Semaphore value;
                };
                struct BatchFenceBacking final {
                    explicit BatchFenceBacking(const vk::Vulkan& vk)
                        : value(vk, vk::Fence::ExternalHandle::SyncFd) {}
                    vk::Fence value;
                };
                struct BatchLifetimeBacking final {
                    std::shared_ptr<D3B3DeviceLifetimeQuarantine> device;
                    std::vector<std::shared_ptr<D2RealWsiState>> d2States;
                };

                static std::atomic_uint64_t nextBatchTransaction{1};
                const auto transactionId = nextBatchTransaction.fetch_add(
                    1, std::memory_order_relaxed);
                auto transaction = BatchPresentTransaction::create(
                    device, queue, transactionId, *info);
                if (!transaction)
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);

                std::vector<PresentBatchPNextProjection::Path> projectionPaths;
                std::vector<VirtualBatchEntry> virtualEntries;
                std::vector<D2BatchEntry> d2Entries;
                std::vector<VkResult> mappedResults;
                try {
                    projectionPaths.reserve(info->swapchainCount);
                    virtualEntries.reserve(info->swapchainCount);
                    d2Entries.reserve(info->swapchainCount);
                    mappedResults.assign(info->swapchainCount, VK_SUCCESS);
                    for (uint32_t i = 0; i < info->swapchainCount; ++i) {
                        const auto swapchain = info->pSwapchains[i];
                        const auto vkIt = instance_info->swapchains.find(swapchain);
                        if (vkIt == instance_info->swapchains.end()
                                || vkIt->second.get().dev() != device)
                            return failBeforeBridge(VK_ERROR_UNKNOWN);
                        const auto runtimeIt =
                            instance_info->virtualSwapchains.find(swapchain);
                        if (runtimeIt != instance_info->virtualSwapchains.end()) {
                            const auto stateIt = instance_info->swapchainInfos.find(swapchain);
                            if (stateIt == instance_info->swapchainInfos.end()
                                    || stateIt->second.d2Foundation
                                    || stateIt->second.fixedContext
                                    || stateIt->second.effectiveMultiplier != 1)
                                return failBeforeBridge(VK_ERROR_UNKNOWN);
                            projectionPaths.push_back(
                                PresentBatchPNextProjection::Path::VirtualPrepared);
                            transaction->entry(i).path =
                                BatchPresentTransaction::Path::VirtualPrepared;
                            virtualEntries.push_back(VirtualBatchEntry{
                                .logicalIndex = i, .runtime = runtimeIt->second});
                        } else {
                            const auto stateIt = instance_info->swapchainInfos.find(swapchain);
                            if (stateIt != instance_info->swapchainInfos.end()
                                    && stateIt->second.d2Foundation) {
                                if (!stateIt->second.d2State)
                                    return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                                projectionPaths.push_back(
                                    PresentBatchPNextProjection::Path::D2);
                                transaction->entry(i).path =
                                    BatchPresentTransaction::Path::D2;
                                d2Entries.emplace_back(i, info->pImageIndices[i],
                                    stateIt->second.d2State);
                            } else {
                                projectionPaths.push_back(
                                    PresentBatchPNextProjection::Path::Native);
                                transaction->entry(i).path =
                                    BatchPresentTransaction::Path::Native;
                            }
                        }
                    }
                } catch (const std::bad_alloc&) {
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                }

                auto projection = PresentBatchPNextProjection::build(
                    info->pNext, info->swapchainCount, projectionPaths);
                if (!projection.supported())
                    return failBeforeBridge(VK_ERROR_UNKNOWN);
                for (uint32_t i = 0; i < info->swapchainCount; ++i)
                    if (projection.applicationFence(i) != VK_NULL_HANDLE)
                        return failBeforeBridge(VK_ERROR_UNKNOWN);

                for (auto& entry : d2Entries) {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
                    if (queuePresentHarnessFailD2BatchReservation.load(
                            std::memory_order_relaxed))
                        return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
#endif
                    const auto generation = entry.state->reserveBatchPresent(
                        entry.imageIndex);
                    if (!generation)
                        return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                    entry.generation = *generation;
                    auto& value = transaction->entry(entry.logicalIndex);
                    value.lifecycleGeneration = *generation + 1;
                    value.preparationAccepted = true;
                }

                std::shared_ptr<BatchLifetimeBacking> batchLifetime;
                try {
                    batchLifetime = std::make_shared<BatchLifetimeBacking>();
                    batchLifetime->device = quarantineIt->second;
                    batchLifetime->d2States.reserve(d2Entries.size());
                    for (const auto& entry : d2Entries)
                        batchLifetime->d2States.push_back(entry.state);
                } catch (const std::bad_alloc&) {
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                }

                for (auto& entry : virtualEntries) {
                    VkResult failure{VK_SUCCESS};
                    entry.runtimeReservation = entry.runtime->reserveBatchPresent(
                        info->pImageIndices[entry.logicalIndex], &failure);
                    if (!entry.runtimeReservation)
                        return failBeforeBridge(failure);
                    try {
                        entry.preparation.emplace(
                            layer_info->root.getSwapchainContext(
                                info->pSwapchains[entry.logicalIndex])
                                .reserveAdaptive1xPreparation(deviceVk->second, queue,
                                    queueMetadata->second.mutex,
                                    info->pSwapchains[entry.logicalIndex],
                                    info->pImageIndices[entry.logicalIndex],
                                    VK_NULL_HANDLE, {},
                                    entry.runtimeReservation->imageBacking(),
                                    reactorIt->second, borrowedIt->second,
                                    entry.runtimeReservation->completionPublication()));
                    } catch (const ls::vulkan_error& e) {
                        return failBeforeBridge(e.error());
                    } catch (const std::bad_alloc&) {
                        return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                    } catch (...) {
                        return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                    }
                }

                auto bridgeFence = std::make_shared<std::shared_ptr<BatchFenceBacking>>();
                auto retirementTicket = std::make_shared<DeviceRetirementTicket>();
                auto reservedJob = std::make_shared<DeviceRetirementReactor::ReservedJob>();
                auto jobReservation = reactorIt->second->reserveJob(
                    DeviceRetirementJob{
                        .deviceIdentity = quarantineIt->second->identity(),
                        .swapchainLifecycleIdentity = transactionId,
                        .operationIdentity = transactionId,
                        .ticket = retirementTicket});
                if (!jobReservation)
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                *reservedJob = std::move(*jobReservation);
                if (!reservedJob->valid())
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                auto bridge = BatchApplicationPresentBridgeAuthority::create(
                    queue, info->pWaitSemaphores, info->waitSemaphoreCount,
                    static_cast<uint32_t>(virtualEntries.size()),
                    batchLifetime,
                    BatchApplicationPresentBridgeAuthority::Operations{
                        .createSemaphore = [&deviceVk]()
                                -> std::optional<BatchApplicationPresentBridgeAuthority::OwnedSemaphore> {
                            try {
                                auto backing = std::make_shared<BatchSemaphoreBacking>(
                                    deviceVk->second);
                                return BatchApplicationPresentBridgeAuthority::OwnedSemaphore{
                                    backing->value.handle(), backing};
                            } catch (...) { return std::nullopt; }
                        },
                        .createFence = [&deviceVk, bridgeFence]()
                                -> std::optional<BatchApplicationPresentBridgeAuthority::OwnedFence> {
                            try {
                                auto backing = std::make_shared<BatchFenceBacking>(
                                    deviceVk->second);
                                *bridgeFence = backing;
                                return BatchApplicationPresentBridgeAuthority::OwnedFence{
                                    backing->value.handle(), backing};
                            } catch (...) { return std::nullopt; }
                        },
                        .submit = [dispatch = deviceVk->second.df(),
                                mutex = queueMetadata->second.mutex](VkQueue q,
                                const VkSubmitInfo& submit, VkFence fence) {
                            const std::scoped_lock lock(*mutex);
                            return dispatch.QueueSubmit(q, 1, &submit, fence);
                        },
                        .waitBridgeFence = [dispatch = deviceVk->second.df(), device](
                                VkFence fence) {
                            return dispatch.WaitForFences(
                                device, 1, &fence, VK_TRUE, UINT64_MAX);
                        },
                        .retireAsync = [reactor = reactorIt->second, reservedJob,
                                bridgeFence, vk = &deviceVk->second](VkFence,
                                std::shared_ptr<void> lifetime) {
                            try {
                                if (!*bridgeFence) return false;
                                const auto fd = (*bridgeFence)->value.exportSyncFd(*vk);
                                if (fd == -1) return true;
                                if (!reactor->registerReservedJob(
                                        *reservedJob, fd, std::move(lifetime))) {
                                    static_cast<void>(::close(fd));
                                    return false;
                                }
                                return true;
                            } catch (...) { return false; }
                        },
                        .retainConservatively = [reactor = reactorIt->second](
                                std::shared_ptr<void> value) {
                            return reactor->retainEventSourceLost(std::move(value));
                        }});
                if (!bridge || !transaction->markWaitBridgePrepared())
                    return failBeforeBridge(VK_ERROR_OUT_OF_HOST_MEMORY);
                for (uint32_t i = 0; i < virtualEntries.size(); ++i)
                    if (!virtualEntries[i].preparation->attachWaitBacking(
                            bridge->signalBacking(i)))
                        return failBeforeBridge(VK_ERROR_UNKNOWN);

                const auto bridgeResult = bridge->submit();
                if (bridgeResult != VK_SUCCESS)
                    return failBeforeBridge(bridgeResult);
                static_cast<void>(transaction->markWaitBridgeAccepted());
                for (auto& entry : virtualEntries)
                    if (!entry.runtimeReservation->commitAfterBridge()) {
                        bridge->retainDeviceLost();
                        entry.runtimeReservation->retainIndeterminate();
                        return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                    }
                static_cast<void>(transaction->markEntriesPreparing());

                VkResult postBridgeFailure{VK_SUCCESS};
                for (uint32_t i = 0; i < virtualEntries.size(); ++i) {
                    auto& virtualEntry = virtualEntries[i];
                    try {
                        virtualEntry.prepared.emplace(
                            virtualEntry.preparation->execute(bridge->signal(i)));
                    } catch (const ls::vulkan_error& e) {
                        postBridgeFailure = e.error();
                        break;
                    } catch (...) {
                        postBridgeFailure = VK_ERROR_DEVICE_LOST;
                        break;
                    }
                    if (!virtualEntry.prepared->valid()
                            || !bridge->markSignalConsumed(i)) {
                        postBridgeFailure = VK_ERROR_DEVICE_LOST;
                        break;
                    }
                    auto& value = transaction->entry(virtualEntry.logicalIndex);
                    value.physicalSwapchain =
                        virtualEntry.prepared->physicalSwapchain();
                    value.physicalImageIndex =
                        virtualEntry.prepared->physicalImageIndex();
                    value.completionSemaphore =
                        virtualEntry.prepared->presentReadySemaphore();
                    value.deviceLifetimeIdentity =
                        virtualEntry.prepared->presentedIdentity()
                            .deviceLifetimeIdentity;
                    value.lifecycleGeneration = virtualEntry.prepared->generation();
                    value.presentOperationIdentity =
                        virtualEntry.prepared->presentedIdentity()
                            .presentOperationIdentity;
                    value.preparationAccepted = true;
                    if (!projection.patchInternalPresentFence(
                            virtualEntry.logicalIndex,
                            virtualEntry.prepared->internalPresentFence())) {
                        postBridgeFailure = VK_ERROR_DEVICE_LOST;
                        break;
                    }
                }
                if (postBridgeFailure != VK_SUCCESS) {
                    for (auto& entry : virtualEntries) {
                        if (entry.prepared && entry.prepared->valid())
                            static_cast<void>(entry.prepared->abandonWithoutLogicalPresent());
                        if (entry.runtimeReservation)
                            static_cast<void>(entry.runtimeReservation->abortCleanly());
                    }
                    for (auto& entry : d2Entries)
                        static_cast<void>(entry.abort());
                    const auto recovered = bridge->recoverPartial();
                    const auto result = recovered == VK_SUCCESS
                        ? publicPresentResult(postBridgeFailure,
                            PresentResultOrigin::INTERNAL,
                            PresentTransactionPhase::POST_COMMIT)
                        : VK_ERROR_DEVICE_LOST;
                    return failBeforeBridge(result);
                }

                static_cast<void>(transaction->markEntriesReady());
                uint32_t virtualOrdinal{};
                for (uint32_t i = 0; i < info->swapchainCount; ++i) {
                    if (transaction->entry(i).path
                            == BatchPresentTransaction::Path::VirtualPrepared) {
                        auto& prepared = *virtualEntries[virtualOrdinal++].prepared;
                        static_cast<void>(transaction->setFinalEntry(i,
                            prepared.physicalSwapchain(),
                            prepared.physicalImageIndex()));
                        static_cast<void>(transaction->appendFinalWait(
                            prepared.presentReadySemaphore()));
                    } else {
                        static_cast<void>(transaction->setFinalEntry(i,
                            info->pSwapchains[i], info->pImageIndices[i]));
                    }
                }
                auto finalInfo = transaction->prepareFinalPresentInfo(
                    projection.head(), info->pResults,
                    static_cast<uint32_t>(virtualEntries.size()));
                if (!finalInfo) {
                    bridge->retainDeviceLost();
                    for (auto& entry : virtualEntries)
                        entry.runtimeReservation->retainIndeterminate();
                    for (auto& entry : d2Entries)
                        entry.retainIndeterminate();
                    return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                }
                for (auto& entry : virtualEntries)
                    if (!entry.prepared->markLogicalPresentCalled()) {
                        bridge->retainDeviceLost();
                        entry.runtimeReservation->retainIndeterminate();
                        for (auto& d2Entry : d2Entries)
                            d2Entry.retainIndeterminate();
                        return failBeforeBridge(VK_ERROR_DEVICE_LOST);
                    }
                static_cast<void>(transaction->markFinalPresentCalled());
                const auto downstream = deviceVk->second.df().QueuePresentKHR(
                    queue, &*finalInfo);
                static_cast<void>(transaction->markFinalPresentResult(downstream));
                const auto downstreamClass = classifyPresentResult(downstream);
                VkResult publicResult = publicPresentResult(downstream,
                    PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
                    PresentTransactionPhase::POST_COMMIT);
                for (auto& entry : virtualEntries) {
                    const bool perEntryResultsDefined = info->pResults
                        && (downstreamClass == PresentResultClass::Normal
                            || downstreamClass
                                == PresentResultClass::EnqueuedRejection);
                    const auto logicalResult = perEntryResultsDefined
                        ? info->pResults[entry.logicalIndex] : downstream;
                    mappedResults[entry.logicalIndex] = logicalResult;
                    const auto finalized =
                        entry.prepared->finalizeLogicalPresent(logicalResult);
                    if (finalized != VK_SUCCESS) publicResult = VK_ERROR_DEVICE_LOST;
                    const auto logicalClass = classifyPresentResult(logicalResult);
                    if (logicalClass == PresentResultClass::Normal
                            || logicalClass == PresentResultClass::EnqueuedRejection) {
                        if (!entry.runtimeReservation->installPublishedCompletion())
                            publicResult = VK_ERROR_DEVICE_LOST;
                    } else if (logicalClass == PresentResultClass::PreEnqueueFailure) {
                        if (!entry.runtimeReservation->abortCleanly())
                            publicResult = VK_ERROR_DEVICE_LOST;
                    } else {
                        entry.runtimeReservation->retainIndeterminate();
                    }
                }
                const auto queueFamily = std::optional<uint32_t>{
                    queueMetadata->second.family};
                for (auto& entry : d2Entries) {
                    const bool perEntryResultsDefined = info->pResults
                        && (downstreamClass == PresentResultClass::Normal
                            || downstreamClass
                                == PresentResultClass::EnqueuedRejection);
                    const auto logicalResult = perEntryResultsDefined
                        ? info->pResults[entry.logicalIndex] : downstream;
                    mappedResults[entry.logicalIndex] = logicalResult;
                    const auto logicalClass = classifyPresentResult(logicalResult);
                    if (logicalClass == PresentResultClass::Normal
                            || logicalClass
                                == PresentResultClass::EnqueuedRejection) {
                        if (!entry.commit(queueFamily))
                            publicResult = VK_ERROR_DEVICE_LOST;
                    } else if (logicalClass
                            == PresentResultClass::PreEnqueueFailure) {
                        if (!entry.abort()) publicResult = VK_ERROR_DEVICE_LOST;
                    } else {
                        entry.retainIndeterminate();
                    }
                }
                if (downstreamClass == PresentResultClass::DeviceLost
                        || downstreamClass == PresentResultClass::Indeterminate)
                    bridge->retainDeviceLost();
                else if (!bridge->retireNormalAsync()) {
                    bridge->retainDeviceLost();
                    publicResult = VK_ERROR_DEVICE_LOST;
                }
                if (downstreamClass == PresentResultClass::PreEnqueueFailure)
                    publicResult = VK_ERROR_DEVICE_LOST;
                if (info->pResults) {
                    for (uint32_t i = 0; i < info->swapchainCount; ++i) {
                        const bool perEntryResultsDefined =
                            downstreamClass == PresentResultClass::Normal
                            || downstreamClass
                                == PresentResultClass::EnqueuedRejection;
                        const auto actual = perEntryResultsDefined
                            ? info->pResults[i] : downstream;
                        info->pResults[i] = publicPresentResult(actual,
                            PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
                            PresentTransactionPhase::POST_COMMIT);
                        mappedResults[i] = info->pResults[i];
                    }
                    const auto mappedAggregate = aggregatePresentResults(mappedResults);
                    if (mappedAggregate != VK_SUCCESS)
                        publicResult = mappedAggregate;
                }
                return publicResult;
            }

            const auto first = instance_info->swapchains.find(info->pSwapchains[0]);
            if (first == instance_info->swapchains.end()) {
                if (info->pResults) {
                    for (uint32_t i = 0; i < info->swapchainCount; ++i)
                        info->pResults[i] = VK_ERROR_UNKNOWN;
                }
                return VK_ERROR_UNKNOWN;
            }
            // The unmodified application batch is the only currently-qualified
            // way to preserve one wait set, pNext array cardinality, aggregate
            // result semantics, and any implementation-supported atomic present.
            return first->second.get().df().QueuePresentKHR(queue, info);
        }

        // Single-swapchain calls need no pNext projection.  Keep caller memory
        // immutable and forward the original chain read-only.  Batch projection
        // is implemented/tested separately for the future batch-aware seam.
        void* projectedNext = const_cast<void*>(info->pNext);

        std::vector<VkResult> perSwapchainResults(
            info->swapchainCount, VK_SUCCESS);

        for (uint32_t i = 0; i < info->swapchainCount; ++i) {
            const auto swapchain = info->pSwapchains[i];
            const auto it = instance_info->swapchains.find(swapchain);
            if (it == instance_info->swapchains.end()) {
                perSwapchainResults[i] = VK_ERROR_DEVICE_LOST;
                if (info->pResults) info->pResults[i] = perSwapchainResults[i];
                continue;
            }

            std::vector<VkSemaphore> waitSemaphores;
            // The original VkPresentInfoKHR owns one binary wait set for the
            // complete queue operation.  Decomposition must never wait the same
            // binary signal more than once.  Queue issue order carries the
            // established dependency into later decomposed presents.
            if (i == 0) {
                waitSemaphores.reserve(info->waitSemaphoreCount);
                for (uint32_t j = 0; j < info->waitSemaphoreCount; ++j)
                    waitSemaphores.push_back(info->pWaitSemaphores[j]);
            }

            VkResult result = VK_SUCCESS;
            PresentedPhysicalImageIdentity presentedIdentity{};
            auto virtualIt = instance_info->virtualSwapchains.find(swapchain);
            try {
                if (virtualIt != instance_info->virtualSwapchains.end()) {
                    const bool synchronous = info->pNext != nullptr;
                    const auto queueMetadata = instance_info->queues.find(queue);
                    if (queueMetadata == instance_info->queues.end()) {
                        result = VK_ERROR_DEVICE_LOST;
                    } else {
                        const bool surfacePresentSupported = std::ranges::find(
                            instance_info->swapchainInfos.at(swapchain).surfacePresentFamilies,
                            queueMetadata->second.family)
                            != instance_info->swapchainInfos.at(swapchain)
                                .surfacePresentFamilies.end();
                        result = virtualIt->second->queuePresent(
                            queue,
                            queueMetadata->second.family,
                            queueMetadata->second.index,
                            queueMetadata->second.flags,
                            surfacePresentSupported,
                            info->pImageIndices[i],
                            waitSemaphores,
                            projectedNext,
                            synchronous,
                            &presentedIdentity);
                    }
                } else {
                    const auto execution = layer_info->root.presentSwapchain(
                        it->second,
                        queue,
                        nullptr,
                        swapchain,
                        projectedNext,
                        info->pImageIndices[i],
                        waitSemaphores,
                        {}, std::nullopt, false, nullptr, nullptr, nullptr,
                        instance_info->deviceQuarantines.at(it->second.get().dev()),
                        instance_info->deviceRetirementReactors.at(
                            it->second.get().dev()),
                        instance_info->devicePresentedPhysicalImages.at(
                            it->second.get().dev()),
                        instance_info->deviceBorrowedPresentFences.at(
                            it->second.get().dev()),
                        nullptr, &presentedIdentity);
                    result = publicPresentResult(execution.result,
                        execution.origin == SwapchainPresentResultOrigin::LOGICAL_DOWNSTREAM
                            ? PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT
                            : PresentResultOrigin::INTERNAL,
                        PresentTransactionPhase::POST_COMMIT);
                }
            } catch (const LogicalPresentError& e) {
                if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                    std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                    std::cerr << "- " << e.what() << '\n';
                }
                result = e.error();
            } catch (const ls::vulkan_error& e) {
                std::cerr << "lsfg-vk: internal failure during lsfg-vk swapchain presentation:\n";
                std::cerr << "- " << e.what() << '\n';
                result = publicPresentResult(e.error(), PresentResultOrigin::INTERNAL,
                    PresentTransactionPhase::POST_COMMIT);
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                std::cerr << "- " << e.what() << '\n';
                result = VK_ERROR_DEVICE_LOST;
            }

            perSwapchainResults[i] = result;
            if (info->pResults) info->pResults[i] = result;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
            if (queuePresentHarnessInstalled && presentedIdentity.valid())
                queuePresentHarnessLastIdentity = presentedIdentity;
#endif

            // If the first decomposed operation failed before queue acceptance,
            // later operations cannot safely inherit a wait that was never
            // consumed.  Resource/host-memory enqueue failures are guaranteed by
            // Vulkan to leave synchronization untouched, so stop immediately.
            if (i == 0 && (result == VK_ERROR_OUT_OF_HOST_MEMORY
                    || result == VK_ERROR_OUT_OF_DEVICE_MEMORY
                    || result == VK_ERROR_DEVICE_LOST
                    || result == VK_ERROR_INITIALIZATION_FAILED
                    || result == VK_ERROR_FEATURE_NOT_PRESENT
                    || result == VK_ERROR_UNKNOWN)) {
                for (uint32_t j = i + 1; j < info->swapchainCount; ++j) {
                    perSwapchainResults[j] = result;
                    if (info->pResults) info->pResults[j] = result;
                }
                break;
            }
        }

        return aggregatePresentResults(perSwapchainResults);
#pragma clang diagnostic pop
    }

    void myvkDestroySwapchainKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            const VkAllocationCallbacks* alloc) {
        const auto& it = instance_info->devices.find(device);
        if (it == instance_info->devices.end())
            return;

        const auto runtime = instance_info->virtualSwapchains.find(swapchain);
        const bool wasVirtual = runtime != instance_info->virtualSwapchains.end();
        if (wasVirtual) {
            const auto deferredOwner =
                instance_info->deviceDeferredVirtualRetirements.at(device);
            const auto detached = runtime->second->stopAndDetach(deferredOwner);
            if (!detached) std::terminate();
            const bool oneShot = d3b1OneShotRequested(
                std::getenv("LSFGVK_D3B1_ONESHOT"));
            if (oneShot)
                std::cerr << "[D3B1-ONESHOT] worker stopped and joined\n";
            // Public destruction cannot defer an application allocator. This
            // idle boundary covers other layer-generated swapchain use, but
            // exact pending-present authorities remain in deferredOwner.
            const auto idle = it->second.df().DeviceWaitIdle(device);
            deferredOwner->notifyDeviceRetirement();
            if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) {
                std::cerr << "lsfg-vk: virtual swapchain destroy idle boundary failed: "
                    << idle << '\n';
            }
        }

        layer_info->root.removeSwapchainContext(swapchain);

        if (runtime != instance_info->virtualSwapchains.end())
            instance_info->virtualSwapchains.erase(runtime);

        instance_info->swapchainInfos.erase(swapchain);
        instance_info->swapchains.erase(swapchain);

        it->second.df().DestroySwapchainKHR(device, swapchain, alloc);
    }
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
namespace lsfgvk::layer::test {
__attribute__((visibility("default")))
bool installDeviceEntrypointHarness(DeviceEntrypointHarnessConfig config) noexcept {
    if (deviceEntrypointTestContext || config.device == VK_NULL_HANDLE
            || !config.downstream.CreateFence || !config.downstream.DestroyFence
            || !config.downstream.GetFenceStatus || !config.downstream.WaitForFences
            || !config.downstream.ResetFences)
        return false;
    try {
        auto leases = std::make_shared<PresentedPhysicalImageLeaseRegistry>();
        auto borrowed = std::make_shared<BorrowedPresentFenceRegistry>(
            config.deviceLifetimeIdentity);
        auto proxies = std::make_shared<AcquireFenceProxyRegistry>(borrowed);
        deviceEntrypointTestContext = std::make_unique<DeviceEntrypointTestContext>(
            DeviceEntrypointTestContext{config.device, config.downstream,
                std::move(leases), std::move(borrowed), std::move(proxies)});
        return true;
    } catch (...) { return false; }
}

__attribute__((visibility("default")))
void uninstallDeviceEntrypointHarness() noexcept {
    deviceEntrypointTestContext.reset();
}

__attribute__((visibility("default")))
PFN_vkVoidFunction deviceEntrypoint(const char* name) noexcept {
    if (!deviceEntrypointTestContext || !name) return nullptr;
#define TEST_ENTRYPOINT(vkname, impl) \
    if (std::strcmp(name, vkname) == 0) \
        return reinterpret_cast<PFN_vkVoidFunction>(impl)
    TEST_ENTRYPOINT("vkCreateFence", myvkCreateFence);
    TEST_ENTRYPOINT("vkDestroyFence", myvkDestroyFence);
    TEST_ENTRYPOINT("vkGetFenceStatus", myvkGetFenceStatus);
    TEST_ENTRYPOINT("vkWaitForFences", myvkWaitForFences);
    TEST_ENTRYPOINT("vkResetFences", myvkResetFences);
    TEST_ENTRYPOINT("vkImportFenceFdKHR", myvkImportFenceFdKHR);
#undef TEST_ENTRYPOINT
    return nullptr;
}

__attribute__((visibility("default")))
bool installPresentedLease(PresentedPhysicalImageIdentity identity,
        std::shared_ptr<void> resources) noexcept {
    return deviceEntrypointTestContext
        && deviceEntrypointTestContext->leases->install(identity, std::move(resources));
}

__attribute__((visibility("default")))
bool associateBorrowedFence(VkFence fence,
        PresentedPhysicalImageIdentity identity) noexcept {
    return deviceEntrypointTestContext
        && deviceEntrypointTestContext->borrowed->associate(fence,
            deviceEntrypointTestContext->leases, identity);
}

__attribute__((visibility("default")))
bool containsPresentedLease(const PresentedPhysicalImageIdentity& identity) noexcept {
    return deviceEntrypointTestContext
        && deviceEntrypointTestContext->leases->contains(identity);
}

__attribute__((visibility("default")))
size_t borrowedAssociationCount() noexcept {
    return deviceEntrypointTestContext
        ? deviceEntrypointTestContext->borrowed->associationCount() : 0;
}

__attribute__((visibility("default")))
bool associateAcquireFenceProxy(VkFence fence, VkFence proxy,
        std::shared_ptr<void> backing) noexcept {
    if (!deviceEntrypointTestContext) return false;
    auto prepared = deviceEntrypointTestContext->proxies->prepare(
        fence, proxy, std::move(backing));
    if (!prepared) return false;
    deviceEntrypointTestContext->proxies->commit(*prepared);
    return !prepared->valid();
}

__attribute__((visibility("default")))
size_t acquireFenceProxyAssociationCount() noexcept {
    return deviceEntrypointTestContext
        ? deviceEntrypointTestContext->proxies->associationCount() : 0;
}

__attribute__((visibility("default")))
VkResult releaseD2PhysicalImageForTesting(VkDevice device,
        VkSwapchainKHR swapchain, uint32_t index) noexcept {
    if (!queuePresentHarnessInstalled) return VK_ERROR_INITIALIZATION_FAILED;
    return releaseD2PhysicalImage(device, swapchain, index);
}

__attribute__((visibility("default")))
bool installQueuePresentEntrypointHarness(
        QueuePresentEntrypointHarnessConfig config) noexcept {
    if (queuePresentHarnessInstalled || layer_info || instance_info
            || config.device == VK_NULL_HANDLE || config.queue == VK_NULL_HANDLE
            || config.swapchain == VK_NULL_HANDLE || !config.vulkan
            || !config.backend || config.vulkan->dev() != config.device)
        return false;
    try {
        auto ownedLayer = std::make_unique<LayerInfo>();
        auto ownedInstance = std::make_unique<InstanceInfo>();
        ls::GameConf profile{};
        profile.multiplier = config.frameGenerationMultiplier;
        profile.frame_generation_mode = ls::FrameGenerationMode::Adaptive;
        ownedLayer->root.setProfileForTesting(profile);
        ownedInstance->devices.emplace(config.device, std::move(*config.vulkan));
        auto& installedVk = ownedInstance->devices.at(config.device);
        ownedInstance->funcs = installedVk.fi();
        std::shared_ptr<VirtualSwapchainRuntime> virtualRuntime;
        auto swapchainInfo = config.swapchainInfo;
        if (config.virtualized) {
            VirtualSwapchainImageSpec spec{
                .extent = swapchainInfo.extent,
                .format = swapchainInfo.format,
                .arrayLayers = swapchainInfo.arrayLayers,
                .effectiveUsage = swapchainInfo.usage
                    ? swapchainInfo.usage
                    : VkImageUsageFlags2KHR(VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
                .usage = swapchainInfo.usage
                    ? static_cast<VkImageUsageFlags>(swapchainInfo.usage)
                    : VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
                .usageFitsLegacy = true,
                .hasRequiredTransferUsage = true,
                .sharingMode = swapchainInfo.sharingMode,
                .queueFamilyIndices = swapchainInfo.queueFamilyIndices};
            virtualRuntime = std::make_shared<VirtualSwapchainRuntime>(installedVk,
                config.queue, std::make_shared<std::mutex>(),
                swapchainInfo.realImages.size(), spec, false);
            swapchainInfo.images = virtualRuntime->imageHandles();
            swapchainInfo.virtualized = true;
        }
        auto context = std::make_shared<Swapchain>(installedVk,
            *config.backend, std::move(config.devicePair), std::move(profile),
            swapchainInfo, config.virtualized ? config.queueFamily
                                              : VK_QUEUE_FAMILY_IGNORED);
        if (!ownedLayer->root.installSwapchainContextForTesting(
                config.swapchain, std::move(context)))
            return false;
        ownedInstance->queues.emplace(config.queue, InstanceInfo::QueueMetadata{
            config.device, config.queueFamily, config.queueIndex, config.queueFlags});
        ownedInstance->swapchains.emplace(config.swapchain, ls::R<vk::Vulkan>(installedVk));
        ownedInstance->swapchainInfos.emplace(config.swapchain, swapchainInfo);
        ownedInstance->deviceD2VulkanRuntimeOwners.emplace(config.device,
            std::vector<std::shared_ptr<D2VulkanShadowRuntime>>{});
        if (swapchainInfo.d2VulkanRuntime)
            ownedInstance->deviceD2VulkanRuntimeOwners.at(config.device)
                .push_back(swapchainInfo.d2VulkanRuntime);
        auto quarantine = std::make_shared<D3B3DeviceLifetimeQuarantine>();
        ownedInstance->deviceQuarantines.emplace(config.device, quarantine);
        ownedInstance->deviceRetirementReactors.emplace(config.device,
            std::make_shared<DeviceRetirementReactor>());
        auto deferredOwner = std::make_shared<DeferredVirtualRetirementOwner>();
        ownedInstance->deviceDeferredVirtualRetirements.emplace(
            config.device, deferredOwner);
        ownedInstance->devicePresentedPhysicalImages.emplace(config.device,
            std::make_shared<PresentedPhysicalImageLeaseRegistry>());
        auto borrowed = std::make_shared<BorrowedPresentFenceRegistry>(
            quarantine->identity());
        ownedInstance->deviceBorrowedPresentFences.emplace(config.device, borrowed);
        ownedInstance->deviceAcquireFenceProxies.emplace(config.device,
            std::make_shared<AcquireFenceProxyRegistry>(borrowed));
        ownedLayer->root.getSwapchainContext(config.swapchain)
            .primeAdaptiveBuilderForTesting(installedVk, config.queueFamily,
                quarantine,
                ownedInstance->devicePresentedPhysicalImages.at(config.device));
        layer_info = ownedLayer.release();
        instance_info = ownedInstance.release();
        if (virtualRuntime) {
            auto* root = &layer_info->root;
            const auto swapchain = config.swapchain;
            const auto device = config.device;
            virtualRuntime->startWorker([root, swapchain, device](uint32_t imageIndex,
                    VkSemaphore ready, VkSemaphore, void* nextChain,
                    std::stop_token stopToken,
                    std::chrono::steady_clock::time_point sourceTime,
                    bool eligible, const GraphicsFinalQueueInfo& queueInfo,
                    BorrowedGraphicsQueueLease& lease, bool& stopAfter,
                    PresentedPhysicalImageIdentity* identity,
                    std::shared_ptr<void> backing) -> PresentExecutionResult {
                try {
                    const auto injected =
                        queuePresentHarnessWorkerInternalFailure.load();
                    if (injected != VK_SUCCESS)
                        throw ls::vulkan_error(injected,
                            "injected internal virtual presenter failure");
                    std::unique_ptr<VirtualPresentPendingOperation> pending;
                    const auto result = root->presentSwapchain(
                        instance_info->devices.at(device), queueInfo.queue,
                        std::make_shared<std::mutex>(), swapchain, nextChain,
                        imageIndex, {ready}, stopToken, sourceTime, eligible,
                        &queueInfo, &lease, &stopAfter,
                        instance_info->deviceQuarantines.at(device),
                        instance_info->deviceRetirementReactors.at(device),
                        instance_info->devicePresentedPhysicalImages.at(device),
                        instance_info->deviceBorrowedPresentFences.at(device),
                        &pending, identity, std::move(backing));
                    return pending
                        ? PresentExecutionResult::pendingCompletion(
                            result.result, std::move(pending),
                            result.origin == SwapchainPresentResultOrigin::LOGICAL_DOWNSTREAM
                                ? PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT
                                : PresentResultOrigin::INTERNAL)
                        : PresentExecutionResult::completed(result.result,
                            result.origin == SwapchainPresentResultOrigin::LOGICAL_DOWNSTREAM
                                ? PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT
                                : PresentResultOrigin::INTERNAL);
                } catch (const LogicalPresentError& e) {
                    return PresentExecutionResult::failed(e.error(),
                        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT);
                } catch (const ls::vulkan_error& e) {
                    return PresentExecutionResult::failed(e.error());
                } catch (...) {
                    return PresentExecutionResult::failed(VK_ERROR_UNKNOWN);
                }
            });
            instance_info->virtualSwapchains.emplace(
                config.swapchain, std::move(virtualRuntime));
        }
        queuePresentHarnessLastIdentity = {};
        queuePresentHarnessDeferredOwner = std::move(deferredOwner);
        queuePresentHarnessInstalled = true;
        return true;
    } catch (...) {
        return false;
    }
}

__attribute__((visibility("default")))
bool addInstalledVirtualSwapchainForTesting(VkSwapchainKHR swapchain) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info || !layer_info
            || swapchain == VK_NULL_HANDLE || instance_info->swapchains.empty()
            || instance_info->virtualSwapchains.empty())
        return false;
    try {
        const auto source = instance_info->virtualSwapchains.begin()->first;
        const auto& sourceInfo = instance_info->swapchainInfos.at(source);
        auto& installedVk = instance_info->swapchains.at(source).get();
        const auto queue = instance_info->queues.begin()->first;
        VirtualSwapchainImageSpec spec{
            .extent = sourceInfo.extent,
            .format = sourceInfo.format,
            .arrayLayers = sourceInfo.arrayLayers,
            .effectiveUsage = sourceInfo.usage,
            .usage = static_cast<VkImageUsageFlags>(sourceInfo.usage),
            .usageFitsLegacy = true,
            .hasRequiredTransferUsage = true,
            .sharingMode = sourceInfo.sharingMode,
            .queueFamilyIndices = sourceInfo.queueFamilyIndices};
        auto runtime = std::make_shared<VirtualSwapchainRuntime>(installedVk,
            queue, std::make_shared<std::mutex>(), sourceInfo.realImages.size(),
            spec, false);
        auto clonedInfo = sourceInfo;
        clonedInfo.images = runtime->imageHandles();
        auto context = layer_info->root.getSwapchainContext(source)
            .shared_from_this();
        if (!layer_info->root.installSwapchainContextForTesting(
                swapchain, context)) return false;
        instance_info->swapchains.emplace(
            swapchain, ls::R<vk::Vulkan>(installedVk));
        instance_info->swapchainInfos.emplace(swapchain, std::move(clonedInfo));
        runtime->startWorker([](uint32_t, VkSemaphore, VkSemaphore, void*,
                std::stop_token, std::chrono::steady_clock::time_point, bool,
                const GraphicsFinalQueueInfo&, BorrowedGraphicsQueueLease&,
                bool&, PresentedPhysicalImageIdentity*, std::shared_ptr<void>) {
            return PresentExecutionResult::failed(VK_ERROR_DEVICE_LOST);
        });
        instance_info->virtualSwapchains.emplace(swapchain, std::move(runtime));
        return true;
    } catch (...) { return false; }
}

__attribute__((visibility("default")))
bool addInstalledNativeSwapchainForTesting(VkSwapchainKHR swapchain) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info || !layer_info
            || swapchain == VK_NULL_HANDLE || instance_info->swapchains.empty())
        return false;
    try {
        const auto source = instance_info->swapchains.begin()->first;
        auto& installedVk = instance_info->swapchains.at(source).get();
        auto info = instance_info->swapchainInfos.at(source);
        info.virtualized = false;
        info.d2Foundation = false;
        info.d2State.reset();
        info.d2VulkanRuntime.reset();
        instance_info->swapchains.emplace(
            swapchain, ls::R<vk::Vulkan>(installedVk));
        instance_info->swapchainInfos.emplace(swapchain, std::move(info));
        return true;
    } catch (...) { return false; }
}

__attribute__((visibility("default")))
bool addInstalledD2SwapchainForTesting(VkSwapchainKHR swapchain) noexcept {
    if (!addInstalledNativeSwapchainForTesting(swapchain)) return false;
    try {
        auto& info = instance_info->swapchainInfos.at(swapchain);
        info.d2Foundation = true;
        info.d2State = std::make_shared<D2RealWsiState>(
            std::max<size_t>(1, info.realImages.size()));
        if (!info.d2State->acquired(0)) return false;
        queuePresentHarnessLastD2State = info.d2State;
        return true;
    } catch (...) {
        instance_info->swapchainInfos.erase(swapchain);
        instance_info->swapchains.erase(swapchain);
        return false;
    }
}

__attribute__((visibility("default")))
std::optional<D2PhysicalImageState> installedD2StateForTesting(
        VkSwapchainKHR swapchain, uint32_t index) noexcept {
    if (!instance_info) return std::nullopt;
    const auto found = instance_info->swapchainInfos.find(swapchain);
    if (found == instance_info->swapchainInfos.end()
            || !found->second.d2Foundation || !found->second.d2State)
        return std::nullopt;
    return found->second.d2State->state(index);
}

__attribute__((visibility("default")))
void setD2BatchReservationFailureForTesting(bool value) noexcept {
    queuePresentHarnessFailD2BatchReservation.store(
        value, std::memory_order_relaxed);
}

__attribute__((visibility("default")))
bool lastInstalledD2StateAliveForTesting() noexcept {
    return !queuePresentHarnessLastD2State.expired();
}

__attribute__((visibility("default")))
bool replaceInstalledRetirementReactorWithFreshForTesting(
        VkDevice device) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || !instance_info->deviceRetirementReactors.contains(device))
        return false;
    try {
        auto fresh = std::make_shared<DeviceRetirementReactor>();
        if (fresh->pendingCount() != 0) return false;
        instance_info->deviceRetirementReactors.at(device) = std::move(fresh);
        return instance_info->deviceRetirementReactors.at(device)
            ->pendingCount() == 0;
    } catch (...) { return false; }
}

__attribute__((visibility("default")))
void uninstallQueuePresentEntrypointHarness() noexcept {
    if (!queuePresentHarnessInstalled) return;
    delete instance_info;
    instance_info = nullptr;
    delete layer_info;
    layer_info = nullptr;
    queuePresentHarnessInstalled = false;
    queuePresentHarnessLastIdentity = {};
    queuePresentHarnessDeferredOwner.reset();
    queuePresentHarnessSeededDone.reset();
    queuePresentHarnessSeededBacking.reset();
    queuePresentHarnessSeededDestroys = 0;
    queuePresentHarnessWorkerInternalFailure.store(VK_SUCCESS);
    queuePresentHarnessFailD2Bookkeeping.store(false);
    queuePresentHarnessFailD2BatchReservation.store(false);
    queuePresentHarnessLastD2State.reset();
}

__attribute__((visibility("default")))
PFN_vkVoidFunction queuePresentEntrypoint() noexcept {
    return queuePresentHarnessInstalled
        ? reinterpret_cast<PFN_vkVoidFunction>(myvkQueuePresentKHR) : nullptr;
}

__attribute__((visibility("default")))
PFN_vkVoidFunction queuePresentAcquireEntrypoint() noexcept {
    return queuePresentHarnessInstalled
        ? reinterpret_cast<PFN_vkVoidFunction>(myvkAcquireNextImageKHR) : nullptr;
}

__attribute__((visibility("default")))
PFN_vkVoidFunction queuePresentDeviceEntrypoint(const char* name) noexcept {
    if (!queuePresentHarnessInstalled || !name) return nullptr;
#define QUEUE_TEST_ENTRYPOINT(vkname, impl) \
    if (std::strcmp(name, vkname) == 0) \
        return reinterpret_cast<PFN_vkVoidFunction>(impl)
    QUEUE_TEST_ENTRYPOINT("vkCreateFence", myvkCreateFence);
    QUEUE_TEST_ENTRYPOINT("vkDestroyFence", myvkDestroyFence);
    QUEUE_TEST_ENTRYPOINT("vkGetFenceStatus", myvkGetFenceStatus);
    QUEUE_TEST_ENTRYPOINT("vkWaitForFences", myvkWaitForFences);
    QUEUE_TEST_ENTRYPOINT("vkResetFences", myvkResetFences);
    QUEUE_TEST_ENTRYPOINT("vkImportFenceFdKHR", myvkImportFenceFdKHR);
    QUEUE_TEST_ENTRYPOINT("vkAcquireNextImage2KHR", myvkAcquireNextImage2KHR);
    QUEUE_TEST_ENTRYPOINT("vkCreateSwapchainKHR", myvkCreateSwapchainKHR);
    QUEUE_TEST_ENTRYPOINT("vkGetSwapchainImagesKHR", myvkGetSwapchainImagesKHR);
    QUEUE_TEST_ENTRYPOINT("vkReleaseSwapchainImagesKHR", myvkReleaseSwapchainImagesKHR);
    QUEUE_TEST_ENTRYPOINT("vkReleaseSwapchainImagesEXT", myvkReleaseSwapchainImagesEXT);
    QUEUE_TEST_ENTRYPOINT("vkDestroySwapchainKHR", myvkDestroySwapchainKHR);
    QUEUE_TEST_ENTRYPOINT("vkDestroyDevice", myvkDestroyDevice);
#undef QUEUE_TEST_ENTRYPOINT
    return nullptr;
}

__attribute__((visibility("default")))
void setQueuePresentWorkerInternalFailure(VkResult result) noexcept {
    queuePresentHarnessWorkerInternalFailure.store(result);
}

__attribute__((visibility("default")))
void setD2CreateBookkeepingFailure(bool value) noexcept {
    queuePresentHarnessFailD2Bookkeeping.store(value);
}

__attribute__((visibility("default")))
size_t rootSwapchainContextCount() noexcept {
    return queuePresentHarnessInstalled && layer_info
        ? layer_info->root.swapchainContextCountForTesting() : 0;
}

__attribute__((visibility("default")))
Adaptive1xPreparationReservation reserveInstalledAdaptive1xForTesting(
        VkFence applicationFence) {
    if (!queuePresentHarnessInstalled || !layer_info || !instance_info
            || instance_info->queues.empty() || instance_info->swapchains.empty())
        throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
            "Adaptive builder harness is unavailable");
    const auto& [queue, metadata] = *instance_info->queues.begin();
    const auto swapchain = instance_info->swapchains.begin()->first;
    const auto device = metadata.device;
    auto publication = std::make_shared<
        std::unique_ptr<VirtualPresentPendingOperation>>();
    return layer_info->root.getSwapchainContext(swapchain)
        .reserveAdaptive1xPreparation(instance_info->devices.at(device), queue,
            std::make_shared<std::mutex>(), swapchain, 0, applicationFence, {},
            std::make_shared<int>(1),
            instance_info->deviceRetirementReactors.at(device),
            instance_info->deviceBorrowedPresentFences.at(device),
            std::move(publication));
}

__attribute__((visibility("default")))
size_t installedPresentedLeaseCount() noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || instance_info->devicePresentedPhysicalImages.empty()) return 0;
    return instance_info->devicePresentedPhysicalImages.begin()->second->size();
}

__attribute__((visibility("default")))
PresentedPhysicalImageLeaseRegistry::PreparationState
installedPresentedLeasePreparationState(
        const PresentedPhysicalImageIdentity& identity) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || instance_info->devicePresentedPhysicalImages.empty())
        return PresentedPhysicalImageLeaseRegistry::PreparationState::Missing;
    return instance_info->devicePresentedPhysicalImages.begin()->second
        ->preparationState(identity);
}

__attribute__((visibility("default")))
BorrowedPresentFenceRegistry::PreparationState
installedBorrowedPreparationState(VkFence fence,
        const PresentedPhysicalImageIdentity& identity) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || instance_info->deviceBorrowedPresentFences.empty())
        return BorrowedPresentFenceRegistry::PreparationState::Missing;
    return instance_info->deviceBorrowedPresentFences.begin()->second
        ->preparationState(fence, identity);
}

__attribute__((visibility("default")))
size_t installedBorrowedAssociationCount() noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || instance_info->deviceBorrowedPresentFences.empty()) return 0;
    return instance_info->deviceBorrowedPresentFences.begin()->second
        ->associationCount();
}

__attribute__((visibility("default")))
SwapchainReleaseBackend setInstalledAdaptiveReleaseBackendForTesting(
        SwapchainReleaseBackend value) noexcept {
    if (!queuePresentHarnessInstalled || !layer_info || !instance_info
            || instance_info->swapchains.empty()) return SwapchainReleaseBackend::None;
    try {
        return layer_info->root.getSwapchainContext(
            instance_info->swapchains.begin()->first)
            .setReleaseBackendForTesting(value);
    } catch (...) { return SwapchainReleaseBackend::None; }
}

__attribute__((visibility("default")))
bool submitInstalledAdaptiveBridgeForTesting(uint32_t waitCount,
        const VkSemaphore* waits, uint32_t signalCount,
        const VkSemaphore* signals) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info
            || instance_info->queues.empty() || (!waits && waitCount)
            || (!signals && signalCount)) return false;
    try {
        const auto& [queue, metadata] = *instance_info->queues.begin();
        std::vector<VkPipelineStageFlags> stages(
            waitCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = waitCount,
            .pWaitSemaphores = waits,
            .pWaitDstStageMask = stages.empty() ? nullptr : stages.data(),
            .signalSemaphoreCount = signalCount,
            .pSignalSemaphores = signals};
        return instance_info->devices.at(metadata.device).df().QueueSubmit(
            queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    } catch (...) { return false; }
}

__attribute__((visibility("default")))
bool trackedSwapchain(VkSwapchainKHR swapchain) noexcept {
    return queuePresentHarnessInstalled && instance_info
        && instance_info->swapchainInfos.contains(swapchain)
        && instance_info->swapchains.contains(swapchain);
}

__attribute__((visibility("default")))
void removeQueueMetadata(VkQueue queue) noexcept {
    if (queuePresentHarnessInstalled && instance_info)
        instance_info->queues.erase(queue);
}

__attribute__((visibility("default")))
bool seedPendingRetirement(VkSwapchainKHR swapchain) noexcept {
    class ControlledPending final : public VirtualPresentPendingOperation {
    public:
        ControlledPending(std::shared_ptr<std::atomic_bool> done,
                std::shared_ptr<void> backing) : done(std::move(done)),
            backing(std::move(backing)) {}
        ~ControlledPending() override { ++queuePresentHarnessSeededDestroys; }
        [[nodiscard]] VirtualPresentCompletionStatus tryComplete() noexcept override {
            return done->load() ? VirtualPresentCompletionStatus::RETIRED
                                : VirtualPresentCompletionStatus::NOT_READY;
        }
    private:
        std::shared_ptr<std::atomic_bool> done;
        std::shared_ptr<void> backing;
    };
    if (!queuePresentHarnessInstalled || !instance_info) return false;
    const auto runtime = instance_info->virtualSwapchains.find(swapchain);
    if (runtime == instance_info->virtualSwapchains.end()) return false;
    queuePresentHarnessSeededDone = std::make_shared<std::atomic_bool>(false);
    auto backing = std::make_shared<uint8_t>(0);
    queuePresentHarnessSeededBacking = backing;
    queuePresentHarnessSeededDestroys = 0;
    return runtime->second->installPendingForTesting(
        UINT32_MAX, UINT64_MAX, std::make_unique<ControlledPending>(
            queuePresentHarnessSeededDone, std::move(backing)));
}

__attribute__((visibility("default")))
void completeSeededPendingRetirement() noexcept {
    if (queuePresentHarnessSeededDone)
        queuePresentHarnessSeededDone->store(true);
}

__attribute__((visibility("default")))
bool seededPendingBackingAlive() noexcept {
    return !queuePresentHarnessSeededBacking.expired();
}

__attribute__((visibility("default")))
uint32_t seededPendingDestroyCount() noexcept {
    return queuePresentHarnessSeededDestroys;
}

__attribute__((visibility("default")))
size_t deferredPendingCount(VkDevice) noexcept {
    return queuePresentHarnessDeferredOwner
        ? queuePresentHarnessDeferredOwner->pendingCount() : 0;
}

__attribute__((visibility("default")))
void notifyDeferredRetirement(VkDevice device) noexcept {
    if (!queuePresentHarnessInstalled || !instance_info) return;
    const auto owner = instance_info->deviceDeferredVirtualRetirements.find(device);
    if (owner != instance_info->deviceDeferredVirtualRetirements.end())
        owner->second->notifyDeviceRetirement();
}

__attribute__((visibility("default")))
PresentedPhysicalImageIdentity lastProductionPresentedIdentity() noexcept {
    return queuePresentHarnessLastIdentity;
}

__attribute__((visibility("default")))
bool productionPresentedLeaseInstalled() noexcept {
    if (!queuePresentHarnessInstalled || !queuePresentHarnessLastIdentity.valid())
        return false;
    const auto& identity = queuePresentHarnessLastIdentity;
    const auto swapchain = reinterpret_cast<VkSwapchainKHR>(
        identity.physicalSwapchainIdentity);
    const auto it = instance_info->swapchains.find(swapchain);
    if (it == instance_info->swapchains.end()) return false;
    return instance_info->devicePresentedPhysicalImages.at(it->second.get().dev())
        ->contains(identity);
}

__attribute__((visibility("default")))
bool maintenanceFeatureUsable(const VkDeviceCreateInfo& info) noexcept {
    VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR owned{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR};
    auto copy = info;
    return enableSwapchainMaintenanceFeature(copy, owned);
}

__attribute__((visibility("default")))
bool timelineFeatureUsable(const VkDeviceCreateInfo& info) noexcept {
    return timelineSemaphoreUsable(info);
}

}
#endif

/// Vulkan layer entrypoint
__attribute__((visibility("default")))
VkResult vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    // ensure loader compatibility
    if (!pVersionStruct
        || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT
        || pVersionStruct->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;

    // if the layer has already been initialized, skip
    if (layer_info) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
        pVersionStruct->pfnGetDeviceProcAddr = myvkGetDeviceProcAddr;
        pVersionStruct->pfnGetInstanceProcAddr = myvkGetInstanceProcAddr;
        return VK_SUCCESS;
    }

    // load the layer configuration
    try {
        layer_info = new LayerInfo { // NOLINT (memory management)
            .map = {
#define VKPTR(name) reinterpret_cast<PFN_vkVoidFunction>(name)
                { "vkCreateInstance", VKPTR(myvkCreateInstance) },
                { "vkCreateDevice", VKPTR(myvkCreateDevice) },
                { "vkDestroyDevice", VKPTR(myvkDestroyDevice) },
                { "vkDestroyInstance", VKPTR(myvkDestroyInstance) },
                { "vkGetDeviceQueue", VKPTR(myvkGetDeviceQueue) },
                { "vkGetDeviceQueue2", VKPTR(myvkGetDeviceQueue2) },
                { "vkCreateFence", VKPTR(myvkCreateFence) },
                { "vkDestroyFence", VKPTR(myvkDestroyFence) },
                { "vkGetFenceStatus", VKPTR(myvkGetFenceStatus) },
                { "vkWaitForFences", VKPTR(myvkWaitForFences) },
                { "vkResetFences", VKPTR(myvkResetFences) },
                { "vkImportFenceFdKHR", VKPTR(myvkImportFenceFdKHR) },
                { "vkCreateSwapchainKHR", VKPTR(myvkCreateSwapchainKHR) },
                { "vkGetSwapchainImagesKHR", VKPTR(myvkGetSwapchainImagesKHR) },
                { "vkAcquireNextImageKHR", VKPTR(myvkAcquireNextImageKHR) },
                { "vkAcquireNextImage2KHR", VKPTR(myvkAcquireNextImage2KHR) },
                { "vkReleaseSwapchainImagesKHR", VKPTR(myvkReleaseSwapchainImagesKHR) },
                { "vkReleaseSwapchainImagesEXT", VKPTR(myvkReleaseSwapchainImagesEXT) },
                { "vkQueuePresentKHR", VKPTR(myvkQueuePresentKHR) },
                { "vkDestroySwapchainKHR", VKPTR(myvkDestroySwapchainKHR) }
#undef VKPTR
            },
            .root = Root()
        };

        if (!layer_info->root.active()) { // skip inactive
            delete layer_info; // NOLINT (memory management)
            layer_info = nullptr;

            return VK_ERROR_INITIALIZATION_FAILED;
        }
    } catch (const std::exception& e) {
        std::cerr << "lsfg-vk: something went wrong during lsfg-vk layer initialization:\n";
        std::cerr << "- " << e.what() << '\n';

        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // emplace function pointers/version
    pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    pVersionStruct->pfnGetDeviceProcAddr = myvkGetDeviceProcAddr;
    pVersionStruct->pfnGetInstanceProcAddr = myvkGetInstanceProcAddr;
    return VK_SUCCESS;
}
