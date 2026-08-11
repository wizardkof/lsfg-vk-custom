/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "fixed_present_mode.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "swapchain.hpp"
#include "virtual_swapchain_image_spec.hpp"
#include "virtual_swapchain_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
        SharedInternallySynchronized
    };

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
                auto* current = reinterpret_cast<VkBaseOutStructure*>(
                    const_cast<void*>(info.pNext));
                while (current) {
                    if (current->sType
                            == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR) {
                        auto* features = reinterpret_cast<
                            VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR*>(current);
                        features->internallySynchronizedQueues = VK_TRUE;
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

    QueueReservation reserveOffloadGraphicsQueue(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
            PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2,
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

            // If the application already consumes the full graphics family,
            // split its final flags==0 logical queue into a separate internally
            // synchronized queue. The layer remaps the application's original
            // logical index to that same VkQueue, allowing the Fixed worker and
            // application to share it without external queue synchronization.
            if (requested == available
                    && queueInfo.queueCount > 0
                    && supportsInternallySynchronizedQueues(
                        physdev, funcs, getPhysicalDeviceFeatures2)) {
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

    // instance-wide info initialized at instance creation(s)
    struct InstanceInfo {
        std::vector<VkInstance> handles; // there may be several instances
        vk::VulkanInstanceFuncs funcs;

        std::unordered_map<VkDevice, vk::Vulkan> devices;
        std::unordered_map<VkDevice, OffloadQueueInfo> offloadQueues;
        std::unordered_map<VkSwapchainKHR, ls::R<vk::Vulkan>> swapchains;
        std::unordered_map<VkSwapchainKHR, SwapchainInfo> swapchainInfos;
        std::unordered_map<VkSwapchainKHR, std::unique_ptr<VirtualSwapchainRuntime>>
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

        try {
            VkInstanceCreateInfo newInfo = *info;
            layer_info->root.modifyInstanceCreateInfo(newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = vkCreateInstance(newInfo, alloc, instance);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateInstance() failed");
                }
            );

            if (!instance_info)
                instance_info = new InstanceInfo{ // NOLINT (memory management)
                    .funcs = vk::initVulkanInstanceFuncs(*instance,
                        layer_info->GetInstanceProcAddr, true),
                };

            instance_info->handles.push_back(*instance);

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

        // Fixed pacing needs a queue that is never exposed to the application.
        // Reserve one only when Fixed is active so Adaptive device creation stays unchanged.
        QueueReservation offloadReservation{};

        // create device
        try {
            VkDeviceCreateInfo newInfo = *info;
            if (layer_info->root.fixedMode()) {
                const auto getPhysicalDeviceFeatures2 = reinterpret_cast<
                    PFN_vkGetPhysicalDeviceFeatures2>(layer_info->GetInstanceProcAddr(
                        instance_info->handles.front(), "vkGetPhysicalDeviceFeatures2"));
                offloadReservation = reserveOffloadGraphicsQueue(
                    physdev, instance_info->funcs, getPhysicalDeviceFeatures2, newInfo);
                offloadReservation.apply(newInfo);
            }

            layer_info->root.modifyDeviceCreateInfo(newInfo,
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
                        true),
                    true, setLoaderData
                )
            );
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk initialization:\n";
            std::cerr << "- " << e.what() << '\n';
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
                        if (offloadReservation.shared())
                            std::cerr << "lsfg-vk: Fixed worker sharing internally synchronized graphics queue "
                                << offloadReservation.familyIndex << ':'
                                << offloadReservation.applicationQueueIndex << "\n";
                    }
                }
            }
        } else if (layer_info->root.fixedMode()) {
            std::cerr << "lsfg-vk: no dedicated or internally synchronized graphics queue "
                "is available for asynchronous fixed pacing; the current synchronous Fixed "
                "path will remain in use\n";
        }

        return VK_SUCCESS;
    }

    // destroy device
    void myvkDestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
        // A well-behaved application destroys swapchains first, but stop any
        // remaining workers defensively before their VkDevice/queue disappears.
        for (auto& [swapchain, runtime] : instance_info->virtualSwapchains) {
            const auto swapchainIt = instance_info->swapchains.find(swapchain);
            if (swapchainIt != instance_info->swapchains.end()
                    && swapchainIt->second.get().dev() == device)
                runtime->stop();
        }

        instance_info->offloadQueues.erase(device);

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
                return;
            }

            const auto it = instance_info->devices.find(device);
            if (it != instance_info->devices.end()) {
                it->second.df().GetDeviceQueue(
                    device, queueFamilyIndex, queueIndex, queue);
                return;
            }

            if (instance_info->funcs.GetDeviceProcAddr) {
                const auto next = reinterpret_cast<PFN_vkGetDeviceQueue>(
                    instance_info->funcs.GetDeviceProcAddr(device, "vkGetDeviceQueue"));
                if (next) {
                    next(device, queueFamilyIndex, queueIndex, queue);
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
                return;
            }

            if (instance_info->funcs.GetDeviceProcAddr) {
                const auto next = reinterpret_cast<PFN_vkGetDeviceQueue2>(
                    instance_info->funcs.GetDeviceProcAddr(device, "vkGetDeviceQueue2"));
                if (next) {
                    next(device, queueInfo, queue);
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
    VkResult myvkCreateSwapchainKHR(
            VkDevice device,
            const VkSwapchainCreateInfoKHR* info,
            const VkAllocationCallbacks* alloc,
            VkSwapchainKHR* swapchain) {
        const auto& it = instance_info->devices.find(device);
        if (it == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;

        try {
            // vkCreateSwapchainKHR retires oldSwapchain immediately. Retired
            // swapchains cannot acquire new real WSI images, so stop/join our
            // asynchronous worker before the driver sees oldSwapchain. Keep
            // the runtime and application-visible virtual VkImages alive until
            // vkDestroySwapchainKHR, but drain hidden queue work first.
            if (info && info->oldSwapchain != VK_NULL_HANDLE) {
                const auto oldRuntime =
                    instance_info->virtualSwapchains.find(info->oldSwapchain);
                if (oldRuntime != instance_info->virtualSwapchains.end()) {
                    oldRuntime->second->stop();
                    const auto idle = it->second.df().DeviceWaitIdle(device);
                    if (idle != VK_SUCCESS)
                        throw ls::vulkan_error(idle,
                            "vkDeviceWaitIdle() failed while retiring old swapchain");
                    std::cerr << "lsfg-vk: retired old Fixed presentation worker before swapchain recreation\n";
                }
            }

            layer_info->root.update(); // ensure config is up to date

            // create underlying real swapchain
            VkSwapchainCreateInfoKHR newInfo = *info;
            const VkPresentModeKHR applicationPresentMode = newInfo.presentMode;
            bool fixedAsyncPresentModeEligible{};
            layer_info->root.modifySwapchainCreateInfo(it->second, newInfo,
                [&, newInfo = &newInfo]() {
                    // Only the asynchronous Fixed path changes WSI mode. The
                    // Adaptive and synchronous 3B paths retain their existing
                    // FIFO behavior. Eligibility is checked before creation so
                    // an unavailable dedicated presentation queue cannot alter
                    // legacy swapchain semantics.
                    if (layer_info->root.fixedMode()) {
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

            std::unique_ptr<VirtualSwapchainRuntime> virtualRuntime;
            std::vector<VkImage> applicationImages = realImages;
            bool virtualized{};

            if (layer_info->root.fixedMode()) {
                const auto queueIt = instance_info->offloadQueues.find(device);
                const auto spec = makeVirtualSwapchainImageSpec(newInfo);

                if (fixedAsyncPresentModeEligible
                        && queueIt != instance_info->offloadQueues.end()
                        && spec.supported()) {
                    try {
                        virtualRuntime = std::make_unique<VirtualSwapchainRuntime>(
                            it->second,
                            queueIt->second.queue,
                            queueIt->second.mutex,
                            realImages.size(),
                            spec);
                        applicationImages = virtualRuntime->imageHandles();
                        virtualized = true;
                    } catch (const std::exception& e) {
                        std::cerr << "lsfg-vk: virtual swapchain setup failed after "
                            "Fixed WSI mode selection; falling back to synchronous 3B path:\n";
                        std::cerr << "- " << e.what() << '\n';
                    }
                } else if (!spec.supported()) {
                    std::cerr << "lsfg-vk: swapchain flags are not supported by the "
                        "virtual bridge; falling back to synchronous 3B path\n";
                } else if (!fixedAsyncPresentModeEligible) {
                    std::cerr << "lsfg-vk: asynchronous Fixed WSI prerequisites unavailable; "
                        "keeping FIFO synchronous 3B path\n";
                }
            }

            auto [infoIt, inserted] = instance_info->swapchainInfos.emplace(
                *swapchain,
                SwapchainInfo {
                    .images = std::move(applicationImages),
                    .realImages = std::move(realImages),
                    .format = newInfo.imageFormat,
                    .colorSpace = newInfo.imageColorSpace,
                    .extent = newInfo.imageExtent,
                    .presentMode = newInfo.presentMode,
                    .virtualized = virtualized
                });
            if (!inserted)
                throw ls::error("swapchain info already exists");

            auto& swapchainInfo = infoIt->second;

            try {
                layer_info->root.createSwapchainContext(
                    it->second, *swapchain, swapchainInfo);

                if (virtualRuntime) {
                    const auto& offload = instance_info->offloadQueues.at(device);
                    const auto workerQueue = offload.queue;
                    const auto workerMutex = offload.mutex;
                    const auto workerSwapchain = *swapchain;
                    virtualRuntime->startWorker(
                        [workerSwapchain, workerQueue, workerMutex, device](
                                uint32_t imageIndex,
                                VkSemaphore readySemaphore,
                                void* nextChain,
                                std::stop_token stopToken,
                                std::chrono::steady_clock::time_point sourcePresentTime) -> VkResult {
                            const auto deviceIt = instance_info->devices.find(device);
                            if (deviceIt == instance_info->devices.end())
                                return VK_ERROR_DEVICE_LOST;

                            try {
                                return layer_info->root.presentSwapchain(
                                    deviceIt->second,
                                    workerQueue,
                                    workerMutex,
                                    workerSwapchain,
                                    nextChain,
                                    imageIndex,
                                    { readySemaphore },
                                    stopToken,
                                    sourcePresentTime);
                            } catch (const ls::vulkan_error& e) {
                                if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                                    std::cerr << "lsfg-vk: asynchronous fixed presentation failed:\n";
                                    std::cerr << "- " << e.what() << '\n';
                                }
                                return e.error();
                            } catch (const std::exception& e) {
                                std::cerr << "lsfg-vk: asynchronous fixed presentation failed:\n";
                                std::cerr << "- " << e.what() << '\n';
                                return VK_ERROR_UNKNOWN;
                            }
                        });
                    std::cerr << "lsfg-vk: asynchronous Fixed presentation worker enabled\n";
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

            return res;
        } catch (const ls::vulkan_error& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain creation:\n";
            std::cerr << "- " << e.what() << '\n';
            return e.error();
        } catch (const std::exception& e) {
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
        return deviceIt->second.df().AcquireNextImageKHR(
            device, swapchain, timeout, semaphore, fence, imageIndex);
    }

    VkResult myvkAcquireNextImage2KHR(
            VkDevice device,
            const VkAcquireNextImageInfoKHR* info,
            uint32_t* imageIndex) {
        if (!info)
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
        return next(device, info, imageIndex);
    }

    VkResult myvkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        VkResult result = VK_SUCCESS;

        // ensure layer config is up to date
        bool reload{};
        try {
            reload = layer_info->root.update();
        } catch (const std::exception&) {
            reload = false; // ignore parse errors
        }

        if (reload) {
            try {
                for (const auto& [swapchain, vk] : instance_info->swapchains) {
                    auto& swapchainInfo = instance_info->swapchainInfos.at(swapchain);

                    // A virtual swapchain's VkImage handles were already returned
                    // to the application. Fixed -> Adaptive therefore requires a
                    // real swapchain recreation instead of destroying/replacing
                    // those images during hot reload.
                    if (swapchainInfo.virtualized && !layer_info->root.fixedMode()) {
                        std::cerr << "lsfg-vk: frame_generation_mode change from Fixed "
                            "requires swapchain recreation; keeping current Fixed context\n";
                        continue;
                    }

                    layer_info->root.recreateSwapchainContext(
                        vk, swapchain, swapchainInfo);
                }

                std::cerr << "lsfg-vk: updated lsfg-vk configuration\n";
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk configuration update:\n";
                std::cerr << "- " << e.what() << '\n';
            }
        }

        // Present each swapchain. A single virtual Fixed swapchain with no
        // pNext chain is the fast asynchronous path. pNext-bearing or batched
        // presents remain synchronous through the worker so caller-owned data
        // stays alive and legacy multi-swapchain behavior is preserved.
        bool allVirtual = info->swapchainCount > 0;
        for (size_t i = 0; i < info->swapchainCount; ++i) {
            if (!instance_info->virtualSwapchains.contains(info->pSwapchains[i])) {
                allVirtual = false;
                break;
            }
        }

        for (size_t i = 0; i < info->swapchainCount; i++) {
            const auto& swapchain = info->pSwapchains[i];

            const auto& it = instance_info->swapchains.find(swapchain);
            if (it == instance_info->swapchains.end())
                return VK_ERROR_INITIALIZATION_FAILED;

            std::vector<VkSemaphore> waitSemaphores;
            // For an all-virtual batch the dedicated queue establishes FIFO
            // ordering after the first bridge submit, so the application's
            // binary present semaphores are consumed exactly once.
            if (!allVirtual || i == 0) {
                waitSemaphores.reserve(info->waitSemaphoreCount);
                for (size_t j = 0; j < info->waitSemaphoreCount; j++)
                    waitSemaphores.push_back(info->pWaitSemaphores[j]);
            }

            auto virtualIt = instance_info->virtualSwapchains.find(swapchain);
            try {
                if (virtualIt != instance_info->virtualSwapchains.end()) {
                    const bool synchronous = info->pNext != nullptr
                        || info->swapchainCount != 1;
                    result = virtualIt->second->queuePresent(
                        queue,
                        info->pImageIndices[i],
                        waitSemaphores,
                        const_cast<void*>(info->pNext),
                        synchronous);
                } else {
                    result = layer_info->root.presentSwapchain(
                        it->second,
                        queue,
                        nullptr,
                        swapchain,
                        const_cast<void*>(info->pNext),
                        info->pImageIndices[i],
                        waitSemaphores);
                }
            } catch (const ls::vulkan_error& e) {
                if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                    std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                    std::cerr << "- " << e.what() << '\n';
                }
                result = e.error();
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                std::cerr << "- " << e.what() << '\n';
                result = VK_ERROR_UNKNOWN;
            }

            if (info->pResults)
                info->pResults[i] = result;
        }

        return result;
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
        if (runtime != instance_info->virtualSwapchains.end()) {
            runtime->second->stop();
            // The asynchronous worker can leave GPU work queued when a stop
            // request interrupts a bounded acquire/fence wait. Swapchain
            // destruction is rare, so conservatively drain the device before
            // destroying worker-owned synchronization and swapchain resources.
            const auto idle = it->second.df().DeviceWaitIdle(device);
            if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) {
                std::cerr << "lsfg-vk: vkDeviceWaitIdle() failed during virtual swapchain destruction: "
                    << idle << '\n';
            }
        }

        layer_info->root.removeSwapchainContext(swapchain);

        if (runtime != instance_info->virtualSwapchains.end())
            instance_info->virtualSwapchains.erase(runtime);

        instance_info->swapchainInfos.erase(swapchain);
        instance_info->swapchains.erase(swapchain);

        // destroy underlying real swapchain
        it->second.df().DestroySwapchainKHR(device, swapchain, alloc);
    }
}

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
                { "vkCreateSwapchainKHR", VKPTR(myvkCreateSwapchainKHR) },
                { "vkGetSwapchainImagesKHR", VKPTR(myvkGetSwapchainImagesKHR) },
                { "vkAcquireNextImageKHR", VKPTR(myvkAcquireNextImageKHR) },
                { "vkAcquireNextImage2KHR", VKPTR(myvkAcquireNextImage2KHR) },
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
