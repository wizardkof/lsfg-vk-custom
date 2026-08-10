/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
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
        std::shared_ptr<std::mutex> mutex{std::make_shared<std::mutex>()};
    };

    struct QueueReservation {
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        std::vector<float> priorities;
        std::optional<uint32_t> queueInfoIndex;
        uint32_t familyIndex{};
        uint32_t queueIndex{};

        [[nodiscard]] bool valid() const { return this->queueInfoIndex.has_value(); }

        void apply(VkDeviceCreateInfo& info) {
            if (!this->valid())
                return;
            info.queueCreateInfoCount = static_cast<uint32_t>(this->queueInfos.size());
            info.pQueueCreateInfos = this->queueInfos.data();
        }
    };

    QueueReservation reserveOffloadGraphicsQueue(
            VkPhysicalDevice physdev,
            const vk::VulkanInstanceFuncs& funcs,
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
            // vkGetDeviceQueue() is only valid for queues created with flags == 0.
            if (queueInfo.flags != 0)
                continue;

            const uint32_t available = families.at(*graphicsFamily).queueCount;
            uint32_t requested{};
            for (const auto& candidate : result.queueInfos) {
                if (candidate.queueFamilyIndex == *graphicsFamily)
                    requested += candidate.queueCount;
            }
            if (requested >= available)
                return QueueReservation{};

            result.priorities.reserve(queueInfo.queueCount + 1);
            if (queueInfo.pQueuePriorities) {
                result.priorities.assign(
                    queueInfo.pQueuePriorities,
                    queueInfo.pQueuePriorities + queueInfo.queueCount);
            } else {
                result.priorities.assign(queueInfo.queueCount, 1.0F);
            }
            result.priorities.push_back(1.0F);

            result.familyIndex = *graphicsFamily;
            result.queueIndex = queueInfo.queueCount;
            result.queueInfoIndex = static_cast<uint32_t>(i);
            queueInfo.queueCount++;
            queueInfo.pQueuePriorities = result.priorities.data();
            return result;
        }

        return QueueReservation{};
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
                offloadReservation = reserveOffloadGraphicsQueue(physdev, instance_info->funcs, newInfo);
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
                it->second.df().GetDeviceQueue(
                    *device,
                    offloadReservation.familyIndex,
                    offloadReservation.queueIndex,
                    &queue);

                if (queue != VK_NULL_HANDLE) {
                    auto res = setLoaderData(*device, queue);
                    if (res != VK_SUCCESS) {
                        std::cerr << "lsfg-vk: failed to attach loader data to fixed-pacing queue\n";
                    } else {
                        instance_info->offloadQueues.emplace(*device, OffloadQueueInfo {
                            .queue = queue,
                            .familyIndex = offloadReservation.familyIndex,
                            .queueIndex = offloadReservation.queueIndex
                        });
                    }
                }
            }
        } else if (layer_info->root.fixedMode()) {
            std::cerr << "lsfg-vk: no spare graphics queue is available for asynchronous fixed pacing; "
                "the current synchronous Fixed path will remain in use\n";
        }

        return VK_SUCCESS;
    }

    // destroy device
    void myvkDestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
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
            // retire old swapchain. Virtual images must outlive Root contexts,
            // but a retired WSI swapchain can no longer be acquired/presented.
            if (info->oldSwapchain) {
                layer_info->root.removeSwapchainContext(info->oldSwapchain);
                instance_info->virtualSwapchains.erase(info->oldSwapchain);
                instance_info->swapchainInfos.erase(info->oldSwapchain);
                instance_info->swapchains.erase(info->oldSwapchain);
            }

            layer_info->root.update(); // ensure config is up to date

            // create underlying real swapchain
            VkSwapchainCreateInfoKHR newInfo = *info;
            layer_info->root.modifySwapchainCreateInfo(it->second, newInfo,
                [=, newInfo = &newInfo]() {
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

                if (queueIt != instance_info->offloadQueues.end() && spec.supported()) {
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
                        std::cerr << "lsfg-vk: virtual swapchain setup failed; "
                            "falling back to synchronous 3B path:\n";
                        std::cerr << "- " << e.what() << '\n';
                    }
                } else if (!spec.supported()) {
                    std::cerr << "lsfg-vk: swapchain flags are not supported by the "
                        "virtual bridge; falling back to synchronous 3B path\n";
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
            } catch (...) {
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

                    layer_info->root.removeSwapchainContext(swapchain);
                    layer_info->root.createSwapchainContext(vk, swapchain, swapchainInfo);
                }

                std::cerr << "lsfg-vk: updated lsfg-vk configuration\n";
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk configuration update:\n";
                std::cerr << "- " << e.what() << '\n';
            }
        }

        // present each swapchain
        for (size_t i = 0; i < info->swapchainCount; i++) {
            const auto& swapchain = info->pSwapchains[i];

            const auto& it = instance_info->swapchains.find(swapchain);
            if (it == instance_info->swapchains.end())
                return VK_ERROR_INITIALIZATION_FAILED;

            auto virtualIt = instance_info->virtualSwapchains.find(swapchain);
            bool virtualPresentBegun{};
            try {
                std::vector<VkSemaphore> waitSemaphores;
                waitSemaphores.reserve(info->waitSemaphoreCount);

                for (size_t j = 0; j < info->waitSemaphoreCount; j++)
                    waitSemaphores.push_back(info->pWaitSemaphores[j]);

                if (virtualIt != instance_info->virtualSwapchains.end()) {
                    virtualPresentBegun =
                        virtualIt->second->beginPresent(info->pImageIndices[i]);
                    if (!virtualPresentBegun) {
                        result = VK_ERROR_OUT_OF_DATE_KHR;
                        if (info->pResults)
                            info->pResults[i] = result;
                        continue;
                    }
                }

                auto& context = layer_info->root.getSwapchainContext(swapchain);
                result = context.present(it->second,
                    queue, swapchain,
                    const_cast<void*>(info->pNext),
                    info->pImageIndices[i],
                    { waitSemaphores.begin(), waitSemaphores.end() }
                );
            } catch (const ls::vulkan_error& e) {
                if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                    std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                    std::cerr << "- " << e.what() << '\n';
                } // silently swallow out-of-date errors

                result = e.error();
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                std::cerr << "- " << e.what() << '\n';
                result = VK_ERROR_UNKNOWN;
            }

            if (virtualPresentBegun)
                virtualIt->second->completePresent(info->pImageIndices[i]);

            if (result != VK_SUCCESS && info->pResults)
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

        layer_info->root.removeSwapchainContext(swapchain);

        const auto runtime = instance_info->virtualSwapchains.find(swapchain);
        if (runtime != instance_info->virtualSwapchains.end()) {
            runtime->second->stop();
            instance_info->virtualSwapchains.erase(runtime);
        }

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
