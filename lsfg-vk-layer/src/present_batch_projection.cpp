/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "present_batch_projection.hpp"

#include <algorithm>
#include <utility>

namespace lsfgvk::layer {
namespace {

template <typename T>
[[nodiscard]] bool validRequiredPerSwapchainArray(const T* pointer,
        uint32_t count, uint32_t expected, uint32_t index) noexcept {
    return count == expected && index < count && pointer != nullptr;
}

[[nodiscard]] bool validOptionalPerSwapchainArray(
        uint32_t count, uint32_t expected, uint32_t index) noexcept {
    return count == expected && index < count;
}

} // namespace

void PresentPNextProjection::append(VkBaseOutStructure* node) noexcept {
    if (!node) return;
    node->pNext = nullptr;
    if (!headValue) headValue = node;
    if (tail) tail->pNext = node;
    tail = node;
}

PresentPNextProjection PresentPNextProjection::build(
        const void* pNext, uint32_t originalSwapchainCount,
        uint32_t index) noexcept {
    PresentPNextProjection result;
    if (!pNext) return result;
    if (originalSwapchainCount == 0 || index >= originalSwapchainCount) {
        result.supportedValue = false;
        return result;
    }

    try {
        auto* current = reinterpret_cast<const VkBaseInStructure*>(pNext);
        while (current) {
            switch (current->sType) {
#if defined(VK_KHR_device_group)
            case VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR: {
                const auto& input = *reinterpret_cast<const VkDeviceGroupPresentInfoKHR*>(current);
                auto projected = input;
                projected.pNext = nullptr;
                if (input.swapchainCount == 0) {
                    projected.swapchainCount = 0;
                    projected.pDeviceMasks = nullptr;
                } else {
                    if (!validRequiredPerSwapchainArray(input.pDeviceMasks,
                            input.swapchainCount, originalSwapchainCount, index)) {
                        result.supportedValue = false;
                        result.unsupportedType = current->sType;
                        return result;
                    }
                    auto* mask = result.store(input.pDeviceMasks[index]);
                    projected.swapchainCount = 1;
                    projected.pDeviceMasks = mask;
                }
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_KHR_display_swapchain)
            case VK_STRUCTURE_TYPE_DISPLAY_PRESENT_INFO_KHR: {
                if (originalSwapchainCount != 1) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto projected = *reinterpret_cast<const VkDisplayPresentInfoKHR*>(current);
                projected.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_KHR_present_id)
            case VK_STRUCTURE_TYPE_PRESENT_ID_KHR: {
                const auto& input = *reinterpret_cast<const VkPresentIdKHR*>(current);
                if (!validOptionalPerSwapchainArray(input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = input.pPresentIds ? result.store(input.pPresentIds[index]) : nullptr;
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pPresentIds = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_KHR_present_id2)
            case VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR: {
                const auto& input = *reinterpret_cast<const VkPresentId2KHR*>(current);
                if (!validOptionalPerSwapchainArray(input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = input.pPresentIds ? result.store(input.pPresentIds[index]) : nullptr;
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pPresentIds = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_KHR_incremental_present)
            case VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR: {
                const auto& input = *reinterpret_cast<const VkPresentRegionsKHR*>(current);
                if (!validOptionalPerSwapchainArray(input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = input.pRegions ? result.store(input.pRegions[index]) : nullptr;
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pRegions = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_GOOGLE_display_timing)
            case VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE: {
                const auto& input = *reinterpret_cast<const VkPresentTimesInfoGOOGLE*>(current);
                if (!validOptionalPerSwapchainArray(input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = input.pTimes ? result.store(input.pTimes[index]) : nullptr;
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pTimes = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_EXT_present_timing)
            case VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT: {
                const auto& input = *reinterpret_cast<const VkPresentTimingsInfoEXT*>(current);
                if (!validOptionalPerSwapchainArray(input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = input.pTimingInfos ? result.store(input.pTimingInfos[index]) : nullptr;
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pTimingInfos = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR: {
                const auto& input = *reinterpret_cast<const VkSwapchainPresentFenceInfoKHR*>(current);
                if (!validRequiredPerSwapchainArray(input.pFences, input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = result.store(input.pFences[index]);
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pFences = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
            case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR: {
                const auto& input = *reinterpret_cast<const VkSwapchainPresentModeInfoKHR*>(current);
                if (!validRequiredPerSwapchainArray(input.pPresentModes, input.swapchainCount,
                        originalSwapchainCount, index)) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto* value = result.store(input.pPresentModes[index]);
                auto projected = input;
                projected.pNext = nullptr;
                projected.swapchainCount = 1;
                projected.pPresentModes = value;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_EXT_frame_boundary)
            case VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT: {
                if (originalSwapchainCount != 1) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto projected = *reinterpret_cast<const VkFrameBoundaryEXT*>(current);
                projected.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_EXT_frame_boundary) && defined(VK_ARM_tensors)
            case VK_STRUCTURE_TYPE_FRAME_BOUNDARY_TENSORS_ARM: {
                if (originalSwapchainCount != 1) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto projected = *reinterpret_cast<const VkFrameBoundaryTensorsARM*>(current);
                projected.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_GGP_frame_token)
            case VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP: {
                if (originalSwapchainCount != 1) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto projected = *reinterpret_cast<const VkPresentFrameTokenGGP*>(current);
                projected.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
#if defined(VK_NV_present_metering)
            case VK_STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV: {
                if (originalSwapchainCount != 1) {
                    result.supportedValue = false;
                    result.unsupportedType = current->sType;
                    return result;
                }
                auto projected = *reinterpret_cast<const VkSetPresentConfigNV*>(current);
                projected.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(projected)));
                break;
            }
#endif
            default:
                result.supportedValue = false;
                result.unsupportedType = current->sType;
                return result;
            }
            current = current->pNext;
        }
    } catch (...) {
        result.supportedValue = false;
        result.headValue = nullptr;
        result.tail = nullptr;
    }
    return result;
}

VkResult aggregatePresentResults(const std::vector<VkResult>& results) noexcept {
    if (results.empty()) return VK_SUCCESS;
    const auto contains = [&](VkResult value) {
        return std::find(results.begin(), results.end(), value) != results.end();
    };
    if (contains(VK_ERROR_DEVICE_LOST)) return VK_ERROR_DEVICE_LOST;
    if (contains(VK_ERROR_SURFACE_LOST_KHR)) return VK_ERROR_SURFACE_LOST_KHR;
    if (contains(VK_ERROR_OUT_OF_DATE_KHR)) return VK_ERROR_OUT_OF_DATE_KHR;
#if defined(VK_EXT_present_timing)
    if (contains(VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT))
        return VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT;
#endif
#if defined(VK_EXT_full_screen_exclusive)
    if (contains(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT))
        return VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT;
#endif
    // A decomposed call can expose enqueue/resource failures that Vulkan's
    // per-swapchain precedence list does not enumerate. Preserve the first
    // fatal result rather than hiding it behind a later success/suboptimal.
    for (const auto value : results) {
        if (value < 0) return value;
    }
    if (contains(VK_SUBOPTIMAL_KHR)) return VK_SUBOPTIMAL_KHR;
    return VK_SUCCESS;
}

void PresentBatchPNextProjection::append(VkBaseOutStructure* node) noexcept {
    node->pNext = nullptr;
    if (!headValue) headValue = node;
    if (tail) tail->pNext = node;
    tail = node;
}

void PresentBatchPNextProjection::reject(VkStructureType type) noexcept {
    supportedValue = false;
    unsupportedType = type;
    headValue = nullptr;
    tail = nullptr;
    fenceNode = nullptr;
}

VkFence PresentBatchPNextProjection::applicationFence(uint32_t index) const noexcept {
    return index < applicationFences.size() ? applicationFences[index]
                                             : VK_NULL_HANDLE;
}

bool PresentBatchPNextProjection::patchInternalPresentFence(
        uint32_t index, VkFence fence) noexcept {
    if (!supportedValue || !fenceNode || index >= finalFences.size()
            || pathsValue[index] != Path::VirtualPrepared
            || applicationFences[index] != VK_NULL_HANDLE || fence == VK_NULL_HANDLE)
        return false;
    finalFences[index] = fence;
    return true;
}

PresentBatchPNextProjection PresentBatchPNextProjection::build(
        const void* pNext, uint32_t swapchainCount,
        const std::vector<Path>& paths) noexcept {
    PresentBatchPNextProjection result;
    if (!swapchainCount || paths.size() != swapchainCount) {
        result.supportedValue = false;
        return result;
    }
    try {
        result.pathsValue = paths;
        auto* current = reinterpret_cast<const VkBaseInStructure*>(pNext);
        while (current) {
            switch (current->sType) {
#if defined(VK_KHR_device_group)
            case VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR: {
                const auto& in = *reinterpret_cast<const VkDeviceGroupPresentInfoKHR*>(current);
                if ((in.swapchainCount != 0 && in.swapchainCount != swapchainCount)
                        || (in.swapchainCount && !in.pDeviceMasks)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_KHR_present_id)
            case VK_STRUCTURE_TYPE_PRESENT_ID_KHR: {
                const auto& in = *reinterpret_cast<const VkPresentIdKHR*>(current);
                if (in.swapchainCount != swapchainCount
                        || (in.swapchainCount && !in.pPresentIds)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_KHR_present_id2)
            case VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR: {
                const auto& in = *reinterpret_cast<const VkPresentId2KHR*>(current);
                if (in.swapchainCount != swapchainCount
                        || (in.swapchainCount && !in.pPresentIds)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_KHR_incremental_present)
            case VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR: {
                const auto& in = *reinterpret_cast<const VkPresentRegionsKHR*>(current);
                if (in.swapchainCount != swapchainCount
                        || (in.swapchainCount && !in.pRegions)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_GOOGLE_display_timing)
            case VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE: {
                const auto& in = *reinterpret_cast<const VkPresentTimesInfoGOOGLE*>(current);
                if (in.swapchainCount != swapchainCount
                        || (in.swapchainCount && !in.pTimes)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_EXT_present_timing)
            case VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT: {
                const auto& in = *reinterpret_cast<const VkPresentTimingsInfoEXT*>(current);
                if (in.swapchainCount != swapchainCount
                        || (in.swapchainCount && !in.pTimingInfos)) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR: {
                const auto& in = *reinterpret_cast<const VkSwapchainPresentFenceInfoKHR*>(current);
                if (in.swapchainCount != swapchainCount || !in.pFences) {
                    result.reject(current->sType); return result;
                }
                result.applicationFences.assign(in.pFences, in.pFences + swapchainCount);
                result.finalFences = result.applicationFences;
                auto out = in; out.pNext = nullptr; out.pFences = result.finalFences.data();
                result.fenceNode = result.store(out);
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.fenceNode));
                break;
            }
            case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR: {
                const auto& in = *reinterpret_cast<const VkSwapchainPresentModeInfoKHR*>(current);
                if (in.swapchainCount != swapchainCount || !in.pPresentModes) {
                    result.reject(current->sType); return result;
                }
                auto out = in; out.pNext = nullptr;
                result.append(reinterpret_cast<VkBaseOutStructure*>(result.store(out)));
                break;
            }
#endif
            default:
                result.reject(current->sType); return result;
            }
            current = current->pNext;
        }
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
        // Adaptive prepared entries require an internal WSI completion fence
        // even when the application supplied no fence structure. Own the dense
        // array here so caller storage is never changed or retained.
        if (!result.fenceNode && std::ranges::find(paths,
                Path::VirtualPrepared) != paths.end()) {
            result.applicationFences.assign(swapchainCount, VK_NULL_HANDLE);
            result.finalFences.assign(swapchainCount, VK_NULL_HANDLE);
            VkSwapchainPresentFenceInfoKHR out{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                .pNext = nullptr,
                .swapchainCount = swapchainCount,
                .pFences = result.finalFences.data()};
            result.fenceNode = result.store(out);
            result.append(reinterpret_cast<VkBaseOutStructure*>(
                result.fenceNode));
        }
#endif
    } catch (...) {
        result.supportedValue = false;
        result.headValue = nullptr;
        result.tail = nullptr;
        result.fenceNode = nullptr;
    }
    return result;
}

} // namespace lsfgvk::layer
