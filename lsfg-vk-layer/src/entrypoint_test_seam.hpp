/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "adaptive_1x_preparation_reservation.hpp"
#include "presented_physical_image_lease.hpp"
#include "swapchain.hpp"

#include <memory>

namespace lsfgvk::layer::test {

enum class AdaptiveBuilderFailurePoint : uint8_t {
    None, AcquireBacking, ProducerBacking, PresentBacking,
    CompositeRecoveryBacking, ConsumedAcquireStorage,
    ReactorPendingOperation, PendingHolder, PreparedCallbacks,
    RecoveryCallbacks, PhysicalLeaseReserve, BorrowedFenceReserve,
    AfterBothRegistryReservations
};
enum class AdaptiveExecuteFailurePoint : uint8_t {
    None, PhysicalLeaseBind, BorrowedFenceBind
};

struct DeviceEntrypointHarnessConfig {
    VkDevice device{VK_NULL_HANDLE};
    vk::VulkanDeviceFuncs downstream{};
    uint64_t deviceLifetimeIdentity{};
};

struct QueuePresentEntrypointHarnessConfig {
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    uint32_t queueFamily{};
    uint32_t queueIndex{};
    VkQueueFlags queueFlags{VK_QUEUE_GRAPHICS_BIT};
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    vk::Vulkan* vulkan{};
    backend::Instance* backend{};
    vk::RuntimeDevicePair devicePair{};
    SwapchainInfo swapchainInfo{};
    uint64_t deviceLifetimeIdentity{};
    bool virtualized{};
    uint32_t frameGenerationMultiplier{1};
};

[[nodiscard]] bool installDeviceEntrypointHarness(
    DeviceEntrypointHarnessConfig) noexcept;
void uninstallDeviceEntrypointHarness() noexcept;
[[nodiscard]] PFN_vkVoidFunction deviceEntrypoint(const char*) noexcept;
[[nodiscard]] bool installPresentedLease(PresentedPhysicalImageIdentity,
    std::shared_ptr<void>) noexcept;
[[nodiscard]] bool associateBorrowedFence(
    VkFence, PresentedPhysicalImageIdentity) noexcept;
[[nodiscard]] bool containsPresentedLease(
    const PresentedPhysicalImageIdentity&) noexcept;
[[nodiscard]] size_t borrowedAssociationCount() noexcept;
[[nodiscard]] bool associateAcquireFenceProxy(
    VkFence, VkFence, std::shared_ptr<void>) noexcept;
[[nodiscard]] size_t acquireFenceProxyAssociationCount() noexcept;
[[nodiscard]] VkResult releaseD2PhysicalImageForTesting(
    VkDevice, VkSwapchainKHR, uint32_t) noexcept;
[[nodiscard]] bool installQueuePresentEntrypointHarness(
    QueuePresentEntrypointHarnessConfig) noexcept;
[[nodiscard]] bool addInstalledVirtualSwapchainForTesting(
    VkSwapchainKHR) noexcept;
[[nodiscard]] bool addInstalledNativeSwapchainForTesting(
    VkSwapchainKHR) noexcept;
[[nodiscard]] bool addInstalledD2SwapchainForTesting(
    VkSwapchainKHR) noexcept;
[[nodiscard]] std::optional<D2PhysicalImageState> installedD2StateForTesting(
    VkSwapchainKHR, uint32_t) noexcept;
void setD2BatchReservationFailureForTesting(bool) noexcept;
[[nodiscard]] bool lastInstalledD2StateAliveForTesting() noexcept;
[[nodiscard]] bool replaceInstalledRetirementReactorWithFreshForTesting(
    VkDevice) noexcept;
void uninstallQueuePresentEntrypointHarness() noexcept;
[[nodiscard]] PFN_vkVoidFunction queuePresentEntrypoint() noexcept;
[[nodiscard]] PFN_vkVoidFunction queuePresentAcquireEntrypoint() noexcept;
[[nodiscard]] PFN_vkVoidFunction queuePresentDeviceEntrypoint(const char*) noexcept;
void setQueuePresentWorkerInternalFailure(VkResult) noexcept;
void setD2CreateBookkeepingFailure(bool) noexcept;
[[nodiscard]] size_t rootSwapchainContextCount() noexcept;
[[nodiscard]] bool trackedSwapchain(VkSwapchainKHR) noexcept;
void removeQueueMetadata(VkQueue) noexcept;
[[nodiscard]] bool seedPendingRetirement(VkSwapchainKHR) noexcept;
void completeSeededPendingRetirement() noexcept;
[[nodiscard]] bool seededPendingBackingAlive() noexcept;
[[nodiscard]] uint32_t seededPendingDestroyCount() noexcept;
[[nodiscard]] size_t deferredPendingCount(VkDevice) noexcept;
void notifyDeferredRetirement(VkDevice) noexcept;
void setAdaptiveReservationExecuteHooks(
    void (*before)(), void (*after)()) noexcept;
void setAdaptiveBuilderFailurePoint(AdaptiveBuilderFailurePoint) noexcept;
void setAdaptiveExecuteFailurePoint(AdaptiveExecuteFailurePoint) noexcept;
[[nodiscard]] Adaptive1xPreparationReservation
reserveInstalledAdaptive1xForTesting(VkFence applicationFence = VK_NULL_HANDLE);
[[nodiscard]] size_t installedPresentedLeaseCount() noexcept;
[[nodiscard]] PresentedPhysicalImageLeaseRegistry::PreparationState
installedPresentedLeasePreparationState(
    const PresentedPhysicalImageIdentity&) noexcept;
[[nodiscard]] BorrowedPresentFenceRegistry::PreparationState
installedBorrowedPreparationState(
    VkFence, const PresentedPhysicalImageIdentity&) noexcept;
[[nodiscard]] size_t installedBorrowedAssociationCount() noexcept;
[[nodiscard]] SwapchainReleaseBackend setInstalledAdaptiveReleaseBackendForTesting(
    SwapchainReleaseBackend) noexcept;
void resetAdaptiveCommandPoolLifecycleCounts() noexcept;
[[nodiscard]] uint32_t adaptiveCommandPoolCreateCount() noexcept;
[[nodiscard]] uint32_t adaptiveCommandPoolDestroyCount() noexcept;
[[nodiscard]] bool waitForAdaptiveCommandPoolDestroyCount(
    uint32_t, uint32_t timeoutMilliseconds) noexcept;
[[nodiscard]] bool submitInstalledAdaptiveBridgeForTesting(
    uint32_t waitCount, const VkSemaphore* waits,
    uint32_t signalCount, const VkSemaphore* signals) noexcept;
[[nodiscard]] PresentedPhysicalImageIdentity lastProductionPresentedIdentity() noexcept;
[[nodiscard]] bool productionPresentedLeaseInstalled() noexcept;
[[nodiscard]] bool maintenanceFeatureUsable(
    const VkDeviceCreateInfo&) noexcept;
[[nodiscard]] bool timelineFeatureUsable(
    const VkDeviceCreateInfo&) noexcept;

}
