/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_production_seams.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace lsfgvk::layer {

enum class TerminalSubmitRole : uint8_t {
    ProducerSignal,
    D3B1TerminalConsumption,
    D3B2TerminalWait,
};

struct TerminalSubmitProvenance {
    TerminalSubmitRole role{};
    uint64_t contextIdentity{};
    uint64_t leaseIdentity{};
    VkSemaphore returnedForGraphics{VK_NULL_HANDLE};
};

class D3B3AReturnHandoffLease;
class ReturnedForGraphicsWaitAuthority;

/// Storage whose lifetime is tied to one logical VkDevice wrapper.  Accepted
/// submissions that cannot be proven retired after device loss transfer their
/// backing here; swapchain/context destruction therefore cannot release it.
class D3B3DeviceLifetimeQuarantine {
public:
    D3B3DeviceLifetimeQuarantine();
    D3B3DeviceLifetimeQuarantine(const D3B3DeviceLifetimeQuarantine&) = delete;
    D3B3DeviceLifetimeQuarantine& operator=(const D3B3DeviceLifetimeQuarantine&) = delete;
    ~D3B3DeviceLifetimeQuarantine();

    [[nodiscard]] uint64_t identity() const noexcept { return identityValue; }
    [[nodiscard]] size_t recordCount() const noexcept;
    [[nodiscard]] bool devicePoisoned() const noexcept {
        return poisoned.load(std::memory_order_acquire);
    }

    // A quarantine slot is reserved before a lease can participate in any GPU
    // submit.  Saturation therefore fails before QueueSubmit rather than on the
    // device-loss error path where allocation/ownership failure is unsafe.
    [[nodiscard]] bool reserve(uint64_t contextIdentity,
        uint64_t leaseIdentity) noexcept;
    void releaseReservation(uint64_t contextIdentity,
        uint64_t leaseIdentity) noexcept;

private:
    struct Record {
        uint64_t contextIdentity{};
        uint64_t leaseIdentity{};
        std::shared_ptr<void> backing;
    };
    [[nodiscard]] bool retain(uint64_t contextIdentity, uint64_t leaseIdentity,
        std::shared_ptr<void> backing) noexcept;
    void poison() noexcept { poisoned.store(true, std::memory_order_release); }
    static constexpr size_t MAX_QUARANTINED_BACKINGS = 4096;
    uint64_t identityValue{};
    std::atomic_bool poisoned{false};
    mutable std::mutex mutex;
    std::array<std::optional<Record>, MAX_QUARANTINED_BACKINGS> records{};
    size_t retainedRecordCount{};
    friend class D3B3PerSwapchainTerminalContext;
};

/// Stable, per-swapchain terminal resources.  This context owns the
/// returned-for-graphics semaphore lifecycle; callers only receive a
/// move-only lease and a borrowed handoff descriptor.
class D3B3PerSwapchainTerminalContext {
public:
    D3B3PerSwapchainTerminalContext(const vk::Vulkan&, uint32_t destinationQueueFamily,
        uint64_t lifecycleGeneration);
    D3B3PerSwapchainTerminalContext(const vk::Vulkan&, uint32_t destinationQueueFamily,
        uint64_t lifecycleGeneration,
        std::shared_ptr<D3B3DeviceLifetimeQuarantine>);
    D3B3PerSwapchainTerminalContext(const D3B3PerSwapchainTerminalContext&) = delete;
    D3B3PerSwapchainTerminalContext& operator=(const D3B3PerSwapchainTerminalContext&) = delete;
    ~D3B3PerSwapchainTerminalContext();

    [[nodiscard]] uint32_t destinationQueueFamily() const noexcept;
    [[nodiscard]] uint64_t lifecycleGeneration() const noexcept;
    [[nodiscard]] uint64_t contextIdentity() const noexcept;
    [[nodiscard]] uint64_t deviceIdentity() const noexcept;
    [[nodiscard]] bool retirementReady() const noexcept;
    [[nodiscard]] bool deviceLostQuarantined() const noexcept;
    void quarantineDeviceLost() noexcept;

private:
    struct State;
    std::shared_ptr<State> state;
    // Standalone contexts (primarily isolated/test users) do not have the
    // per-VkDevice owner map that keeps the quarantine alive in production.
    // Keep that default owner here, outside State, so quarantined backings do
    // not form a device-lifetime ownership cycle through State.
    std::shared_ptr<D3B3DeviceLifetimeQuarantine> standaloneDeviceLifetime;
    friend class D3B3AReturnHandoffBinding;
    friend class D3B3PerSwapchainTerminalContext;
    friend class D3B3AReturnHandoffLease;
};

class D3B3AReturnHandoffLease {
public:
    D3B3AReturnHandoffLease() noexcept = default;
    D3B3AReturnHandoffLease(const D3B3AReturnHandoffLease&) = delete;
    D3B3AReturnHandoffLease& operator=(const D3B3AReturnHandoffLease&) = delete;
    D3B3AReturnHandoffLease(D3B3AReturnHandoffLease&&) noexcept = default;
    D3B3AReturnHandoffLease& operator=(D3B3AReturnHandoffLease&&) noexcept = default;
    ~D3B3AReturnHandoffLease() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] uint64_t generation() const noexcept;
    [[nodiscard]] uint64_t leaseIdentity() const noexcept;
    [[nodiscard]] uint64_t contextIdentity() const noexcept;
    [[nodiscard]] TerminalSubmitProvenance provenance(TerminalSubmitRole) const;
    [[nodiscard]] ReturnedForGraphicsWaitAuthority deriveWaitAuthority();
    [[nodiscard]] vk::RuntimeForeignImageHandoffInfo handoffInfo() const;
    [[nodiscard]] VkSemaphore signalSemaphore() const;
    void signalSubmitted();
    void terminalWaitSubmitted();
    void terminalWaitRetired();

private:
    struct State;
    explicit D3B3AReturnHandoffLease(std::shared_ptr<State>) noexcept;
    std::shared_ptr<State> state;
    friend class D3B3AReturnHandoffBinding;
    friend class D3B3PerSwapchainTerminalContext;
    friend class ReturnedForGraphicsWaitAuthority;
};

class ReturnedForGraphicsWaitAuthority {
public:
    ReturnedForGraphicsWaitAuthority() noexcept = default;
    ReturnedForGraphicsWaitAuthority(const ReturnedForGraphicsWaitAuthority&) = delete;
    ReturnedForGraphicsWaitAuthority& operator=(const ReturnedForGraphicsWaitAuthority&) = delete;
    ReturnedForGraphicsWaitAuthority(ReturnedForGraphicsWaitAuthority&&) noexcept = default;
    ReturnedForGraphicsWaitAuthority& operator=(ReturnedForGraphicsWaitAuthority&&) noexcept = default;
    ~ReturnedForGraphicsWaitAuthority() = default;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] VkSemaphore semaphore() const;
    [[nodiscard]] uint64_t contextIdentity() const noexcept;
    [[nodiscard]] uint64_t leaseIdentity() const noexcept;
    [[nodiscard]] TerminalSubmitProvenance provenance(TerminalSubmitRole) const;
    void terminalWaitSubmitted();
    void terminalWaitRetired();
private:
    struct State;
    explicit ReturnedForGraphicsWaitAuthority(std::shared_ptr<State>) noexcept;
    std::shared_ptr<State> state;
    friend class D3B3AReturnHandoffLease;
    friend void submitReturnedForGraphicsTerminalWait(
        ReturnedForGraphicsWaitAuthority&, TerminalSubmitRole, uint64_t,
        const vk::CommandBuffer&, const vk::Vulkan&, VkQueue,
        std::vector<VkSemaphore>, std::vector<VkSemaphore>, VkFence,
        VkPipelineStageFlags, const vk::SubmitObserver&,
        const vk::SubmitResultObserver&);
};

class D3B3AReturnHandoffBinding {
public:
    explicit D3B3AReturnHandoffBinding(
        std::shared_ptr<D3B3PerSwapchainTerminalContext>);
    [[nodiscard]] D3B3AReturnHandoffLease acquire(uint64_t pairGeneration);
    [[nodiscard]] bool retirementReady() const noexcept;

private:
    std::shared_ptr<D3B3PerSwapchainTerminalContext> context;
};

void submitReturnedForGraphicsTerminalWait(ReturnedForGraphicsWaitAuthority&,
    TerminalSubmitRole, uint64_t expectedContextIdentity,
    const vk::CommandBuffer&, const vk::Vulkan&, VkQueue,
    std::vector<VkSemaphore> waitSemaphores,
    std::vector<VkSemaphore> signalSemaphores, VkFence,
    VkPipelineStageFlags,
    const vk::SubmitObserver& observer = {},
    const vk::SubmitResultObserver& resultObserver = {});

}
