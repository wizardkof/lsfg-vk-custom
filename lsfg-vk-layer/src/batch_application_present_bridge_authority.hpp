/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {

/// Owns the single submit which consumes an application's batch waits and
/// signals one independently-backed semaphore for each virtual preparation.
class BatchApplicationPresentBridgeAuthority final {
public:
    BatchApplicationPresentBridgeAuthority() = default;
    enum class State : uint8_t {
        Prepared, Accepted, Retiring, Retired, Recovered,
        ConservativelyRetained
    };
    struct OwnedSemaphore final {
        VkSemaphore handle{};
        std::shared_ptr<void> backing;
    };
    struct OwnedFence final {
        VkFence handle{};
        std::shared_ptr<void> backing;
    };
    struct Operations final {
        std::function<std::optional<OwnedSemaphore>()> createSemaphore;
        std::function<std::optional<OwnedFence>()> createFence;
        std::function<VkResult(VkQueue, const VkSubmitInfo&, VkFence)> submit;
        std::function<VkResult(VkFence)> waitBridgeFence;
        std::function<bool(VkFence, std::shared_ptr<void>)> retireAsync;
        std::function<bool(std::shared_ptr<void>)> retainConservatively;
    };

    [[nodiscard]] static std::optional<BatchApplicationPresentBridgeAuthority>
    create(VkQueue, const VkSemaphore* waits, uint32_t waitCount,
        uint32_t virtualCount, std::shared_ptr<void> deviceLifetime,
        Operations) noexcept;

    BatchApplicationPresentBridgeAuthority(
        const BatchApplicationPresentBridgeAuthority&) = delete;
    BatchApplicationPresentBridgeAuthority& operator=(
        const BatchApplicationPresentBridgeAuthority&) = delete;
    BatchApplicationPresentBridgeAuthority(
        BatchApplicationPresentBridgeAuthority&&) noexcept;
    BatchApplicationPresentBridgeAuthority& operator=(
        BatchApplicationPresentBridgeAuthority&&) noexcept;
    ~BatchApplicationPresentBridgeAuthority() noexcept;

    [[nodiscard]] VkResult submit() noexcept;
    [[nodiscard]] bool markSignalConsumed(uint32_t) noexcept;
    [[nodiscard]] bool retireNormalAsync() noexcept;
    [[nodiscard]] VkResult recoverPartial() noexcept;
    void retainDeviceLost() noexcept;

    [[nodiscard]] State state() const noexcept { return stateValue; }
    [[nodiscard]] bool accepted() const noexcept { return acceptedValue; }
    [[nodiscard]] uint32_t signalCount() const noexcept;
    [[nodiscard]] VkSemaphore signal(uint32_t) const noexcept;
    [[nodiscard]] const std::shared_ptr<void>& signalBacking(uint32_t) const noexcept;
    [[nodiscard]] VkFence bridgeFence() const noexcept;

private:
    struct Lifetime final {
        std::shared_ptr<void> device;
        OwnedFence fence;
        std::vector<OwnedSemaphore> signals;
        std::vector<bool> consumed;
    };

    VkQueue queueValue{};
    std::vector<VkSemaphore> waitsValue;
    std::vector<VkPipelineStageFlags> stagesValue;
    std::vector<VkSemaphore> signalHandles;
    std::shared_ptr<Lifetime> lifetime;
    Operations ops;
    State stateValue{State::Prepared};
    bool acceptedValue{};
};

} // namespace lsfgvk::layer
