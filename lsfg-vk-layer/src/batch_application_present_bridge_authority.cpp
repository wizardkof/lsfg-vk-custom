/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "batch_application_present_bridge_authority.hpp"

#include <utility>

namespace lsfgvk::layer {

BatchApplicationPresentBridgeAuthority::BatchApplicationPresentBridgeAuthority(
        BatchApplicationPresentBridgeAuthority&& other) noexcept
    : queueValue(other.queueValue), waitsValue(std::move(other.waitsValue)),
      stagesValue(std::move(other.stagesValue)),
      signalHandles(std::move(other.signalHandles)),
      lifetime(std::move(other.lifetime)), ops(std::move(other.ops)),
      stateValue(other.stateValue), acceptedValue(other.acceptedValue) {
    other.acceptedValue = false;
}

BatchApplicationPresentBridgeAuthority&
BatchApplicationPresentBridgeAuthority::operator=(
        BatchApplicationPresentBridgeAuthority&& other) noexcept {
    if (this == &other) return *this;
    if (acceptedValue && lifetime) retainDeviceLost();
    queueValue = other.queueValue;
    waitsValue = std::move(other.waitsValue);
    stagesValue = std::move(other.stagesValue);
    signalHandles = std::move(other.signalHandles);
    lifetime = std::move(other.lifetime);
    ops = std::move(other.ops);
    stateValue = other.stateValue;
    acceptedValue = other.acceptedValue;
    other.acceptedValue = false;
    return *this;
}

BatchApplicationPresentBridgeAuthority::
~BatchApplicationPresentBridgeAuthority() noexcept {
    // An accepted submit is never rolled back. If its explicit retirement path
    // was not selected, preserve all GPU-visible objects conservatively.
    if (acceptedValue && lifetime) retainDeviceLost();
}

std::optional<BatchApplicationPresentBridgeAuthority>
BatchApplicationPresentBridgeAuthority::create(VkQueue queue,
        const VkSemaphore* waits, uint32_t waitCount, uint32_t virtualCount,
        std::shared_ptr<void> deviceLifetime, Operations operations) noexcept {
    if (!queue || !virtualCount || (waitCount && !waits) || !deviceLifetime
            || !operations.createSemaphore || !operations.createFence
            || !operations.submit || !operations.waitBridgeFence
            || !operations.retireAsync || !operations.retainConservatively)
        return std::nullopt;
    try {
        BatchApplicationPresentBridgeAuthority result;
        result.queueValue = queue;
        result.ops = std::move(operations);
        if (waitCount) result.waitsValue.assign(waits, waits + waitCount);
        result.stagesValue.assign(waitCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        result.signalHandles.reserve(virtualCount);
        result.lifetime = std::make_shared<Lifetime>();
        result.lifetime->device = std::move(deviceLifetime);
        result.lifetime->signals.reserve(virtualCount);
        result.lifetime->consumed.assign(virtualCount, false);
        auto fence = result.ops.createFence();
        if (!fence || !fence->handle || !fence->backing) return std::nullopt;
        result.lifetime->fence = std::move(*fence);
        for (uint32_t i = 0; i < virtualCount; ++i) {
            auto semaphore = result.ops.createSemaphore();
            if (!semaphore || !semaphore->handle || !semaphore->backing)
                return std::nullopt;
            result.signalHandles.push_back(semaphore->handle);
            result.lifetime->signals.push_back(std::move(*semaphore));
        }
        return std::optional<BatchApplicationPresentBridgeAuthority>(
            std::move(result));
    } catch (...) {
        return std::nullopt;
    }
}

VkResult BatchApplicationPresentBridgeAuthority::submit() noexcept {
    if (stateValue != State::Prepared || !lifetime) return VK_ERROR_INITIALIZATION_FAILED;
    const VkSubmitInfo info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = static_cast<uint32_t>(waitsValue.size()),
        .pWaitSemaphores = waitsValue.empty() ? nullptr : waitsValue.data(),
        .pWaitDstStageMask = stagesValue.empty() ? nullptr : stagesValue.data(),
        .signalSemaphoreCount = static_cast<uint32_t>(signalHandles.size()),
        .pSignalSemaphores = signalHandles.data()};
    const auto result = ops.submit(queueValue, info, lifetime->fence.handle);
    if (result == VK_SUCCESS) {
        acceptedValue = true;
        stateValue = State::Accepted;
    } else if (result == VK_ERROR_DEVICE_LOST) {
        retainDeviceLost();
    }
    return result;
}

bool BatchApplicationPresentBridgeAuthority::markSignalConsumed(uint32_t index) noexcept {
    if (!acceptedValue || !lifetime || index >= lifetime->consumed.size()) return false;
    lifetime->consumed[index] = true;
    return true;
}

bool BatchApplicationPresentBridgeAuthority::retireNormalAsync() noexcept {
    if (stateValue != State::Accepted || !lifetime) return false;
    if (!ops.retireAsync(lifetime->fence.handle, lifetime)) return false;
    stateValue = State::Retiring;
    lifetime.reset();
    return true;
}

VkResult BatchApplicationPresentBridgeAuthority::recoverPartial() noexcept {
    if (stateValue != State::Accepted || !lifetime)
        return VK_ERROR_INITIALIZATION_FAILED;
    const auto result = ops.waitBridgeFence(lifetime->fence.handle);
    if (result == VK_SUCCESS) {
        lifetime.reset();
        stateValue = State::Recovered;
    } else {
        retainDeviceLost();
    }
    return result;
}

void BatchApplicationPresentBridgeAuthority::retainDeviceLost() noexcept {
    if (lifetime) {
        static_cast<void>(ops.retainConservatively(lifetime));
        lifetime.reset();
    }
    stateValue = State::ConservativelyRetained;
}

uint32_t BatchApplicationPresentBridgeAuthority::signalCount() const noexcept {
    return static_cast<uint32_t>(signalHandles.size());
}
VkSemaphore BatchApplicationPresentBridgeAuthority::signal(uint32_t index) const noexcept {
    return index < signalHandles.size() ? signalHandles[index] : VK_NULL_HANDLE;
}
const std::shared_ptr<void>& BatchApplicationPresentBridgeAuthority::signalBacking(
        uint32_t index) const noexcept {
    static const std::shared_ptr<void> empty;
    return lifetime && index < lifetime->signals.size()
        ? lifetime->signals[index].backing : empty;
}
VkFence BatchApplicationPresentBridgeAuthority::bridgeFence() const noexcept {
    return lifetime ? lifetime->fence.handle : VK_NULL_HANDLE;
}

} // namespace lsfgvk::layer
