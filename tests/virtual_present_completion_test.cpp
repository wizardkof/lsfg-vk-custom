/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "../lsfg-vk-layer/src/virtual_swapchain_runtime.hpp"

#include <cassert>
#include <atomic>
#include <cstdlib>
#include <new>
#include <memory>
#include <utility>
#include <vector>

using namespace lsfgvk::layer;

static std::atomic_bool failAllocation{};
void* operator new(std::size_t size) {
    if (failAllocation.load()) throw std::bad_alloc{};
    if (auto* value = std::malloc(size)) return value;
    throw std::bad_alloc{};
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

namespace {
class ImmediateCompletion final : public VirtualPresentPendingOperation {
public:
    VirtualPresentCompletionStatus tryComplete() noexcept override {
        return VirtualPresentCompletionStatus::RETIRED;
    }
};
class ScriptedCompletion final : public VirtualPresentPendingOperation {
public:
    ScriptedCompletion(std::vector<VirtualPresentCompletionStatus> values,
            uint32_t& calls, uint32_t& destroys) : script(std::move(values)),
        callCount(&calls), destroyCount(&destroys) {}
    ~ScriptedCompletion() override { ++*destroyCount; }
    VirtualPresentCompletionStatus tryComplete() noexcept override {
        ++*callCount;
        if (cursor >= script.size()) return script.back();
        return script[cursor++];
    }
private:
    std::vector<VirtualPresentCompletionStatus> script;
    size_t cursor{};
    uint32_t* callCount{};
    uint32_t* destroyCount{};
};

class DetachedCompletion final : public VirtualPresentPendingOperation {
public:
    DetachedCompletion(std::shared_ptr<void> retained,
            std::shared_ptr<std::atomic_bool> completed,
            uint64_t lifecycle, uint32_t& redirects,
            uint32_t* destroys = nullptr) : backing(std::move(retained)),
        done(std::move(completed)), lifecycleIdentity(lifecycle),
        redirectCount(&redirects), destroyCount(destroys) {}
    ~DetachedCompletion() override {
        if (destroyCount) ++*destroyCount;
    }
    VirtualPresentCompletionStatus tryComplete() noexcept override {
        return done->load() ? VirtualPresentCompletionStatus::RETIRED
                            : VirtualPresentCompletionStatus::NOT_READY;
    }
    void redirectCompletionWake(
            const std::shared_ptr<DeviceRetirementWakeTarget>&) noexcept override {
        ++*redirectCount;
    }
    uint64_t swapchainLifecycleIdentity() const noexcept override {
        return lifecycleIdentity;
    }
private:
    std::shared_ptr<void> backing;
    std::shared_ptr<std::atomic_bool> done;
    uint64_t lifecycleIdentity{};
    uint32_t* redirectCount{};
    uint32_t* destroyCount{};
};

uint32_t makePresenting(VirtualSwapchainState& state, uint64_t serial) {
    const auto image = state.tryAcquire();
    assert(image && state.queuePresent(*image, serial));
    const auto present = state.waitPresent(std::chrono::nanoseconds(1));
    assert(present && present->imageIndex == *image && present->serial == serial);
    return *image;
}

void pendingThenRetiredIsExactOnce() {
    VirtualSwapchainState state(1);
    VirtualPresentPendingSet pending;
    uint32_t calls{}, destroys{};
    const auto image = makePresenting(state, 41);
    assert(pending.install(image, 41, std::make_unique<ScriptedCompletion>(
        std::vector{VirtualPresentCompletionStatus::NOT_READY,
                    VirtualPresentCompletionStatus::RETIRED}, calls, destroys)));
    uint32_t rejectedCalls{}, rejectedDestroys{};
    assert(!pending.install(image, 42, std::make_unique<ScriptedCompletion>(
        std::vector{VirtualPresentCompletionStatus::RETIRED},
        rejectedCalls, rejectedDestroys)));
    assert(rejectedDestroys == 1 && pending.size() == 1);

    assert(pending.tryCompleteOnce(state) == VirtualPresentCompletionStatus::NOT_READY);
    assert(calls == 1 && !state.tryAcquire() && pending.contains(image));
    assert(pending.tryCompleteOnce(state) == VirtualPresentCompletionStatus::RETIRED);
    assert(calls == 2 && destroys == 1 && pending.empty());
    assert(state.tryAcquire().has_value());
    assert(pending.tryCompleteOnce(state) == VirtualPresentCompletionStatus::NOT_READY);
    assert(calls == 2 && destroys == 1);
}

void deviceLostNeverCompletesImage() {
    VirtualSwapchainState state(1);
    uint32_t calls{}, destroys{};
    {
        VirtualPresentPendingSet pending;
        const auto image = makePresenting(state, 51);
        assert(pending.install(image, 51, std::make_unique<ScriptedCompletion>(
            std::vector{VirtualPresentCompletionStatus::DEVICE_LOST}, calls, destroys)));
        assert(pending.tryCompleteOnce(state)
            == VirtualPresentCompletionStatus::DEVICE_LOST);
        assert(calls == 1 && pending.contains(image));
        assert(!state.tryAcquire().has_value());
    }
    assert(destroys == 1 && !state.tryAcquire().has_value());
}

void pendingImageDoesNotCorruptSecondImage() {
    VirtualSwapchainState state(2);
    VirtualPresentPendingSet pending;
    uint32_t callsA{}, destroysA{}, callsB{}, destroysB{};
    const auto imageA = makePresenting(state, 61);
    const auto imageB = makePresenting(state, 62);
    assert(imageA != imageB);
    assert(pending.install(imageA, 61, std::make_unique<ScriptedCompletion>(
        std::vector{VirtualPresentCompletionStatus::NOT_READY}, callsA, destroysA)));
    assert(pending.install(imageB, 62, std::make_unique<ScriptedCompletion>(
        std::vector{VirtualPresentCompletionStatus::RETIRED}, callsB, destroysB)));
    assert(pending.tryCompleteOnce(state) == VirtualPresentCompletionStatus::RETIRED);
    assert(callsA == 1 && callsB == 1 && pending.contains(imageA));
    assert(!pending.contains(imageB) && destroysB == 1);
    const auto reusable = state.tryAcquire();
    assert(reusable && *reusable == imageB);
}

void typedOutcomeOwnsPendingAuthority() {
    uint32_t calls{}, destroys{};
    auto outcome = PresentExecutionResult::pendingCompletion(VK_SUCCESS,
        std::make_unique<ScriptedCompletion>(
            std::vector{VirtualPresentCompletionStatus::NOT_READY}, calls, destroys),
        PresentResultOrigin::INTERNAL);
    assert(outcome.status == PresentExecutionStatus::PENDING);
    assert(outcome.result == VK_SUCCESS && outcome.pending);
    auto moved = std::move(outcome);
    assert(!outcome.pending && moved.pending);
}

void publicPresentResultFirewall() {
    assert(publicPresentResult(VK_SUCCESS, PresentResultOrigin::INTERNAL,
        PresentTransactionPhase::POST_COMMIT) == VK_SUCCESS);
    assert(publicPrePresentGateResult(PrePresentGateResult::TIMEOUT)
        == VK_ERROR_DEVICE_LOST);
    assert(publicPrePresentGateResult(PrePresentGateResult::TIMEOUT) != VK_TIMEOUT);
    assert(publicPrePresentGateResult(PrePresentGateResult::FAILED)
        == VK_ERROR_DEVICE_LOST);
    assert(publicPrePresentGateResult(PrePresentGateResult::FAILED)
        != VK_ERROR_INITIALIZATION_FAILED);
    assert(publicPresentResult(VK_ERROR_FEATURE_NOT_PRESENT,
        PresentResultOrigin::INTERNAL, PresentTransactionPhase::PRE_COMMIT)
        == VK_ERROR_DEVICE_LOST);
    assert(publicBridgePresentResult(VK_ERROR_DEVICE_LOST)
        == VK_ERROR_DEVICE_LOST);
    assert(publicBridgePresentResult(VK_ERROR_OUT_OF_HOST_MEMORY)
        == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(publicBridgePresentResult(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(publicBridgePresentResult(VK_TIMEOUT) == VK_ERROR_DEVICE_LOST);
    assert(publicPresentResult(VK_ERROR_OUT_OF_HOST_MEMORY,
        PresentResultOrigin::INTERNAL, PresentTransactionPhase::POST_COMMIT)
        == VK_ERROR_DEVICE_LOST);
    assert(publicPresentResult(VK_ERROR_OUT_OF_HOST_MEMORY,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_ERROR_DEVICE_LOST);
    assert(publicPresentResult(VK_ERROR_OUT_OF_DEVICE_MEMORY,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_ERROR_DEVICE_LOST);
    assert(publicPresentResult(VK_ERROR_OUT_OF_DATE_KHR,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_ERROR_OUT_OF_DATE_KHR);
    assert(publicPresentResult(VK_SUBOPTIMAL_KHR,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_SUBOPTIMAL_KHR);
    assert(publicPresentResult(VK_NOT_READY,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_ERROR_DEVICE_LOST);
    assert(publicPresentResult(VK_TIMEOUT,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT,
        PresentTransactionPhase::POST_COMMIT) == VK_ERROR_DEVICE_LOST);
    assert(!allowedPublicPresentResult(VK_NOT_READY));
    assert(!allowedPublicPresentResult(VK_TIMEOUT));
    assert(!allowedPublicPresentResult(VK_ERROR_INITIALIZATION_FAILED));
    assert(!allowedPublicPresentResult(VK_ERROR_FEATURE_NOT_PRESENT));
    assert(allowedPublicPresentResult(VK_ERROR_OUT_OF_DATE_KHR));
    assert(allowedPublicPresentResult(VK_SUBOPTIMAL_KHR));

    const auto internal = PresentExecutionResult::failed(VK_ERROR_UNKNOWN);
    assert(internal.origin == PresentResultOrigin::INTERNAL);
    const auto downstream = PresentExecutionResult::completed(
        VK_ERROR_OUT_OF_DATE_KHR,
        PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT);
    assert(downstream.origin == PresentResultOrigin::DOWNSTREAM_LOGICAL_PRESENT);
}

void detachedBackingOutlivesControlPlaneAndSharesImage() {
    auto owner = std::make_shared<DeferredVirtualRetirementOwner>();
    auto image = std::make_shared<int>(7);
    const std::weak_ptr<int> lifetime = image;
    auto completedA = std::make_shared<std::atomic_bool>(false);
    auto completedB = std::make_shared<std::atomic_bool>(false);
    uint32_t redirects{};
    VirtualPresentPendingSet source;
    assert(source.install(0, 101, std::make_unique<DetachedCompletion>(
        image, completedA, 101, redirects)));
    assert(source.install(1, 101, std::make_unique<DetachedCompletion>(
        image, completedB, 101, redirects)));
    image.reset();
    owner->adopt(source);
    assert(source.empty());
    assert(redirects == 2 && owner->pendingCount() == 2 && !lifetime.expired());
    completedA->store(true);
    owner->notifyDeviceRetirement();
    assert(owner->pendingCount() == 1 && !lifetime.expired());
    completedB->store(true);
    owner->notifyDeviceRetirement();
    assert(owner->pendingCount() == 0 && lifetime.expired());
}

void infallibleAdoptionHasNoHistoricalCapacityLimit() {
    auto owner = std::make_shared<DeferredVirtualRetirementOwner>();
    VirtualPresentPendingSet source;
    auto done = std::make_shared<std::atomic_bool>(false);
    uint32_t redirects{};
    constexpr uint32_t count = 4097;
    for (uint32_t i = 0; i < count; ++i)
        assert(source.install(i, i + 1,
            std::make_unique<DetachedCompletion>(
                std::make_shared<int>(static_cast<int>(i)), done, i + 1, redirects)));
    owner->adopt(source);
    assert(source.empty());
    assert(owner->pendingCount() == count);
    assert(redirects == count);
    done->store(true);
    owner->notifyDeviceRetirement();
    assert(owner->pendingCount() == 0);
}

void waitIdleCannotSubstituteForExactCompletion() {
    auto owner = std::make_shared<DeferredVirtualRetirementOwner>();
    auto done = std::make_shared<std::atomic_bool>(false);
    uint32_t redirects{}, destroys{};
    VirtualPresentPendingSet source;
    auto retained = std::make_shared<int>(9);
    const std::weak_ptr<int> lifetime = retained;
    assert(source.install(0, 900, std::make_unique<DetachedCompletion>(
        retained, done, 900, redirects, &destroys)));
    retained.reset();
    owner->adopt(source);
    assert(source.empty());
    assert(owner->pendingCount() == 1);
    assert(redirects == 1);

    // A successful or failed DeviceWaitIdle is irrelevant to presentation
    // authority. Only its exact completion mechanism may retire it.
    owner->notifyDeviceRetirement();
    assert(owner->pendingCount() == 1);
    assert(destroys == 0);
    assert(!lifetime.expired());
    done->store(true);
    owner->notifyDeviceRetirement();
    assert(owner->pendingCount() == 0);
    assert(destroys == 1);
    assert(lifetime.expired());
}

}

int main() {
    {
        VirtualPresentPendingSet pending;
        auto reserved = pending.reserveInstall(0, 77);
        assert(reserved && reserved->valid());
        auto operation = std::make_unique<ImmediateCompletion>();
        failAllocation.store(true);
        assert(pending.installReserved(*reserved, std::move(operation)));
        failAllocation.store(false);
        assert(pending.contains(0));
    }
    pendingThenRetiredIsExactOnce();
    deviceLostNeverCompletesImage();
    pendingImageDoesNotCorruptSecondImage();
    typedOutcomeOwnsPendingAuthority();
    publicPresentResultFirewall();
    detachedBackingOutlivesControlPlaneAndSharesImage();
    infallibleAdoptionHasNoHistoricalCapacityLimit();
    waitIdleCannotSubstituteForExactCompletion();
    return 0;
}
