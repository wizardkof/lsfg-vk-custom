#include "batch_application_present_bridge_authority.hpp"

#include <cassert>
#include <type_traits>

using namespace lsfgvk::layer;
template <typename T> T h(uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}

int main() {
    uint32_t nextSemaphore = 10;
    uint32_t submits = 0;
    uint32_t waits = 0;
    uint32_t asyncRetires = 0;
    std::shared_ptr<void> retained;
    VkResult submitResult = VK_SUCCESS;
    uint32_t expectedWaitCount = 2;
    auto operations = [&] {
        return BatchApplicationPresentBridgeAuthority::Operations{
            .createSemaphore = [&]() -> std::optional<BatchApplicationPresentBridgeAuthority::OwnedSemaphore> {
                auto backing = std::make_shared<uint32_t>(nextSemaphore++);
                return BatchApplicationPresentBridgeAuthority::OwnedSemaphore{
                    h<VkSemaphore>(*backing), backing};
            },
            .createFence = []() -> std::optional<BatchApplicationPresentBridgeAuthority::OwnedFence> {
                auto backing = std::make_shared<uint32_t>(50);
                return BatchApplicationPresentBridgeAuthority::OwnedFence{
                    h<VkFence>(*backing), backing};
            },
            .submit = [&](VkQueue queue, const VkSubmitInfo& info, VkFence fence) {
                ++submits;
                assert(queue == h<VkQueue>(2));
                assert(fence == h<VkFence>(50));
                assert(info.waitSemaphoreCount == expectedWaitCount);
                assert(info.signalSemaphoreCount == 2);
                assert(info.pSignalSemaphores[0] != info.pSignalSemaphores[1]);
                return submitResult;
            },
            .waitBridgeFence = [&](VkFence fence) {
                ++waits; assert(fence == h<VkFence>(50)); return VK_SUCCESS;
            },
            .retireAsync = [&](VkFence, std::shared_ptr<void> value) {
                ++asyncRetires; retained = std::move(value); return true;
            },
            .retainConservatively = [&](std::shared_ptr<void> value) {
                retained = std::move(value); return true;
            }};
    };

    VkSemaphore applicationWaits[]{h<VkSemaphore>(1), h<VkSemaphore>(3)};
    auto bridge = BatchApplicationPresentBridgeAuthority::create(h<VkQueue>(2),
        applicationWaits, 2, 2, std::make_shared<int>(1), operations());
    assert(bridge && bridge->signalCount() == 2);
    assert(bridge->signal(0) != bridge->signal(1));
    auto producer0 = bridge->signalBacking(0);
    assert(producer0);
    assert(bridge->submit() == VK_SUCCESS && bridge->accepted());
    assert(bridge->markSignalConsumed(0));
    assert(submits == 1 && waits == 0);
    assert(bridge->recoverPartial() == VK_SUCCESS);
    assert(waits == 1 && producer0.use_count() >= 1);

    // Exactly one application wait, with both signals consumed.
    expectedWaitCount = 1;
    auto oneWait = BatchApplicationPresentBridgeAuthority::create(h<VkQueue>(2),
        applicationWaits, 1, 2, std::make_shared<int>(5), operations());
    assert(oneWait && oneWait->submit() == VK_SUCCESS);
    assert(oneWait->markSignalConsumed(0));
    assert(oneWait->markSignalConsumed(1));
    auto consumedProducer = oneWait->signalBacking(0);
    std::weak_ptr<void> consumedAlive = consumedProducer;
    assert(oneWait->retireNormalAsync());
    retained.reset();
    assert(!consumedAlive.expired());
    consumedProducer.reset();
    assert(consumedAlive.expired());

    retained.reset();
    expectedWaitCount = 0;
    auto normal = BatchApplicationPresentBridgeAuthority::create(h<VkQueue>(2),
        nullptr, 0, 2, std::make_shared<int>(2), operations());
    assert(normal && normal->submit() == VK_SUCCESS);
    assert(normal->retireNormalAsync());
    assert(asyncRetires == 2 && waits == 1 && retained);

    retained.reset();
    submitResult = VK_ERROR_DEVICE_LOST;
    auto lost = BatchApplicationPresentBridgeAuthority::create(h<VkQueue>(2),
        nullptr, 0, 2, std::make_shared<int>(3), operations());
    assert(lost && lost->submit() == VK_ERROR_DEVICE_LOST);
    assert(lost->state() == BatchApplicationPresentBridgeAuthority::State::ConservativelyRetained);
    assert(retained && waits == 1);

    submitResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    auto rejected = BatchApplicationPresentBridgeAuthority::create(h<VkQueue>(2),
        nullptr, 0, 2, std::make_shared<int>(4), operations());
    assert(rejected && rejected->submit() == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(!rejected->accepted());

    submitResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    auto deviceMemoryRejected = BatchApplicationPresentBridgeAuthority::create(
        h<VkQueue>(2), nullptr, 0, 2, std::make_shared<int>(6), operations());
    assert(deviceMemoryRejected
        && deviceMemoryRejected->submit() == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(!deviceMemoryRejected->accepted());

    // A pre-submit destructor owns no GPU work and performs no fence wait.
    std::weak_ptr<void> preSubmitSignal;
    const auto waitsBeforeDestructor = waits;
    {
        auto preSubmit = BatchApplicationPresentBridgeAuthority::create(
            h<VkQueue>(2), nullptr, 0, 2, std::make_shared<int>(7), operations());
        assert(preSubmit);
        preSubmitSignal = preSubmit->signalBacking(0);
        assert(!preSubmitSignal.expired());
    }
    assert(preSubmitSignal.expired());
    assert(waits == waitsBeforeDestructor);
    assert(submits == 6);
    return 0;
}
