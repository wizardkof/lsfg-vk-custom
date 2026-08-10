/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "../lsfg-vk-layer/src/virtual_swapchain_state.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using lsfgvk::layer::VirtualSwapchainState;
using namespace std::chrono_literals;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void basicCycle() {
    VirtualSwapchainState state(2);

    const auto first = state.tryAcquire();
    const auto second = state.tryAcquire();
    require(first.has_value() && second.has_value(), "two images should be acquirable");
    require(*first != *second, "acquires must return distinct images");
    require(!state.tryAcquire().has_value(), "third acquire must fail while both are busy");

    require(state.queuePresent(*first, 41), "acquired image should queue for presentation");
    const auto present = state.waitPresent(1ms);
    require(present.has_value(), "queued present should be fetchable");
    require(present->imageIndex == *first && present->serial == 41,
        "present payload should be preserved");
    require(state.complete(*first), "presenting image should complete");
    require(state.tryAcquire().has_value(), "completed image should become available again");

    require(state.release(*second), "unused acquired image should be releasable");
}

void invalidTransitions() {
    VirtualSwapchainState state(1);

    require(!state.release(0), "available image cannot be released");
    require(!state.queuePresent(0, 1), "available image cannot be presented");
    require(!state.complete(0), "available image cannot complete");

    const auto idx = state.tryAcquire();
    require(idx.has_value(), "image should acquire");
    require(!state.complete(*idx), "acquired image cannot complete before presentation");
    require(state.queuePresent(*idx, 2), "acquired image should queue");
    require(!state.release(*idx), "queued image cannot be released");
}

void timeoutAndStop() {
    VirtualSwapchainState state(1);
    const auto idx = state.tryAcquire();
    require(idx.has_value(), "image should acquire");
    require(!state.waitAcquire(1ms).has_value(), "busy pool should time out");

    std::atomic_bool woke{false};
    std::thread waiter([&]() {
        const auto present = state.waitPresent(VirtualSwapchainState::Duration::max());
        require(!present.has_value(), "stop should wake present waiter without payload");
        woke.store(true);
    });

    std::this_thread::sleep_for(1ms);
    state.stop();
    waiter.join();
    require(woke.load(), "stop should unblock waiter");
    require(state.stopped(), "state should report stopped");
    require(!state.tryAcquire().has_value(), "stopped state must not acquire");
}

void threadedHandoff() {
    constexpr uint64_t iterations = 500;
    VirtualSwapchainState state(3);
    std::atomic_uint64_t consumed{0};

    std::thread consumer([&]() {
        for (uint64_t i = 0; i < iterations; ++i) {
            const auto present =
                state.waitPresent(VirtualSwapchainState::Duration::max());
            require(present.has_value(), "consumer should receive queued present");
            require(present->serial == i, "present serials must preserve FIFO order");
            require(state.complete(present->imageIndex), "consumer should complete image");
            consumed.fetch_add(1);
        }
    });

    for (uint64_t i = 0; i < iterations; ++i) {
        const auto idx =
            state.waitAcquire(VirtualSwapchainState::Duration::max());
        require(idx.has_value(), "producer should acquire an image");
        require(state.queuePresent(*idx, i), "producer should queue acquired image");
    }

    consumer.join();
    require(consumed.load() == iterations, "all presents should be consumed");
}

}

int main() {
    basicCycle();
    invalidTransitions();
    timeoutAndStop();
    threadedHandoff();

    std::cout << "All VirtualSwapchainState tests passed.\n";
    return 0;
}
