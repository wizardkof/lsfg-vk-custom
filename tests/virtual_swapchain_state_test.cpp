/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "../lsfg-vk-layer/src/virtual_swapchain_state.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

using lsfgvk::layer::VirtualSwapchainState;
using namespace std::chrono_literals;

namespace {
std::atomic_bool failNextAllocation{};
}

void* operator new(std::size_t size) {
    if (failNextAllocation.exchange(false))
        throw std::bad_alloc();
    if (void* storage = std::malloc(size))
        return storage;
    throw std::bad_alloc();
}

void operator delete(void* storage) noexcept { std::free(storage); }
void operator delete(void* storage, std::size_t) noexcept { std::free(storage); }

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

void preparedPresentIsTwoPhaseAndCommitIsAtomic() {
    VirtualSwapchainState state(1);
    const auto image = state.tryAcquire();
    require(image.has_value(), "image should acquire before prepare");
    {
        auto abandoned = state.preparePresent(*image, 70);
        require(abandoned && abandoned->valid(), "prepare should own one node");
        require(!state.waitPresent(1ns), "prepare must not publish or change state");
    }
    require(!state.tryAcquire(), "abandoning prepare must leave image acquired");

    auto prepared = state.preparePresent(*image, 71);
    require(prepared && prepared->valid(), "second prepare should succeed");
    require(state.commitPreparedPresent(*prepared), "prepared node should commit");
    require(!prepared->valid(), "commit should consume prepared node");
    const auto present = state.waitPresent(1ns);
    require(present && present->imageIndex == *image && present->serial == 71,
        "state and present payload must publish together");
}

void preparedCommitInvariantFailureDoesNotHalfCommit() {
    VirtualSwapchainState state(1);
    const auto image = state.tryAcquire();
    require(image.has_value(), "image should acquire before invariant test");
    auto prepared = state.preparePresent(*image, 72);
    require(prepared.has_value(), "prepare should succeed before stop");
    state.stop();
    require(!state.commitPreparedPresent(*prepared),
        "stopped state must reject post-bridge-style commit");
    require(prepared->valid(), "failed commit must retain its node authority");
    require(!state.waitPresent(1ns), "failed commit must not publish a half-state");
}

void preparedPresentOomIsCleanAndCommitDoesNotAllocate() {
    VirtualSwapchainState state(1);
    const auto image = state.tryAcquire();
    require(image.has_value(), "image should acquire before allocation test");

    bool sawBadAlloc{};
    failNextAllocation.store(true);
    try {
        (void)state.preparePresent(*image, 73);
    } catch (const std::bad_alloc&) {
        sawBadAlloc = true;
    }
    failNextAllocation.store(false);
    require(sawBadAlloc, "prepared node allocation failure must remain precommit");
    require(!state.waitPresent(1ns), "prepare OOM must not publish a present");
    require(!state.tryAcquire(), "prepare OOM must leave the image acquired");

    auto prepared = state.preparePresent(*image, 74);
    require(prepared.has_value(), "prepare should recover after allocation failure");
    failNextAllocation.store(true);
    const bool committed = state.commitPreparedPresent(*prepared);
    const bool allocationStillArmed = failNextAllocation.exchange(false);
    require(committed, "legal prepared commit must succeed");
    require(allocationStillArmed, "prepared commit must not allocate");
    const auto present = state.waitPresent(1ns);
    require(present && present->serial == 74,
        "nonallocating commit must atomically publish its node");
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


void eventDrivenWakeGeneration() {
    VirtualSwapchainState state(1);
    const auto generation = state.wakeGeneration();
    std::atomic_bool woke{false};

    std::thread waiter([&]() {
        state.waitForWake(generation, VirtualSwapchainState::Duration::max());
        woke.store(true);
    });

    std::this_thread::sleep_for(1ms);
    require(!woke.load(), "generation waiter must sleep without a timed retry");
    state.wake();
    waiter.join();
    require(woke.load(), "explicit reactor-style wake should unblock generation waiter");
    require(state.wakeGeneration() != generation,
        "explicit wake must advance the generation");
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
    {
        VirtualSwapchainState state(1);
        auto image = state.tryAcquire();
        require(image.has_value(), "batch image acquire");
        auto prebridge = state.prepareBatchPresent(*image, 900);
        require(prebridge.has_value(), "batch state reservation");
        require(state.abortBatchPresent(*prebridge, false),
            "prebridge abort restores Acquired");
        require(state.release(*image), "restored Acquired releases");
        image = state.tryAcquire();
        auto accepted = state.prepareBatchPresent(*image, 901);
        require(accepted.has_value() && state.commitBatchPresent(*accepted),
            "accepted batch enters Presenting");
        require(state.abortBatchPresent(*accepted, true),
            "clean postbridge abort becomes Available");
        require(state.tryAcquire().has_value(), "postbridge abort is reusable");
    }
    basicCycle();
    invalidTransitions();
    preparedPresentIsTwoPhaseAndCommitIsAtomic();
    preparedCommitInvariantFailureDoesNotHalfCommit();
    preparedPresentOomIsCleanAndCommitDoesNotAllocate();
    timeoutAndStop();
    eventDrivenWakeGeneration();
    threadedHandoff();

    std::cout << "All VirtualSwapchainState tests passed.\n";
    return 0;
}
