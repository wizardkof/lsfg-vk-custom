#include "borrowed_present_fence_registry.hpp"

#include <cassert>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <new>

using namespace lsfgvk::layer;

namespace { std::atomic_bool failNextAllocation{}; }
void* operator new(std::size_t size) {
    if (failNextAllocation.exchange(false)) throw std::bad_alloc();
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

namespace {
VkFence fence(uintptr_t value) {
    return reinterpret_cast<VkFence>(value);
}
PresentedPhysicalImageIdentity present(uint64_t operation, uint32_t image = 0) {
    return {71, 81, 0x1234, image, operation};
}
}

int main() {
    BorrowedPresentFenceRegistry observer(71);
    auto leases = std::make_shared<PresentedPhysicalImageLeaseRegistry>();
    const auto f = fence(0x4567);
    const auto lifecycle10 = observer.created(f);
    assert(lifecycle10.valid());

    // Two-phase reservation allocates its list node before physical identity
    // exists; bind mutates that exact node with allocations disabled.
    const auto reservedFence = fence(0x4550);
    assert(observer.created(reservedFence).valid());
    BorrowedPresentFenceRegistry::PrepareFailure reserveFailure{};
    failNextAllocation = true;
    assert(!observer.reserve(reservedFence, leases, &reserveFailure));
    assert(reserveFailure
        == BorrowedPresentFenceRegistry::PrepareFailure::OutOfHostMemory);
    auto reserved = observer.reserve(reservedFence, leases);
    assert(reserved && reserved->valid());
    const auto epochBeforeReset = observer.identity(reservedFence).fenceLifecycleEpoch;
    assert(observer.resetBoundary(reservedFence) == 0);
    assert(observer.identity(reservedFence).fenceLifecycleEpoch == epochBeforeReset);
    failNextAllocation = true;
    auto bound = observer.bindPresentedIdentity(*reserved, present(90));
    assert(bound && bound->valid());
    assert(failNextAllocation.exchange(false));
    observer.abort(*bound);
    assert(observer.associationCount() == 0);

    auto first = std::make_shared<int>(1);
    const std::weak_ptr<int> firstLifetime = first;
    assert(leases->install(present(91), first));
    assert(leases->contains(present(91)));
    first.reset();
    auto preparedFirst = observer.prepare(f, leases, present(91));
    assert(preparedFirst && preparedFirst->valid());
    // F1: completion can race ahead of QueuePresent classification.
    assert(observer.retireObserved(f) == 0);
    assert(observer.associationCount() == 1);
    failNextAllocation = true;
    assert(observer.commit(*preparedFirst) == 1);
    assert(failNextAllocation.exchange(false));
    assert(observer.associationCount() == 0 && firstLifetime.expired());

    // NOT_READY is represented by making no observer call. SUCCESS retires once.
    assert(observer.retireObserved(f) == 0);
    assert(firstLifetime.expired());
    assert(observer.retireObserved(f) == 0);

    // Reset is an application-owned lifecycle boundary and permits a new op.
    auto second = std::make_shared<int>(2);
    const std::weak_ptr<int> secondLifetime = second;
    assert(leases->install(present(92, 1), second));
    second.reset();
    auto preparedSecond = observer.prepare(f, leases, present(92, 1));
    assert(preparedSecond && observer.commit(*preparedSecond) == 0);
    uint32_t failedResetCalls{};
    assert(observeBorrowedResetFences(observer, 1, &f, [&] {
        ++failedResetCalls; return VK_ERROR_DEVICE_LOST;
    }) == VK_ERROR_DEVICE_LOST);
    assert(failedResetCalls == 1 && !secondLifetime.expired());
    uint32_t preResetStatusCalls{};
    assert(observeBorrowedResetFences(observer, 1, &f, [&] {
        ++failedResetCalls; return VK_SUCCESS;
    }, [&](VkFence value) {
        ++preResetStatusCalls;
        assert(value == f);
        return VK_NOT_READY;
    }) == VK_SUCCESS);
    assert(failedResetCalls == 2 && secondLifetime.expired());
    assert(preResetStatusCalls == 1 && observer.associationCount() == 0);

    // Destroy closes generation 10 before the numeric handle can be reused.
    auto third = std::make_shared<int>(3);
    const std::weak_ptr<int> thirdLifetime = third;
    assert(leases->install(present(93, 2), third));
    third.reset();
    auto staleEpoch = observer.prepare(f, leases, present(93, 2));
    assert(staleEpoch
        && staleEpoch->fenceLifecycleEpoch == lifecycle10.fenceLifecycleEpoch + 1);
    assert(staleEpoch && observer.commit(*staleEpoch) == 0);
    assert(observer.destroyBoundary(f) == 1 && thirdLifetime.expired());
    assert(!observer.contains(f));
    const auto lifecycle11 = observer.created(f);
    assert(lifecycle11.valid());
    assert(lifecycle11.fenceObjectId != lifecycle10.fenceObjectId);
    assert(observer.retireObserved(f) == 0);

    // Wrong device identities and unknown raw handles never associate.
    auto invalid = present(94, 3);
    invalid.deviceLifetimeIdentity = 72;
    assert(!observer.associate(f, leases, invalid));
    assert(!observer.associate(fence(0x9999), leases, present(95, 3)));
    assert(observer.destroyBoundary(f) == 0);

    // F2/F7: commit-before-completion retires once; an observed but aborted
    // generation never retires its prepared lease.
    const auto phaseFence = fence(0x8888);
    assert(observer.created(phaseFence).valid());
    auto committedBacking = std::make_shared<int>(8);
    auto committedLease = leases->prepare(present(108, 8), committedBacking);
    assert(committedLease && leases->commit(*committedLease));
    auto committedGeneration = observer.prepare(phaseFence, leases, present(108, 8));
    assert(committedGeneration && observer.commit(*committedGeneration) == 0);
    assert(observer.retireObserved(phaseFence) == 1);
    auto abortedBacking = std::make_shared<int>(9);
    auto abortedLease = leases->prepare(present(109, 9), abortedBacking);
    assert(abortedLease && leases->commit(*abortedLease));
    auto abortedGeneration = observer.prepare(phaseFence, leases, present(109, 9));
    assert(abortedGeneration && observer.retireObserved(phaseFence) == 0);
    observer.abort(*abortedGeneration);
    assert(leases->contains(present(109, 9)));
    assert(leases->retireAfterPresentFence(present(109, 9)));

    // F5/F6: waitAny SUCCESS and TIMEOUT both harvest exact signaled fences
    // through observation-only GetFenceStatus calls without changing VkResult.
    const auto harvestA = fence(0x9001), harvestB = fence(0x9002);
    assert(observer.created(harvestA).valid());
    assert(observer.created(harvestB).valid());
    auto harvestBackingA = std::make_shared<int>(10);
    auto harvestBackingB = std::make_shared<int>(11);
    assert(leases->install(present(110, 10), harvestBackingA));
    assert(leases->install(present(111, 11), harvestBackingB));
    assert(observer.associate(harvestA, leases, present(110, 10)));
    assert(observer.associate(harvestB, leases, present(111, 11)));
    const VkFence harvestFences[]{harvestA, harvestB};
    uint32_t statusQueries{};
    assert(observeBorrowedWaitForFences(observer, 2, harvestFences, VK_FALSE,
        [] { return VK_SUCCESS; }, [&](VkFence value) {
            ++statusQueries;
            return value == harvestB ? VK_SUCCESS : VK_NOT_READY;
        }) == VK_SUCCESS);
    assert(statusQueries == 2 && leases->contains(present(110, 10))
        && !leases->contains(present(111, 11)));
    statusQueries = 0;
    assert(observeBorrowedWaitForFences(observer, 2, harvestFences, VK_TRUE,
        [] { return VK_TIMEOUT; }, [&](VkFence value) {
            ++statusQueries;
            return value == harvestA ? VK_SUCCESS : VK_NOT_READY;
        }) == VK_TIMEOUT);
    assert(statusQueries == 2 && !leases->contains(present(110, 10)));

    // 4B1C import boundary retires the old payload epoch immediately. A newly
    // imported unsignaled payload is not teardown authority for that old WSI
    // lease.
    const auto importFence = fence(0x9003);
    const auto importIdentity = observer.created(importFence);
    assert(importIdentity.valid());
    auto importBacking = std::make_shared<int>(13);
    assert(leases->install(present(113, 13), importBacking));
    assert(observer.associate(importFence, leases, present(113, 13)));
    assert(observer.importBoundary(importFence) == 1);
    const auto importedIdentity = observer.identity(importFence);
    assert(importedIdentity.fenceObjectId == importIdentity.fenceObjectId
        && importedIdentity.fenceLifecycleEpoch
            == importIdentity.fenceLifecycleEpoch + 1
        && observer.associationCount() == 0
        && !leases->contains(present(113, 13)));
    uint32_t importedPayloadStatus{}, importedPayloadWait{};
    assert(observer.drainForTeardown([&](VkFence) {
        ++importedPayloadStatus;
        return VK_NOT_READY;
    }, [&](VkFence) {
        ++importedPayloadWait;
        return VK_TIMEOUT;
    }) == 0);
    assert(importedPayloadStatus == 0 && importedPayloadWait == 0);

    // BF-T1/BF-T2: neither reservation-only state is evidence that a logical
    // QueuePresentKHR carrying the application fence was ever submitted.
    const auto unboundFence = fence(0x9004);
    const auto boundFence = fence(0x9005);
    assert(observer.created(unboundFence).valid());
    assert(observer.created(boundFence).valid());
    auto teardownUnbound = observer.reserve(unboundFence, leases);
    auto teardownBoundReservation = observer.reserve(boundFence, leases);
    assert(teardownUnbound && teardownBoundReservation);
    auto teardownBound = observer.bindPresentedIdentity(
        *teardownBoundReservation, present(114, 14));
    assert(teardownBound);
    uint32_t uncommittedStatus{}, uncommittedWait{};
    assert(observer.drainForTeardown([&](VkFence) {
        ++uncommittedStatus;
        return VK_NOT_READY;
    }, [&](VkFence) {
        ++uncommittedWait;
        return VK_SUCCESS;
    }) == 0);
    assert(uncommittedStatus == 0 && uncommittedWait == 0);
    observer.abort(*teardownUnbound);
    observer.abort(*teardownBound);

    // F14: device teardown may observe/wait, but never reset or destroy, an
    // application-owned fence whose completion was not seen through wrappers.
    const auto teardownFence = fence(0x9010);
    assert(observer.created(teardownFence).valid());
    auto teardownBacking = std::make_shared<int>(12);
    assert(leases->install(present(112, 12), teardownBacking));
    assert(observer.associate(teardownFence, leases, present(112, 12)));
    uint32_t teardownStatus{}, teardownWait{};
    failNextAllocation = true;
    assert(observer.drainForTeardown([&](VkFence value) {
        ++teardownStatus;
        assert(value == teardownFence);
        return VK_NOT_READY;
    }, [&](VkFence value) {
        ++teardownWait;
        assert(value == teardownFence);
        return VK_SUCCESS;
    }) == 1);
    assert(failNextAllocation.exchange(false));
    assert(teardownStatus == 1 && teardownWait == 1
        && !leases->contains(present(112, 12)));

    // The exact policy used by entrypoint forwards one call, preserves the
    // raw handle/result and never manufactures a wait/reset/destroy.
    const auto transparentFence = fence(0x7777);
    uint32_t createCalls{}, statusCalls{}, waitCalls{}, resetCalls{}, destroyCalls{};
    const auto fakeCreate = [&] { ++createCalls; return VK_SUCCESS; };
    assert(fakeCreate() == VK_SUCCESS);
    assert(observer.created(transparentFence).valid() && createCalls == 1);
    assert(observeBorrowedGetFenceStatus(observer, transparentFence, [&] {
        ++statusCalls; return VK_NOT_READY; }) == VK_NOT_READY);
    assert(statusCalls == 1);
    assert(observeBorrowedWaitForFences(observer, 1, &transparentFence, VK_TRUE, [&] {
        ++waitCalls; return VK_TIMEOUT; }) == VK_TIMEOUT);
    assert(waitCalls == 1);
    assert(observeBorrowedResetFences(observer, 1, &transparentFence, [&] {
        ++resetCalls; return VK_SUCCESS; }) == VK_SUCCESS);
    assert(resetCalls == 1);
    observeBorrowedDestroyFence(observer, transparentFence, [&] { ++destroyCalls; });
    assert(destroyCalls == 1 && !observer.contains(transparentFence));
    return 0;
}
