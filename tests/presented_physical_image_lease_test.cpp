#include "presented_physical_image_lease.hpp"

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
PresentedPhysicalImageIdentity identity(uint64_t lifecycle, uint32_t image,
        uint64_t operation) {
    return {
        .deviceLifetimeIdentity = 11,
        .swapchainLifecycleIdentity = lifecycle,
        .physicalSwapchainIdentity = 0x1234,
        .physicalImageIndex = image,
        .presentOperationIdentity = operation};
}
}

int main() {
    PresentedPhysicalImageLeaseRegistry registry;
    // E1/E2: all storage and the image reservation exist before the simulated
    // downstream call; commit itself only flips prepared state.
    auto preparedBacking = std::make_shared<int>(0);
    const auto preparedIdentity = identity(20, 7, 30);
    auto prepared = registry.prepare(preparedIdentity, preparedBacking);
    assert(prepared && !registry.contains(preparedIdentity));
    assert(!registry.prepare(identity(20, 7, 300), preparedBacking));
    assert(registry.commit(*prepared) && registry.contains(preparedIdentity));
    assert(registry.retireAfterPresentFence(preparedIdentity));

    // E3/E6/E7: abort releases a reservation and invalid/conflicting prepare
    // never publishes a lease.
    auto aborted = registry.prepare(identity(20, 8, 301), preparedBacking);
    assert(aborted);
    registry.abort(*aborted);
    assert(registry.size() == 0);
    auto retried = registry.prepare(identity(20, 8, 302), preparedBacking);
    assert(retried);
    registry.abort(*retried);

    PresentedPhysicalImageLeaseRegistry::PrepareFailure injectedFailure{};
    failNextAllocation = true;
    auto allocationFailure = registry.prepare(
        identity(20, 9, 303), preparedBacking, &injectedFailure);
    assert(!allocationFailure && injectedFailure
        == PresentedPhysicalImageLeaseRegistry::PrepareFailure::OutOfHostMemory);

    auto allocationProof = registry.prepare(identity(20, 9, 304), preparedBacking);
    assert(allocationProof);
    failNextAllocation = true;
    assert(registry.commit(*allocationProof));
    assert(failNextAllocation.exchange(false));
    assert(registry.retireAfterPresentFence(identity(20, 9, 304)));

    // F13/F15: DeviceWaitIdle alone does not touch a committed internal-fence
    // lease. The teardown-only exact fence wait retires it afterwards.
    uint32_t internalWaits{};
    auto internalBacking = std::make_shared<int>(13);
    auto internalPrepared = registry.prepare(identity(20, 10, 305),
        internalBacking, nullptr, [&] {
            ++internalWaits;
            return VK_SUCCESS;
        });
    assert(internalPrepared && registry.commit(*internalPrepared));
    assert(registry.contains(identity(20, 10, 305)) && internalWaits == 0);
    failNextAllocation = true;
    assert(registry.drainInternalFencesForTeardown() == 1);
    assert(failNextAllocation.exchange(false));
    assert(internalWaits == 1 && !registry.contains(identity(20, 10, 305)));

    // Producer-only prepared ownership may be released by final device idle;
    // it is never confused with a committed WSI-present lease.
    auto producerPrepared = registry.prepare(identity(20, 11, 306),
        std::make_shared<int>(14));
    assert(producerPrepared);
    assert(registry.releasePreparedProducerBackingsAfterDeviceIdle() == 1);
    assert(registry.size() == 0);
    auto resourceI = std::make_shared<int>(1);
    auto resourceJ = std::make_shared<int>(2);
    const std::weak_ptr<int> observeI = resourceI;
    const std::weak_ptr<int> observeJ = resourceJ;
    const auto presentI = identity(21, 0, 31);
    const auto presentJ = identity(21, 1, 32);
    assert(registry.install(presentI, resourceI));
    assert(registry.install(presentJ, resourceJ));
    resourceI.reset();
    resourceJ.reset();
    assert(registry.size() == 2 && !observeI.expired() && !observeJ.expired());

    // Reacquiring J does not release I. Acquire alone also does not release J.
    const auto acquireJ = registry.reacquired(11, 21, 0x1234, 1, 41);
    assert(acquireJ && registry.size() == 2);
    assert(!observeI.expired() && !observeJ.expired());
    assert(!registry.retireAfterSafeCompletion(*acquireJ));
    assert(registry.markAcquireConsumed(*acquireJ));
    assert(!observeJ.expired());
    assert(registry.retireAfterSafeCompletion(*acquireJ));
    assert(observeJ.expired() && !observeI.expired());

    auto acquireOnlyResource = std::make_shared<int>(22);
    const std::weak_ptr<int> acquireOnlyObserver = acquireOnlyResource;
    const auto acquireOnlyIdentity = identity(22, 2, 43);
    assert(registry.install(acquireOnlyIdentity, acquireOnlyResource));
    acquireOnlyResource.reset();
    const auto acquireOnly = registry.reacquired(11, 22, 0x1234, 2, 44);
    assert(acquireOnly && !acquireOnlyObserver.expired());
    assert(registry.retireAfterAcquireCompletion(*acquireOnly));
    assert(acquireOnlyObserver.expired());
    assert(!registry.retireAfterAcquireCompletion(*acquireOnly));

    // Same-image I follows acquire -> consumed by real submit -> safe event.
    const auto acquireI = registry.reacquired(11, 21, 0x1234, 0, 42);
    assert(acquireI && registry.contains(presentI) && !observeI.expired());
    assert(!registry.retireAfterSafeCompletion(*acquireI));
    assert(registry.markAcquireConsumed(*acquireI));
    auto nextIResource = std::make_shared<int>(4);
    const std::weak_ptr<int> nextIObserver = nextIResource;
    const auto nextPresentI = identity(21, 0, 33);
    assert(registry.install(nextPresentI, nextIResource));
    nextIResource.reset();
    assert(registry.size() == 2 && !nextIObserver.expired());
    assert(registry.contains(presentI) && !observeI.expired());

    // The old generation remains committed while its reacquire is awaiting
    // safe completion. A later committed presentation of the same physical
    // image is now the active reacquire target and must not be shadowed by
    // unordered-map iteration finding the older generation first.
    const auto acquireNextI = registry.reacquired(11, 21, 0x1234, 0, 45);
    assert(acquireNextI
        && acquireNextI->presented.presentOperationIdentity
            == nextPresentI.presentOperationIdentity);
    assert(registry.markAcquireConsumed(*acquireNextI));

    assert(registry.retireAfterSafeCompletion(*acquireI));
    assert(observeI.expired() && registry.size() == 1);
    assert(registry.retireAfterSafeCompletion(*acquireNextI));
    assert(nextIObserver.expired() && registry.size() == 0);

    // A numeric index from lifecycle N+1 cannot mutate lifecycle N.
    auto staleResource = std::make_shared<int>(3);
    const std::weak_ptr<int> staleObserver = staleResource;
    const auto stale = identity(51, 2, 61);
    assert(registry.install(stale, staleResource));
    staleResource.reset();
    assert(!registry.reacquired(11, 52, 0x1234, 2, 62));
    assert(registry.contains(stale) && !staleObserver.expired());
    assert(!registry.reacquired(11, 51, 0x5678, 2, 63));
    assert(registry.contains(stale) && !staleObserver.expired());

    // Equal numeric image indices in different swapchains/generations are
    // independent physical-image identities and must never alias.
    auto generationA = std::make_shared<int>(5);
    auto generationB = std::make_shared<int>(6);
    const auto imageA = identity(71, 0, 81);
    auto imageB = identity(72, 0, 82);
    imageB.physicalSwapchainIdentity = 0x5678;
    assert(registry.install(imageA, generationA));
    assert(registry.install(imageB, generationB));
    const auto acquireA = registry.reacquired(11, 71, 0x1234, 0, 91);
    assert(acquireA && registry.markAcquireConsumed(*acquireA));
    assert(registry.retireAfterSafeCompletion(*acquireA));
    assert(!registry.contains(imageA) && registry.contains(imageB));
    const auto acquireB = registry.reacquired(11, 72, 0x5678, 0, 92);
    assert(acquireB && registry.markAcquireConsumed(*acquireB));
    assert(registry.retireAfterSafeCompletion(*acquireB));
    assert(!registry.contains(imageB));

    return 0;
}
