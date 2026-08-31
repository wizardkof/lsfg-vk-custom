#include "adaptive_1x_prepared_logical_final.hpp"

#include <cassert>
#include <memory>

namespace {

using lsfgvk::layer::Adaptive1xPreparedLogicalFinal;
using lsfgvk::layer::Adaptive1xRecoveryAuthority;
using lsfgvk::layer::BorrowedPresentFenceRegistry;
using lsfgvk::layer::PresentedPhysicalImageAcquireToken;
using lsfgvk::layer::PresentedPhysicalImageIdentity;
using lsfgvk::layer::PresentedPhysicalImageLeaseRegistry;
using lsfgvk::layer::SwapchainReleaseBackend;

template<class T> T handle(uintptr_t value) {
    return reinterpret_cast<T>(value);
}

struct Probe {
    uint32_t acquireWaits{};
    uint32_t producerWaits{};
    uint32_t releases{};
    uint32_t arms{};
    uint32_t exports{};
    uint32_t publishes{};
    uint32_t retains{};
    VkResult exportResult{VK_SUCCESS};
    VkFence producerFence{};
    VkSemaphore readySemaphore{};
};

struct Fixture {
    std::shared_ptr<Probe> probe{std::make_shared<Probe>()};
    std::shared_ptr<PresentedPhysicalImageLeaseRegistry> leases{
        std::make_shared<PresentedPhysicalImageLeaseRegistry>()};
    std::shared_ptr<BorrowedPresentFenceRegistry> borrowed{
        std::make_shared<BorrowedPresentFenceRegistry>(7)};
    uint64_t generation{1};

    Adaptive1xPreparedLogicalFinal make(bool applicationFence = false,
            SwapchainReleaseBackend backend = SwapchainReleaseBackend::Khr,
            uint32_t imageIndex = 0) {
        const auto swapchain = handle<VkSwapchainKHR>(0x1000 + generation);
        const auto acquireSemaphore = handle<VkSemaphore>(0x2000 + generation);
        const auto acquireFence = handle<VkFence>(0x3000 + generation);
        probe->producerFence = handle<VkFence>(0x4000 + generation);
        probe->readySemaphore = handle<VkSemaphore>(0x5000 + generation);
        const auto appFence = applicationFence
            ? handle<VkFence>(0x6000 + generation) : VK_NULL_HANDLE;
        const auto internalFence = handle<VkFence>(0x7000 + generation);
        auto backing = std::make_shared<uint64_t>(generation);
        const PresentedPhysicalImageIdentity identity{
            .deviceLifetimeIdentity = 7,
            .swapchainLifecycleIdentity = 11 + generation,
            .physicalSwapchainIdentity = reinterpret_cast<uintptr_t>(swapchain),
            .physicalImageIndex = imageIndex,
            .presentOperationIdentity = 100 + generation};
        auto preparedLease = leases->prepare(identity, backing);
        assert(preparedLease);

        std::optional<BorrowedPresentFenceRegistry::PreparedBorrowedPresentFence>
            preparedFence;
        if (appFence) {
            assert(borrowed->created(appFence).valid());
            preparedFence = borrowed->prepare(appFence, leases, identity);
            assert(preparedFence);
        }
        Adaptive1xRecoveryAuthority recovery(7, 11 + generation, generation,
            swapchain, backend, acquireSemaphore, acquireFence,
            probe->producerFence, backing, {
                .waitAcquireCompletion = [probe = probe] {
                    ++probe->acquireWaits;
                    return VK_SUCCESS;
                },
                .waitProducerCompletion = [probe = probe] {
                    ++probe->producerWaits;
                    return VK_SUCCESS;
                },
                .releasePhysicalImage = [probe = probe](uint32_t) {
                    ++probe->releases;
                    return VK_SUCCESS;
                },
                .retainTerminal = [probe = probe](std::shared_ptr<void>) {
                    ++probe->retains;
                    return true;
                }});
        assert(recovery.markAcquireCalled());
        assert(recovery.bindAcquireResult(VK_SUCCESS, imageIndex));
        assert(recovery.markProducerPrepared());
        assert(recovery.bindProducerSubmitResult(VK_SUCCESS));
        assert(recovery.markReadyForLogicalPresent());

        ++generation;
        return Adaptive1xPreparedLogicalFinal(swapchain, imageIndex,
            probe->readySemaphore, appFence, internalFence, identity,
            std::move(recovery), leases, *preparedLease, borrowed,
            std::move(preparedFence), {}, backing, {
                .armInternalPresentFence = [probe = probe](
                        const PresentedPhysicalImageIdentity&) {
                    ++probe->arms;
                    return true;
                },
                .exportProducerCompletion = [probe = probe](int* fd) {
                    ++probe->exports;
                    if (fd) *fd = -1;
                    return probe->exportResult;
                },
                .publishCompletion = [probe = probe](int,
                        std::vector<PresentedPhysicalImageAcquireToken>&&) {
                    ++probe->publishes;
                    return VK_SUCCESS;
                },
                .retainConservatively = [probe = probe](std::shared_ptr<void>) {
                    ++probe->retains;
                    return true;
                }});
    }
};

void finalizeEnqueued(VkResult result) {
    Fixture fixture;
    auto prepared = fixture.make();
    assert(prepared.valid());
    assert(prepared.markLogicalPresentCalled());
    assert(prepared.finalizeLogicalPresent(result) == VK_SUCCESS);
    assert(prepared.state() == Adaptive1xPreparedLogicalFinal::State::Finalized);
    assert(fixture.probe->producerWaits == 0);
    assert(fixture.probe->releases == 0);
    assert(fixture.probe->arms == 1);
    assert(fixture.probe->exports == 1);
    assert(fixture.probe->publishes == 1);
}

void testFinalizationMatrix() {
    finalizeEnqueued(VK_SUCCESS);
    finalizeEnqueued(VK_SUBOPTIMAL_KHR);
    finalizeEnqueued(VK_ERROR_OUT_OF_DATE_KHR);
    finalizeEnqueued(VK_ERROR_SURFACE_LOST_KHR);
    finalizeEnqueued(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
    finalizeEnqueued(VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);

    for (const auto result : {VK_ERROR_OUT_OF_HOST_MEMORY,
            VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        Fixture fixture;
        auto prepared = fixture.make();
        assert(prepared.markLogicalPresentCalled());
        assert(prepared.finalizeLogicalPresent(result) == VK_SUCCESS);
        assert(fixture.probe->producerWaits == 1);
        assert(fixture.probe->releases == 1);
        assert(fixture.probe->exports == 0);
    }

    for (const auto result : {VK_ERROR_DEVICE_LOST, VK_ERROR_UNKNOWN}) {
        Fixture fixture;
        auto prepared = fixture.make();
        assert(prepared.markLogicalPresentCalled());
        assert(prepared.finalizeLogicalPresent(result) == VK_SUCCESS);
        assert(prepared.state() == Adaptive1xPreparedLogicalFinal::State::Retained);
        assert(fixture.probe->retains == 1);
        assert(fixture.probe->releases == 0);
    }
}

void testFenceSplitAndExportFailure() {
    Fixture application;
    auto borrowed = application.make(true);
    assert(!borrowed.requiresInternalPresentFence());
    assert(borrowed.applicationPresentFence() != VK_NULL_HANDLE);
    assert(borrowed.markLogicalPresentCalled());
    assert(borrowed.finalizeLogicalPresent(VK_SUCCESS) == VK_SUCCESS);
    assert(application.probe->arms == 0);

    Fixture failed;
    failed.probe->exportResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    auto retained = failed.make();
    assert(retained.markLogicalPresentCalled());
    assert(retained.finalizeLogicalPresent(VK_SUCCESS)
        == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(retained.state() == Adaptive1xPreparedLogicalFinal::State::Retained);
    assert(failed.probe->retains == 1);
    assert(failed.leases->size() == 1);
}

void testAbandonAndCoexistence() {
    Fixture fixture;
    auto first = fixture.make(false, SwapchainReleaseBackend::Khr, 0);
    const auto firstFence = first.producerFence();
    const auto firstReady = first.presentReadySemaphore();
    auto second = fixture.make(false, SwapchainReleaseBackend::Ext, 1);
    assert(first.valid() && second.valid());
    assert(firstFence != second.producerFence());
    assert(firstReady != second.presentReadySemaphore());
    assert(fixture.probe->producerWaits == 0);

    assert(second.abandonWithoutLogicalPresent() == VK_SUCCESS);
    assert(second.state() == Adaptive1xPreparedLogicalFinal::State::Abandoned);
    assert(first.valid());
    assert(first.markLogicalPresentCalled());
    assert(first.finalizeLogicalPresent(VK_SUCCESS) == VK_SUCCESS);
    assert(fixture.probe->producerWaits == 1);
    assert(fixture.probe->releases == 1);

    Fixture both;
    auto s0 = both.make(false, SwapchainReleaseBackend::Khr, 0);
    auto s1 = both.make(false, SwapchainReleaseBackend::Khr, 1);
    assert(s0.markLogicalPresentCalled());
    assert(s0.finalizeLogicalPresent(VK_SUCCESS) == VK_SUCCESS);
    assert(s1.markLogicalPresentCalled());
    assert(s1.finalizeLogicalPresent(VK_SUCCESS) == VK_SUCCESS);
    assert(both.probe->producerWaits == 0);
    assert(both.probe->releases == 0);
}

void testOutstandingDestructorIsPassiveAndConservative() {
    Fixture fixture;
    {
        auto prepared = fixture.make();
        assert(prepared.valid());
    }
    assert(fixture.probe->producerWaits == 0);
    assert(fixture.probe->releases == 0);
    assert(fixture.leases->size() == 1);
}

} // namespace

int main() {
    testFinalizationMatrix();
    testFenceSplitAndExportFailure();
    testAbandonAndCoexistence();
    testOutstandingDestructorIsPassiveAndConservative();
}
