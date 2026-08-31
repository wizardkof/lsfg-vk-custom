#include "adaptive_1x_recovery_authority.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>

using namespace lsfgvk::layer;

template <typename T> T fake(uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}

struct Counters {
    int acquireWaits{};
    int producerWaits{};
    int releases{};
    int retains{};
    uint32_t releasedIndex{};
};

static Adaptive1xRecoveryAuthority makeAuthority(
        const std::shared_ptr<Counters>& counters,
        SwapchainReleaseBackend backend = SwapchainReleaseBackend::Khr,
        uint64_t generation = 3) {
    return Adaptive1xRecoveryAuthority(1, 2, generation, fake<VkSwapchainKHR>(4), backend,
        fake<VkSemaphore>(5), fake<VkFence>(6), fake<VkFence>(7),
        std::make_shared<int>(8), {
            .waitAcquireCompletion = [counters] { ++counters->acquireWaits; return VK_SUCCESS; },
            .waitProducerCompletion = [counters] { ++counters->producerWaits; return VK_SUCCESS; },
            .releasePhysicalImage = [counters](uint32_t index) {
                ++counters->releases; counters->releasedIndex = index; return VK_SUCCESS;
            },
            .retainTerminal = [counters](std::shared_ptr<void>) {
                ++counters->retains; return true;
            }});
}

static Adaptive1xRecoveryAuthority readyForPresent(
        const std::shared_ptr<Counters>& counters) {
    auto value = makeAuthority(counters);
    assert(value.markAcquireCalled());
    assert(value.bindAcquireResult(VK_SUCCESS, 9));
    assert(value.markProducerPrepared());
    assert(value.bindProducerSubmitResult(VK_SUCCESS));
    assert(value.markReadyForLogicalPresent());
    assert(value.markLogicalPresentCalled());
    return value;
}

int main() {
    {
        auto counters = std::make_shared<Counters>();
        auto value = makeAuthority(counters, SwapchainReleaseBackend::None);
        assert(!value.eligible());
        assert(!value.markAcquireCalled());
    }
    for (auto backend : {SwapchainReleaseBackend::Khr, SwapchainReleaseBackend::Ext}) {
        auto counters = std::make_shared<Counters>();
        auto value = makeAuthority(counters, backend);
        assert(value.eligible() && value.generation() == 3);
        assert(value.matchesGeneration(3) && !value.matchesGeneration(4));
        assert(value.markAcquireCalled());
        assert(!value.markAcquireCalled());
        assert(value.bindAcquireResult(VK_SUCCESS, 9));
        assert(value.recoverAcquireFailureOnly() == VK_SUCCESS);
        assert(counters->acquireWaits == 1 && counters->producerWaits == 0);
        assert(counters->releases == 1 && counters->releasedIndex == 9);
        assert(value.terminalAction()
            == Adaptive1xRecoveryAuthority::TerminalAction::ReleaseWithoutPresent);
        assert(value.recoverAcquireFailureOnly() == VK_ERROR_INITIALIZATION_FAILED);
        assert(counters->releases == 1);
        auto recycled = makeAuthority(counters, backend, 4);
        assert(recycled.matchesGeneration(4) && recycled.markAcquireCalled());
    }
    for (auto failure : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        auto counters = std::make_shared<Counters>();
        auto value = makeAuthority(counters);
        assert(value.markAcquireCalled());
        assert(value.bindAcquireResult(VK_SUCCESS, 10));
        assert(value.markProducerPrepared());
        assert(value.bindProducerSubmitResult(failure));
        assert(value.recoverAcquireFailureOnly() == VK_SUCCESS);
        assert(counters->acquireWaits == 1 && counters->releases == 1);
    }
    for (auto failure : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
        auto counters = std::make_shared<Counters>();
        auto value = readyForPresent(counters);
        assert(value.finalizeLogicalPresent(failure));
        assert(value.recoverProducerFailureOnly() == VK_SUCCESS);
        assert(counters->producerWaits == 1 && counters->acquireWaits == 0);
        assert(counters->releases == 1);
    }
    for (auto result : {VK_SUCCESS, VK_SUBOPTIMAL_KHR, VK_ERROR_OUT_OF_DATE_KHR,
            VK_ERROR_SURFACE_LOST_KHR, VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
            VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT}) {
        auto counters = std::make_shared<Counters>();
        auto value = readyForPresent(counters);
        assert(value.finalizeLogicalPresent(result));
        assert(value.terminalAction()
            == Adaptive1xRecoveryAuthority::TerminalAction::PresentRetirement);
        assert(counters->releases == 0 && counters->acquireWaits == 0
            && counters->producerWaits == 0);
        assert(value.markPresentRetired());
        assert(!value.markPresentRetired());
    }
    for (auto result : {VK_ERROR_DEVICE_LOST, VK_ERROR_VALIDATION_FAILED_EXT}) {
        auto counters = std::make_shared<Counters>();
        auto value = readyForPresent(counters);
        assert(value.finalizeLogicalPresent(result));
        assert(counters->retains == 1 && counters->releases == 0);
        assert(value.terminalAction()
            == Adaptive1xRecoveryAuthority::TerminalAction::ConservativeRetention);
    }
    {
        auto counters = std::make_shared<Counters>();
        auto value = makeAuthority(counters);
        assert(value.markAcquireCalled());
        assert(value.bindAcquireResult(VK_ERROR_OUT_OF_DATE_KHR, 99));
        assert(value.state() == Adaptive1xRecoveryAuthority::State::AcquireFailed);
        assert(counters->releases == 0);
    }
    return 0;
}
