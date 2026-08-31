#include "device_retirement_reactor.hpp"
#include "owned_present_sync_fd.hpp"
#include "sync_file_completion.hpp"

#include <cassert>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <new>
#include <cstdlib>
#include <sys/eventfd.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace lsfgvk::layer;

namespace {
thread_local bool failAllocations{};
}

void* operator new(std::size_t size) {
    if (failAllocations) throw std::bad_alloc();
    if (auto* value = std::malloc(size ? size : 1)) return value;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace {
class Wake final : public DeviceRetirementWakeTarget {
public:
    void notifyDeviceRetirement() noexcept override {
        {
            const std::scoped_lock lock(mutex);
            notified = true;
        }
        cv.notify_all();
    }

    bool wait() {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(2), [this] { return notified; });
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    bool notified{};
};
}

int main() {
    // F17/F18: status classification is shared by reactor and teardown, and
    // the original descriptor remains valid until its backing owner dies.
    assert(classifySyncFileCompletion(-5) == SyncFileCompletionState::GpuError);
    assert(classifySyncFileCompletion(0) == SyncFileCompletionState::Pending);
    assert(classifySyncFileCompletion(1) == SyncFileCompletionState::Completed);
    int original{-1};
    {
        int descriptors[2]{};
        assert(::pipe(descriptors) == 0);
        static_cast<void>(::close(descriptors[1]));
        original = descriptors[0];
        OwnedPresentSyncFd owner;
        owner.accept(original);
        const auto waiterSnapshot = owner.snapshot();
        // Models reactor markCompleted(): it changes authority only and cannot
        // invalidate the already captured original descriptor.
        assert(waiterSnapshot == original);
        assert(::fcntl(waiterSnapshot, F_GETFD) != -1);
        assert(querySyncFileCompletion(waiterSnapshot)
            == SyncFileCompletionState::FailedPermanent);
    }
    errno = 0;
    assert(::fcntl(original, F_GETFD) == -1 && errno == EBADF);

    assert(classifySyncFileStatus(0) == DeviceRetirementTicketState::Pending);
    assert(classifySyncFileStatus(1) == DeviceRetirementTicketState::Completed);
    assert(classifySyncFileStatus(-5) == DeviceRetirementTicketState::GpuError);
    assert(classifySyncFileStatus(2)
        == DeviceRetirementTicketState::FailedPermanent);

    DeviceRetirementReactor reactor;
    auto ticket = std::make_shared<DeviceRetirementTicket>();
    auto wake = std::make_shared<Wake>();
    auto backing = std::make_shared<int>(7);
    const int fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    assert(fd >= 0);
    const int observedFd = fd;
    DeviceRetirementTicketState retired{DeviceRetirementTicketState::Pending};
    assert(reactor.registerJob(DeviceRetirementJob{
        .deviceIdentity = 1,
        .swapchainLifecycleIdentity = 2,
        .operationIdentity = 3,
        .completionFd = fd,
        .retainedBacking = backing,
        .ticket = ticket,
        .wakeTarget = wake,
        .retireBacking = [&](DeviceRetirementTicketState state) { retired = state; }
    }));
    const uint64_t signal{1};
    assert(::write(observedFd, &signal, sizeof(signal)) == sizeof(signal));
    assert(wake->wait());
    // eventfd is deliberately not a sync_file, so the production ioctl must
    // conservatively classify it rather than manufacture successful retirement.
    assert(ticket->state() == DeviceRetirementTicketState::FailedPermanent);
    assert(retired == DeviceRetirementTicketState::FailedPermanent);
    assert(::fcntl(observedFd, F_GETFD) == -1);
    assert(reactor.pendingCount() == 0);
    assert(reactor.eventSourceLostCount() == 1);
    reactor.stop();

    // R0-1/R0-2: a cold reactor accepts several extracted tree nodes without
    // allocating destination buckets or other indispensable host storage.
    DeviceRetirementReactor cold;
    std::array<DeviceRetirementReactor::ReservedJob, 3> reserved;
    std::array<int, 3> coldFds{};
    std::array<std::shared_ptr<int>, 3> coldBackings{
        std::make_shared<int>(1), std::make_shared<int>(2),
        std::make_shared<int>(3)};
    for (size_t i = 0; i < reserved.size(); ++i) {
        auto ticket = std::make_shared<DeviceRetirementTicket>();
        auto job = cold.reserveJob(DeviceRetirementJob{
            .deviceIdentity = 31, .swapchainLifecycleIdentity = 32,
            .operationIdentity = static_cast<uint64_t>(33 + i),
            .ticket = ticket});
        assert(job);
        reserved[i] = std::move(*job);
        coldFds[i] = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        assert(coldFds[i] >= 0);
    }
    failAllocations = true;
    assert(cold.registerReservedJob(reserved[2], coldFds[2], coldBackings[2]));
    assert(cold.registerReservedJob(reserved[0], coldFds[0], coldBackings[0]));
    assert(cold.registerReservedJob(reserved[1], coldFds[1], coldBackings[1]));
    failAllocations = false;
    assert(cold.pendingCount() == 3);
    cold.stop();

    // R0-C: collisions and invalid lifecycle operations reject without
    // consuming caller-owned descriptors or retaining caller backing.
    DeviceRetirementReactor collisions;
    collisions.setNextRegistrationForTesting(77);
    auto collisionTicket0 = std::make_shared<DeviceRetirementTicket>();
    auto collision0 = collisions.reserveJob(DeviceRetirementJob{
        .deviceIdentity = 40, .swapchainLifecycleIdentity = 41,
        .operationIdentity = 42, .ticket = collisionTicket0});
    collisions.setNextRegistrationForTesting(77);
    auto collisionTicket1 = std::make_shared<DeviceRetirementTicket>();
    auto collision1 = collisions.reserveJob(DeviceRetirementJob{
        .deviceIdentity = 40, .swapchainLifecycleIdentity = 41,
        .operationIdentity = 43, .ticket = collisionTicket1});
    assert(collision0 && collision1);
    auto collisionBacking0 = std::make_shared<int>(40);
    auto collisionBacking1 = std::make_shared<int>(41);
    const auto collisionFd0 = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    const auto collisionFd1 = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    assert(collisionFd0 >= 0 && collisionFd1 >= 0);
    failAllocations = true;
    assert(collisions.registerReservedJob(
        *collision0, collisionFd0, collisionBacking0));
    assert(!collisions.registerReservedJob(
        *collision1, collisionFd1, collisionBacking1));
    failAllocations = false;
    assert(collision1->valid());
    assert(::fcntl(collisionFd1, F_GETFD) != -1);
    static_cast<void>(::close(collisionFd1));

    DeviceRetirementReactor::ReservedJob invalid;
    const auto invalidFd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    assert(invalidFd >= 0);
    assert(!collisions.registerReservedJob(
        invalid, invalidFd, collisionBacking1));
    assert(::fcntl(invalidFd, F_GETFD) != -1);
    static_cast<void>(::close(invalidFd));

    const auto doubleFd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    assert(doubleFd >= 0);
    assert(!collisions.registerReservedJob(
        *collision0, doubleFd, collisionBacking0));
    assert(::fcntl(doubleFd, F_GETFD) != -1);
    static_cast<void>(::close(doubleFd));
    collisions.stop();

    const auto stoppedFd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    assert(stoppedFd >= 0);
    assert(!collisions.registerReservedJob(
        *collision1, stoppedFd, collisionBacking1));
    assert(::fcntl(stoppedFd, F_GETFD) != -1);
    static_cast<void>(::close(stoppedFd));

    DeviceRetirementReactor immediate;
    auto immediateTicket = std::make_shared<DeviceRetirementTicket>();
    auto immediateWake = std::make_shared<Wake>();
    auto immediateBacking = std::make_shared<int>(9);
    assert(immediate.completeImmediately(DeviceRetirementJob{
        .deviceIdentity = 1,
        .swapchainLifecycleIdentity = 2,
        .operationIdentity = 4,
        .completionFd = -1,
        .retainedBacking = immediateBacking,
        .ticket = immediateTicket,
        .wakeTarget = immediateWake
    }));
    assert(immediateTicket->state() == DeviceRetirementTicketState::Completed);
    assert(immediateWake->wait());
    assert(immediate.retainEventSourceLost(immediateBacking));
    assert(immediate.eventSourceLostCount() == 1);
    immediate.stop();

    // A5G-TICKET-SPLIT: terminal completion is observed exactly once while
    // both D3B2 WSI presentation domains remain independently pending. The
    // physical backing stays alive until the whole pending operation releases
    // it after the last presentation ticket.
    DeviceRetirementReactor splitReactor;
    std::vector<std::shared_ptr<DeviceRetirementTicket>> splitTickets{
        std::make_shared<DeviceRetirementTicket>(),
        std::make_shared<DeviceRetirementTicket>(),
        std::make_shared<DeviceRetirementTicket>()};
    auto splitBacking = std::make_shared<int>(11);
    const std::weak_ptr<int> backingObserver = splitBacking;
    auto complete = [&](size_t index, uint64_t identity) {
        assert(splitReactor.completeImmediately(DeviceRetirementJob{
            .deviceIdentity = 21,
            .swapchainLifecycleIdentity = 22,
            .operationIdentity = identity,
            .completionFd = -1,
            .retainedBacking = splitBacking,
            .ticket = splitTickets.at(index)
        }));
    };
    TerminalWsiCompletionGate splitGate;
    assert(splitGate.observe(splitTickets)
        == TerminalWsiCompletionGateStatus::Pending);
    complete(0, 23);
    assert(splitGate.observe(splitTickets)
        == TerminalWsiCompletionGateStatus::TerminalReady);
    uint32_t terminalRetirements{};
    ++terminalRetirements;
    splitGate.markTerminalRetired();
    assert(terminalRetirements == 1 && splitGate.terminalRetired());
    assert(splitGate.observe(splitTickets)
        == TerminalWsiCompletionGateStatus::Pending);
    assert(splitTickets[1]->state() == DeviceRetirementTicketState::Pending);
    assert(splitTickets[2]->state() == DeviceRetirementTicketState::Pending);
    assert(!backingObserver.expired());
    complete(1, 24);
    assert(splitGate.observe(splitTickets)
        == TerminalWsiCompletionGateStatus::Pending);
    assert(splitTickets[2]->state() == DeviceRetirementTicketState::Pending);
    assert(terminalRetirements == 1 && !backingObserver.expired());
    complete(2, 25);
    assert(splitGate.observe(splitTickets)
        == TerminalWsiCompletionGateStatus::WsiReady);
    assert(terminalRetirements == 1 && !backingObserver.expired());
    splitReactor.stop();
}
