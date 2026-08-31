#include "device_retirement_reactor.hpp"
#include "sync_file_completion.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <map>
#include <optional>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace lsfgvk::layer {
namespace {
constexpr uint64_t CONTROL_EVENT_ID{};
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
std::mutex controlledFdsMutex;
std::unordered_set<int> controlledFds;
#endif

void closeFd(int& fd) noexcept {
    if (fd >= 0) {
        static_cast<void>(::close(fd));
        fd = -1;
    }
}

DeviceRetirementTicketState syncFileState(int fd) noexcept {
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    {
        const std::scoped_lock lock(controlledFdsMutex);
        if (controlledFds.erase(fd) != 0)
            return DeviceRetirementTicketState::Completed;
    }
#endif
    switch (querySyncFileCompletion(fd)) {
    case SyncFileCompletionState::Pending:
        return DeviceRetirementTicketState::Pending;
    case SyncFileCompletionState::Completed:
        return DeviceRetirementTicketState::Completed;
    case SyncFileCompletionState::GpuError:
        return DeviceRetirementTicketState::GpuError;
    case SyncFileCompletionState::FailedPermanent:
        return DeviceRetirementTicketState::FailedPermanent;
    }
    return DeviceRetirementTicketState::FailedPermanent;
}
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
void markControlledRetirementFd(int fd) noexcept {
    if (fd < 0) return;
    try {
        const std::scoped_lock lock(controlledFdsMutex);
        controlledFds.insert(fd);
    } catch (...) {}
}
#endif

TerminalWsiCompletionGateStatus TerminalWsiCompletionGate::observe(
        std::span<const std::shared_ptr<DeviceRetirementTicket>> tickets) const noexcept {
    if (tickets.empty() || !tickets.front())
        return TerminalWsiCompletionGateStatus::Pending;
    const auto classifyFailure = [](DeviceRetirementTicketState state) {
        if (state == DeviceRetirementTicketState::DeviceLost)
            return TerminalWsiCompletionGateStatus::DeviceLost;
        if (state == DeviceRetirementTicketState::GpuError
                || state == DeviceRetirementTicketState::FailedPermanent)
            return TerminalWsiCompletionGateStatus::Failed;
        return TerminalWsiCompletionGateStatus::Pending;
    };
    const auto terminal = tickets.front()->state();
    if (terminal != DeviceRetirementTicketState::Completed)
        return classifyFailure(terminal);
    if (!terminalRetiredValue)
        return TerminalWsiCompletionGateStatus::TerminalReady;
    for (size_t index = 1; index < tickets.size(); ++index) {
        if (!tickets[index]) return TerminalWsiCompletionGateStatus::Failed;
        const auto state = tickets[index]->state();
        if (state == DeviceRetirementTicketState::Completed) continue;
        return classifyFailure(state);
    }
    return TerminalWsiCompletionGateStatus::WsiReady;
}

void TerminalWsiCompletionGate::markTerminalRetired() {
    if (terminalRetiredValue)
        throw std::logic_error("terminal completion retired more than once");
    terminalRetiredValue = true;
}

struct DeviceRetirementReactor::State {
    int epollFd{-1};
    int controlFd{-1};
    mutable std::mutex mutex;
    std::map<uint64_t, DeviceRetirementJob> jobs;
    static constexpr size_t MAX_EVENT_SOURCE_LOST_BACKINGS = 4096;
    std::array<std::shared_ptr<void>, MAX_EVENT_SOURCE_LOST_BACKINGS> eventSourceLost{};
    size_t eventSourceLostUsed{};
    std::atomic_bool stopping{false};
    std::atomic<uint64_t> nextRegistration{1};
    std::thread worker;

    ~State() {
        for (auto& [identity, job] : jobs) {
            (void)identity;
            closeFd(job.completionFd);
        }
        closeFd(controlFd);
        closeFd(epollFd);
    }

    void wake() const noexcept {
        const uint64_t value{1};
        if (controlFd >= 0)
            static_cast<void>(::write(controlFd, &value, sizeof(value)));
    }

    bool retainEventSourceLostLocked(std::shared_ptr<void> backing) noexcept {
        if (!backing) return false;
        for (auto& slot : eventSourceLost) {
            if (slot) continue;
            slot = std::move(backing);
            ++eventSourceLostUsed;
            return true;
        }
        return false;
    }

    void run() noexcept {
        std::vector<epoll_event> events(16);
        while (!stopping.load(std::memory_order_acquire)) {
            const auto count = ::epoll_wait(epollFd, events.data(),
                static_cast<int>(events.size()), -1);
            if (count < 0) {
                if (errno == EINTR) continue;
                break;
            }
            for (int index = 0; index < count; ++index) {
                const auto registration = events[index].data.u64;
                if (registration == CONTROL_EVENT_ID) {
                    uint64_t value{};
                    while (::read(controlFd, &value, sizeof(value)) > 0) {}
                    continue;
                }

                DeviceRetirementJob job;
                DeviceRetirementTicketState result{
                    DeviceRetirementTicketState::FailedPermanent};
                std::shared_ptr<DeviceRetirementTicket> strandedTicket;
                std::weak_ptr<DeviceRetirementWakeTarget> strandedWake;
                bool extracted{};
                {
                    const std::scoped_lock lock(mutex);
                    const auto iterator = jobs.find(registration);
                    if (iterator == jobs.end()) continue;
                    result = syncFileState(iterator->second.completionFd);
                    if (result == DeviceRetirementTicketState::Pending) {
                        epoll_event event{};
                        event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLONESHOT;
                        event.data.u64 = registration;
                        if (::epoll_ctl(epollFd, EPOLL_CTL_MOD,
                                iterator->second.completionFd, &event) == 0)
                            continue;
                        // Losing the event source after an accepted submit is
                        // conservative failure, not permission to release GPU
                        // backing. Fall through to fixed retention.
                        result = DeviceRetirementTicketState::FailedPermanent;
                    }
                    static_cast<void>(::epoll_ctl(epollFd, EPOLL_CTL_DEL,
                        iterator->second.completionFd, nullptr));

                    if (result != DeviceRetirementTicketState::Completed
                            && !retainEventSourceLostLocked(
                                iterator->second.retainedBacking)) {
                        // Fixed retention is exhausted. Keep the job itself in
                        // the map (with no live event FD) so its strong backing
                        // survives until the explicit VkDevice teardown drain.
                        closeFd(iterator->second.completionFd);
                        strandedTicket = iterator->second.ticket;
                        strandedWake = iterator->second.wakeTarget;
                    } else {
                        job = std::move(iterator->second);
                        jobs.erase(iterator);
                        extracted = true;
                    }
                }

                if (!extracted) {
                    if (strandedTicket)
                        strandedTicket->value.store(result, std::memory_order_release);
                    if (strandedTicket) strandedTicket->notifyWakeTarget();
                    if (const auto target = strandedWake.lock())
                        target->notifyDeviceRetirement();
                    continue;
                }

                closeFd(job.completionFd);
                if (job.retireBacking) {
                    try {
                        job.retireBacking(result);
                    } catch (...) {
                        result = DeviceRetirementTicketState::FailedPermanent;
                    }
                }
                if (job.ticket)
                    job.ticket->value.store(result, std::memory_order_release);
                if (job.ticket) job.ticket->notifyWakeTarget();
            }
        }
    }
};

namespace {
struct ReservedReactorJobStorage {
    using Map = std::map<uint64_t, DeviceRetirementJob>;
    Map::node_type node;
};
}

DeviceRetirementReactor::DeviceRetirementReactor() : state(std::make_shared<State>()) {
    state->epollFd = ::epoll_create1(EPOLL_CLOEXEC);
    state->controlFd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (state->epollFd < 0 || state->controlFd < 0)
        throw std::runtime_error(std::strerror(errno));
    epoll_event control{};
    control.events = EPOLLIN;
    control.data.u64 = CONTROL_EVENT_ID;
    if (::epoll_ctl(state->epollFd, EPOLL_CTL_ADD, state->controlFd, &control) != 0)
        throw std::runtime_error(std::strerror(errno));
    state->worker = std::thread([value = state] { value->run(); });
}

DeviceRetirementReactor::~DeviceRetirementReactor() { stop(); }

bool DeviceRetirementReactor::registerJob(DeviceRetirementJob job) noexcept {
    if (!state || state->stopping.load(std::memory_order_acquire)
            || job.completionFd < 0 || !job.ticket || !job.retainedBacking)
        return false;
    if (const auto wake = job.wakeTarget.lock()) job.ticket->setWakeTarget(wake);
    const auto registration = state->nextRegistration.fetch_add(1);
    epoll_event event{};
    event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLONESHOT;
    event.data.u64 = registration;
    try {
        const std::scoped_lock lock(state->mutex);
        if (state->stopping.load(std::memory_order_relaxed)) return false;
        const auto [iterator, inserted] = state->jobs.emplace(
            registration, std::move(job));
        if (!inserted) return false;
        if (::epoll_ctl(state->epollFd, EPOLL_CTL_ADD,
                iterator->second.completionFd, &event) != 0) {
            state->jobs.erase(iterator);
            return false;
        }
    } catch (...) {
        return false;
    }
    state->wake();
    return true;
}

std::optional<DeviceRetirementReactor::ReservedJob>
DeviceRetirementReactor::reserveJob(DeviceRetirementJob job) noexcept {
    if (!state || state->stopping.load(std::memory_order_acquire) || !job.ticket)
        return std::nullopt;
    try {
        const auto registration = state->nextRegistration.fetch_add(1);
        ReservedReactorJobStorage::Map temporary;
        job.completionFd = -1;
        job.retainedBacking.reset();
        temporary.emplace(registration, std::move(job));
        auto storage = std::make_shared<ReservedReactorJobStorage>();
        storage->node = temporary.extract(registration);
        ReservedJob result;
        result.storage = std::move(storage);
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

bool DeviceRetirementReactor::registerReservedJob(ReservedJob& reserved,
        int completionFd, std::shared_ptr<void> retainedBacking) noexcept {
    if (!state || state->stopping.load(std::memory_order_acquire)
            || completionFd < 0 || !retainedBacking || !reserved.storage)
        return false;
    auto storage = std::static_pointer_cast<ReservedReactorJobStorage>(
        reserved.storage);
    if (!storage || storage->node.empty()) return false;
    auto& job = storage->node.mapped();
    job.completionFd = completionFd;
    job.retainedBacking = std::move(retainedBacking);
    if (const auto wake = job.wakeTarget.lock()) job.ticket->setWakeTarget(wake);
    epoll_event event{};
    event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLONESHOT;
    event.data.u64 = storage->node.key();
    const std::scoped_lock lock(state->mutex);
    if (state->stopping.load(std::memory_order_relaxed)) return false;
    auto inserted = state->jobs.insert(std::move(storage->node));
    if (!inserted.inserted) {
        storage->node = std::move(inserted.node);
        storage->node.mapped().completionFd = -1;
        storage->node.mapped().retainedBacking.reset();
        return false;
    }
    if (::epoll_ctl(state->epollFd, EPOLL_CTL_ADD,
            inserted.position->second.completionFd, &event) != 0) {
        storage->node = state->jobs.extract(inserted.position);
        storage->node.mapped().completionFd = -1;
        storage->node.mapped().retainedBacking.reset();
        return false;
    }
    reserved.storage.reset();
    state->wake();
    return true;
}

bool DeviceRetirementReactor::completeImmediately(DeviceRetirementJob job) noexcept {
    if (!state || state->stopping.load(std::memory_order_acquire)
            || job.completionFd != -1 || !job.ticket || !job.retainedBacking)
        return false;
    if (const auto wake = job.wakeTarget.lock()) job.ticket->setWakeTarget(wake);
    auto result = DeviceRetirementTicketState::Completed;
    if (job.retireBacking) {
        try {
            job.retireBacking(result);
        } catch (...) {
            result = DeviceRetirementTicketState::FailedPermanent;
            const std::scoped_lock lock(state->mutex);
            if (!state->retainEventSourceLostLocked(job.retainedBacking))
                return false;
        }
    }
    job.ticket->value.store(result, std::memory_order_release);
    job.ticket->notifyWakeTarget();
    return true;
}

bool DeviceRetirementReactor::retainEventSourceLost(
        std::shared_ptr<void> backing) noexcept {
    if (!state || !backing) return false;
    const std::scoped_lock lock(state->mutex);
    return state->retainEventSourceLostLocked(std::move(backing));
}

void DeviceRetirementReactor::stop() noexcept {
    if (!state || state->stopping.exchange(true)) return;
    state->wake();
    if (state->worker.joinable()) state->worker.join();
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
void DeviceRetirementReactor::setNextRegistrationForTesting(
        uint64_t value) noexcept {
    if (state) state->nextRegistration.store(value, std::memory_order_relaxed);
}
#endif

size_t DeviceRetirementReactor::pendingCount() const noexcept {
    if (!state) return 0;
    const std::scoped_lock lock(state->mutex);
    return state->jobs.size();
}

size_t DeviceRetirementReactor::eventSourceLostCount() const noexcept {
    if (!state) return 0;
    const std::scoped_lock lock(state->mutex);
    return state->eventSourceLostUsed;
}

}
