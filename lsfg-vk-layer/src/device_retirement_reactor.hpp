#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

namespace lsfgvk::layer {

enum class DeviceRetirementTicketState : uint8_t {
    Pending,
    Completed,
    DeviceLost,
    GpuError,
    FailedPermanent
};

[[nodiscard]] constexpr DeviceRetirementTicketState classifySyncFileStatus(
        int status) noexcept {
    if (status < 0) return DeviceRetirementTicketState::GpuError;
    if (status == 1) return DeviceRetirementTicketState::Completed;
    if (status == 0) return DeviceRetirementTicketState::Pending;
    return DeviceRetirementTicketState::FailedPermanent;
}

class DeviceRetirementWakeTarget {
public:
    virtual ~DeviceRetirementWakeTarget() = default;
    virtual void notifyDeviceRetirement() noexcept = 0;
};

class DeviceRetirementTicket final {
public:
    [[nodiscard]] DeviceRetirementTicketState state() const noexcept {
        return value.load(std::memory_order_acquire);
    }
    void setWakeTarget(const std::shared_ptr<DeviceRetirementWakeTarget>& target) noexcept {
        const std::scoped_lock lock(wakeMutex);
        wakeTarget = target;
    }

private:
    friend class DeviceRetirementReactor;
    void notifyWakeTarget() noexcept {
        std::weak_ptr<DeviceRetirementWakeTarget> target;
        { const std::scoped_lock lock(wakeMutex); target = wakeTarget; }
        if (const auto value = target.lock()) value->notifyDeviceRetirement();
    }
    std::atomic<DeviceRetirementTicketState> value{
        DeviceRetirementTicketState::Pending};
    std::mutex wakeMutex;
    std::weak_ptr<DeviceRetirementWakeTarget> wakeTarget;
};

enum class TerminalWsiCompletionGateStatus : uint8_t {
    Pending,
    TerminalReady,
    WsiReady,
    DeviceLost,
    Failed
};

/// Separates logical terminal-submit completion from presentation-engine
/// completion. Ticket zero is the terminal domain; all later tickets retain
/// WSI resources and never delay the one-time TerminalReady transition.
class TerminalWsiCompletionGate final {
public:
    [[nodiscard]] TerminalWsiCompletionGateStatus observe(
        std::span<const std::shared_ptr<DeviceRetirementTicket>> tickets) const noexcept;
    void markTerminalRetired();
    [[nodiscard]] bool terminalRetired() const noexcept {
        return terminalRetiredValue;
    }
private:
    bool terminalRetiredValue{};
};

struct DeviceRetirementJob {
    uint64_t deviceIdentity{};
    uint64_t swapchainLifecycleIdentity{};
    uint64_t operationIdentity{};
    int completionFd{-1};
    std::shared_ptr<void> retainedBacking;
    std::shared_ptr<DeviceRetirementTicket> ticket;
    std::weak_ptr<DeviceRetirementWakeTarget> wakeTarget;
    std::function<void(DeviceRetirementTicketState)> retireBacking;
};

class DeviceRetirementReactor final {
public:
    class ReservedJob {
    public:
        ReservedJob() = default;
        ReservedJob(const ReservedJob&) = delete;
        ReservedJob& operator=(const ReservedJob&) = delete;
        ReservedJob(ReservedJob&&) noexcept = default;
        ReservedJob& operator=(ReservedJob&&) noexcept = default;
        [[nodiscard]] bool valid() const noexcept { return storage != nullptr; }
    private:
        std::shared_ptr<void> storage;
        friend class DeviceRetirementReactor;
    };
    DeviceRetirementReactor();
    ~DeviceRetirementReactor();

    DeviceRetirementReactor(const DeviceRetirementReactor&) = delete;
    DeviceRetirementReactor& operator=(const DeviceRetirementReactor&) = delete;

    [[nodiscard]] bool registerJob(DeviceRetirementJob job) noexcept;
    [[nodiscard]] std::optional<ReservedJob> reserveJob(
        DeviceRetirementJob templateJob) noexcept;
    [[nodiscard]] bool registerReservedJob(ReservedJob&, int completionFd,
        std::shared_ptr<void> retainedBacking) noexcept;
    [[nodiscard]] bool completeImmediately(DeviceRetirementJob job) noexcept;
    [[nodiscard]] bool retainEventSourceLost(
        std::shared_ptr<void> backing) noexcept;
    void stop() noexcept;
    [[nodiscard]] size_t pendingCount() const noexcept;
    [[nodiscard]] size_t eventSourceLostCount() const noexcept;
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    void setNextRegistrationForTesting(uint64_t) noexcept;
#endif

private:
    struct State;
    std::shared_ptr<State> state;
};

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
__attribute__((visibility("default")))
void markControlledRetirementFd(int fd) noexcept;
#endif

}
