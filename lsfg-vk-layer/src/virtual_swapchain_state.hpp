/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <vector>

namespace lsfgvk::layer {

class VirtualSwapchainState {
public:
    using Duration = std::chrono::nanoseconds;

    struct Present {
        uint32_t imageIndex{};
        uint64_t serial{};
    };

    class PreparedPresent {
    public:
        PreparedPresent() = default;
        PreparedPresent(const PreparedPresent&) = delete;
        PreparedPresent& operator=(const PreparedPresent&) = delete;
        PreparedPresent(PreparedPresent&&) noexcept = default;
        PreparedPresent& operator=(PreparedPresent&&) noexcept = default;
        [[nodiscard]] bool valid() const noexcept { return !node.empty(); }
    private:
        std::list<Present> node;
        friend class VirtualSwapchainState;
    };
    struct PreparedBatchPresent {
        uint32_t imageIndex{};
        uint64_t serial{};
        bool valid{};
    };

    explicit VirtualSwapchainState(size_t imageCount);

    [[nodiscard]] size_t imageCount() const noexcept;
    [[nodiscard]] std::optional<uint32_t> tryAcquire();
    [[nodiscard]] std::optional<uint32_t> waitAcquire(Duration timeout);

    [[nodiscard]] bool release(uint32_t imageIndex);
    [[nodiscard]] std::optional<PreparedPresent> preparePresent(
        uint32_t imageIndex, uint64_t serial);
    [[nodiscard]] bool commitPreparedPresent(PreparedPresent&) noexcept;
    [[nodiscard]] std::optional<PreparedBatchPresent> prepareBatchPresent(
        uint32_t imageIndex, uint64_t serial) noexcept;
    [[nodiscard]] bool commitBatchPresent(PreparedBatchPresent&) noexcept;
    [[nodiscard]] bool abortBatchPresent(
        PreparedBatchPresent&, bool bridgeAccepted) noexcept;
    [[nodiscard]] bool queuePresent(uint32_t imageIndex, uint64_t serial);
    [[nodiscard]] std::optional<Present> waitPresent(Duration timeout);
    [[nodiscard]] uint64_t wakeGeneration() const noexcept;
    void waitForWake(uint64_t observedGeneration, Duration timeout);
    void waitForWake(Duration timeout);
    void wake();
    [[nodiscard]] bool complete(uint32_t imageIndex);

    void stop();
    [[nodiscard]] bool stopped() const;

private:
    enum class ImageState : uint8_t {
        Available,
        Acquired,
        BatchPrepared,
        Queued,
        Presenting
    };

    [[nodiscard]] std::optional<uint32_t> acquireUnlocked();

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::vector<ImageState> images;
    std::list<Present> presents;
    uint64_t wakeCounter{};
    bool stopping{};
};

}
