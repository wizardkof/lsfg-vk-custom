/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
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

    explicit VirtualSwapchainState(size_t imageCount);

    [[nodiscard]] size_t imageCount() const noexcept;
    [[nodiscard]] std::optional<uint32_t> tryAcquire();
    [[nodiscard]] std::optional<uint32_t> waitAcquire(Duration timeout);

    [[nodiscard]] bool release(uint32_t imageIndex);
    [[nodiscard]] bool queuePresent(uint32_t imageIndex, uint64_t serial);
    [[nodiscard]] std::optional<Present> waitPresent(Duration timeout);
    [[nodiscard]] bool complete(uint32_t imageIndex);

    void stop();
    [[nodiscard]] bool stopped() const;

private:
    enum class ImageState : uint8_t {
        Available,
        Acquired,
        Queued,
        Presenting
    };

    [[nodiscard]] std::optional<uint32_t> acquireUnlocked();

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::vector<ImageState> images;
    std::deque<Present> presents;
    bool stopping{};
};

}
