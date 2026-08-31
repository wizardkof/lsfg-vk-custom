/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <mutex>
#include <unistd.h>

namespace lsfgvk::layer {

/// Owns the original exported present SYNC_FD for the complete backing
/// lifetime. Observers may snapshot its numeric value, but only this object's
/// destructor closes it; the reactor receives and owns a separate dup().
class OwnedPresentSyncFd final {
public:
    OwnedPresentSyncFd() = default;
    ~OwnedPresentSyncFd() {
        const std::scoped_lock lock(mutex);
        if (fd >= 0) static_cast<void>(::close(fd));
    }
    OwnedPresentSyncFd(const OwnedPresentSyncFd&) = delete;
    OwnedPresentSyncFd& operator=(const OwnedPresentSyncFd&) = delete;

    void accept(int value) noexcept {
        const std::scoped_lock lock(mutex);
        fd = value;
    }
    [[nodiscard]] int snapshot() const noexcept {
        const std::scoped_lock lock(mutex);
        return fd;
    }

private:
    mutable std::mutex mutex;
    int fd{-1};
};

}
