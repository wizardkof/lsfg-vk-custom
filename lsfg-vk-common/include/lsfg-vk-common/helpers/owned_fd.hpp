/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <utility>

#include <unistd.h>

namespace ls {
    /// Unique owner for a POSIX file descriptor.
    class OwnedFd {
    public:
        OwnedFd() noexcept = default;
        explicit OwnedFd(int fd) noexcept : fd(fd) {}

        OwnedFd(const OwnedFd&) = delete;
        OwnedFd& operator=(const OwnedFd&) = delete;

        OwnedFd(OwnedFd&& other) noexcept : fd(other.release()) {}
        OwnedFd& operator=(OwnedFd&& other) noexcept {
            if (this != &other)
                this->reset(other.release());
            return *this;
        }

        ~OwnedFd() { this->reset(); }

        [[nodiscard]] int get() const noexcept { return this->fd; }

        [[nodiscard]] int release() noexcept {
            return std::exchange(this->fd, -1);
        }

        void reset(int replacement = -1) noexcept {
            if (this->fd == replacement)
                return;
            if (this->fd >= 0)
                ::close(this->fd);
            this->fd = replacement;
        }

        explicit operator bool() const noexcept { return this->fd >= 0; }

    private:
        int fd{-1};
    };
}
