/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <linux/sync_file.h>
#include <sys/ioctl.h>

namespace lsfgvk::layer {

enum class SyncFileCompletionState : uint8_t {
    Pending,
    Completed,
    GpuError,
    FailedPermanent
};

[[nodiscard]] constexpr SyncFileCompletionState classifySyncFileCompletion(
        int status) noexcept {
    if (status < 0) return SyncFileCompletionState::GpuError;
    if (status == 1) return SyncFileCompletionState::Completed;
    if (status == 0) return SyncFileCompletionState::Pending;
    return SyncFileCompletionState::FailedPermanent;
}

[[nodiscard]] inline SyncFileCompletionState querySyncFileCompletion(
        int fd) noexcept {
    sync_file_info info{};
    if (fd < 0 || ::ioctl(fd, SYNC_IOC_FILE_INFO, &info) != 0)
        return SyncFileCompletionState::FailedPermanent;
    return classifySyncFileCompletion(info.status);
}

}
