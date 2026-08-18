/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lsfgvk::cli::interop_sync_fd_probe {
    struct Options { std::string allocator; size_t deviceA{}; size_t deviceB{}; };
    enum class Failure { None, Capability, ExtensionMissing, ExportCreate, ImportCreate,
        SubmitA, ExportAB, ImportB, SubmitB, ExportBA, ImportA, SubmitFinal,
        DataAB, DataBA, UnexpectedState };
    [[nodiscard]] const char* failureName(Failure) noexcept;
    [[nodiscard]] bool hasCrossDeviceHostWait(const std::vector<std::string>& events) noexcept;
    [[nodiscard]] std::optional<Options> parse(const std::vector<std::string>& args, std::string& error);
    int run(const Options& options);
}
