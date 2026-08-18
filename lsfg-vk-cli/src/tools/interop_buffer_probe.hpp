/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace lsfgvk::cli::interop_buffer_probe {
    struct Options { std::string allocator; size_t deviceA{}; size_t deviceB{}; };
    struct PatternResult { bool matches{}; size_t mismatchIndex{}; };
    [[nodiscard]] uint32_t usableMemoryTypes(uint32_t fdBits, uint32_t bufferBits);
    [[nodiscard]] std::optional<uint32_t> selectMemoryType(
        uint32_t bits, const std::vector<uint32_t>& flags, uint32_t required,
        uint32_t preferred = 0);
    [[nodiscard]] PatternResult verifyPattern(
        const uint32_t* words, size_t count, uint32_t expected);
    [[nodiscard]] std::optional<Options> parse(const std::vector<std::string>& args,
        std::string& error);
    int run(const Options& options);
}
