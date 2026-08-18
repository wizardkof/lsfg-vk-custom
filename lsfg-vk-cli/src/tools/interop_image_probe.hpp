/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace lsfgvk::cli::interop_image_probe {
    struct Options { std::string allocator; size_t deviceA{}; size_t deviceB{}; };
    [[nodiscard]] std::optional<Options> parse(const std::vector<std::string>&,
        std::string& error);
    int run(const Options&);
}
