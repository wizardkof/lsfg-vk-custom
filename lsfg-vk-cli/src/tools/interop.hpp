/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>
#include <vulkan/vulkan_core.h>

namespace lsfgvk::cli::interop {
    [[nodiscard]] std::string formatHandleTypes(
        VkFlags flags);
    [[nodiscard]] bool dmaBufImportCandidate(bool firstImportable,
        bool secondImportable);
    [[nodiscard]] bool syncFdBridgeCandidate(bool firstExportable,
        bool secondImportable);
    int run();
}
