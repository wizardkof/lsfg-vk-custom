/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/interop.hpp"

#include <cassert>

#include <string>

int main() {
    assert(lsfgvk::cli::interop::formatHandleTypes(
        static_cast<VkFlags>(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
        | static_cast<VkFlags>(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT))
        == "0x11 OPAQUE_FD SYNC_FD");
    assert(lsfgvk::cli::interop::formatHandleTypes(
        static_cast<VkFlags>(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT))
        == "0x10 SYNC_FD");
    assert(lsfgvk::cli::interop::dmaBufImportCandidate(true, true));
    assert(!lsfgvk::cli::interop::dmaBufImportCandidate(true, false));
    assert(!lsfgvk::cli::interop::dmaBufImportCandidate(false, true));
    assert(!lsfgvk::cli::interop::dmaBufImportCandidate(false, false));

    assert(lsfgvk::cli::interop::syncFdBridgeCandidate(true, true));
    assert(!lsfgvk::cli::interop::syncFdBridgeCandidate(true, false));
    assert(!lsfgvk::cli::interop::syncFdBridgeCandidate(false, true));
    assert(!lsfgvk::cli::interop::syncFdBridgeCandidate(false, false));
    return 0;
}
