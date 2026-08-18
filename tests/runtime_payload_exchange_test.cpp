/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <cassert>

int main() {
    using namespace vk;
    assert(validRuntimePayloadExchangeOrder({
        "A_WRITE", "A_SIGNAL_EXPORT", "B_IMPORT_WAIT", "B_OBSERVE_A",
        "B_WRITE", "B_SIGNAL_EXPORT", "A_IMPORT_WAIT", "A_OBSERVE_B",
        "FINAL_SUBMIT", "HOST_WAIT"}));
    assert(!validRuntimePayloadExchangeOrder({
        "A_WRITE", "HOST_WAIT", "A_SIGNAL_EXPORT"}));
    assert(verifyRuntimePayloadPattern({0xA5A5A5A5, 0xA5A5A5A5}, 0xA5A5A5A5));
    assert(!verifyRuntimePayloadPattern({0xA5A5A5A5, 0x5A5A5A5A}, 0xA5A5A5A5));
}
