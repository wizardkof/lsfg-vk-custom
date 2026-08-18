/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "tools/interop_sync_fd_probe.hpp"

#include <cassert>
#include <string>
#include <vector>

using namespace lsfgvk::cli::interop_sync_fd_probe;

int main() {
    assert(std::string(failureName(Failure::ImportB)) ==
        "A_SYNC_FD_IMPORT_ON_B_FAILED");
    assert(std::string(failureName(Failure::SubmitFinal)) ==
        "A_FINAL_SUBMIT_FAILED");

    const std::vector<std::string> gpuChain{
        "A_SUBMIT",
        "EXPORT_AB_FD",
        "IMPORT_B_TEMPORARY",
        "B_SUBMIT",
        "EXPORT_BA_FD",
        "IMPORT_A_TEMPORARY",
        "A_FINAL_SUBMIT",
        "HOST_WAIT_FINAL_READBACK",
    };
    assert(!hasCrossDeviceHostWait(gpuChain));
    assert(hasCrossDeviceHostWait({
        "A_SUBMIT",
        "HOST_WAIT_A",
        "B_SUBMIT",
        "A_FINAL_SUBMIT",
    }));
    assert(hasCrossDeviceHostWait({
        "A_SUBMIT",
        "WAIT_A_BEFORE_B_SUBMIT",
        "B_SUBMIT",
        "A_FINAL_SUBMIT",
    }));
    assert(!hasCrossDeviceHostWait({
        "A_SUBMIT",
        "B_SUBMIT",
        "A_FINAL_SUBMIT",
        "HOST_WAIT_AFTER_FINAL",
    }));

    std::string error;
    auto options = parse({
        "--allocator", "/dev/dri/renderD129",
        "--device-a", "0",
        "--device-b", "1",
    }, error);
    assert(options && options->deviceA == 0 && options->deviceB == 1);
    assert(options->allocator == "/dev/dri/renderD129");
    assert(!parse({"--device-a", "0", "--device-b", "1"}, error));
    assert(!parse({
        "--allocator", "x",
        "--device-a", "1",
        "--device-b", "1",
    }, error));
}
