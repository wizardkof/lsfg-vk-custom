/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "tools/interop_buffer_probe.hpp"
#include <cassert>
#include <string>
#include <vector>
using namespace lsfgvk::cli::interop_buffer_probe;
int main(){
    assert(usableMemoryTypes(0b1010,0b1100)==0b1000);
    assert(usableMemoryTypes(0b0010,0b1100)==0);
    const std::vector<uint32_t> flags{1,3,1,3};
    assert(selectMemoryType(0b1010,flags,1,2)==1);
    assert(selectMemoryType(0b0101,flags,1,2)==0);
    assert(!selectMemoryType(0b0100,flags,2));
    const uint32_t good[]{0xA5A5A5A5,0xA5A5A5A5,0xA5A5A5A5};
    assert(verifyPattern(good,3,0xA5A5A5A5).matches);
    const uint32_t bad[]{0x5A5A5A5A,0,0x5A5A5A5A};
    auto mismatch=verifyPattern(bad,3,0x5A5A5A5A); assert(!mismatch.matches&&mismatch.mismatchIndex==1);
    std::string error;
    auto opts=parse({"--allocator","/dev/dri/renderD129","--device-a","0","--device-b","1"},error);
    assert(opts&&opts->allocator=="/dev/dri/renderD129"&&opts->deviceA==0&&opts->deviceB==1);
    assert(!parse({"--allocator","x","--device-a","1","--device-b","1"},error));
    assert(!parse({"--device-a","0","--device-b","1"},error));
}
