#pragma once
#include <cstddef>
#include <cstdint>
namespace lsfgvk::common {
inline uint64_t fnv1a64(const uint8_t* bytes, size_t count) noexcept {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= 0x100000001b3ULL; }
    return hash;
}
}
