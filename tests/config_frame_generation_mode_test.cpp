/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/configuration/config.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
    void expect(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void writeText(const std::filesystem::path& path, const std::string& text) {
        std::ofstream out(path);
        out << text;
    }
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "lsfg-vk-config-mode-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    try {
        const auto adaptive = dir / "adaptive.toml";
        writeText(adaptive, R"(version = 2
[global]
allow_fp16 = true
[[profile]]
name = "adaptive-1x"
multiplier = 1
frame_generation_mode = "adaptive"
target_fps = 0
)");
        ls::ConfigFile adaptiveConfig(adaptive);
        expect(adaptiveConfig.profiles().size() == 1, "adaptive profile missing");
        expect(adaptiveConfig.profiles().front().multiplier == 1, "adaptive 1x not parsed");
        expect(adaptiveConfig.profiles().front().frame_generation_mode == ls::FrameGenerationMode::Adaptive,
            "adaptive mode not parsed");

        const auto fixed = dir / "fixed.toml";
        writeText(fixed, R"(version = 2
[global]
allow_fp16 = true
[[profile]]
name = "fixed-240"
multiplier = 1
frame_generation_mode = "fixed"
target_fps = 240
)");
        ls::ConfigFile fixedConfig(fixed);
        expect(fixedConfig.profiles().front().frame_generation_mode == ls::FrameGenerationMode::Fixed,
            "fixed mode not parsed");
        expect(fixedConfig.profiles().front().target_fps == 240, "fixed target not parsed");

        const auto roundtrip = dir / "roundtrip.toml";
        fixedConfig.write(roundtrip);
        ls::ConfigFile roundtripConfig(roundtrip);
        expect(roundtripConfig.profiles().front().frame_generation_mode == ls::FrameGenerationMode::Fixed,
            "fixed mode not preserved by write/read");
        expect(roundtripConfig.profiles().front().target_fps == 240,
            "fixed target not preserved by write/read");

        const auto invalidFixed = dir / "invalid-fixed.toml";
        writeText(invalidFixed, R"(version = 2
[[profile]]
frame_generation_mode = "fixed"
target_fps = 0
)");
        bool invalidFixedRejected = false;
        try { ls::ConfigFile ignored(invalidFixed); }
        catch (const std::exception&) { invalidFixedRejected = true; }
        expect(invalidFixedRejected, "fixed target_fps=0 was not rejected");

        const auto invalidAdaptive = dir / "invalid-adaptive.toml";
        writeText(invalidAdaptive, R"(version = 2
[[profile]]
frame_generation_mode = "adaptive"
multiplier = 6
)");
        bool invalidAdaptiveRejected = false;
        try { ls::ConfigFile ignored(invalidAdaptive); }
        catch (const std::exception&) { invalidAdaptiveRejected = true; }
        expect(invalidAdaptiveRejected, "adaptive multiplier > 5 was not rejected");

        std::filesystem::remove_all(dir);
        std::cout << "All frame generation configuration tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(dir);
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
