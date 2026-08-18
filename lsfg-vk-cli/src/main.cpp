/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/benchmark.hpp"
#include "tools/debug.hpp"
#include "tools/devices.hpp"
#include "tools/interop.hpp"
#include "tools/interop_buffer_probe.hpp"
#include "tools/validate.hpp"

#include <array>
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <getopt.h> // NOLINT (IWYU)
#include <bits/getopt_core.h>
#include <bits/getopt_ext.h>

using namespace lsfgvk::cli;

namespace {
    constexpr int OPT_HELP = 1000;

    /// print usage information
    void usage(const std::string& prog, const std::string& command = {}) {
        std::cerr << "Inspect, validate, benchmark, and debug lsfg-vk.\n\n"
                     "USAGE:\n    " << prog;

        if (command == "validate")
            std::cerr << " validate [OPTIONS]\n\n";
        else if (command == "benchmark")
            std::cerr << " benchmark [OPTIONS]\n\n";
        else if (command == "debug")
            std::cerr << " debug [OPTIONS] <folder>\n\n";
        else if (command == "devices")
            std::cerr << " devices [--help]\n\n";
        else if (command == "interop")
            std::cerr << " interop\n\n";
        else if (command == "interop-buffer-probe")
            std::cerr << " interop-buffer-probe --allocator PATH --device-a INDEX --device-b INDEX\n\n";
        else
            std::cerr << " <COMMAND> [OPTIONS] [ARGS]\n\n";

        std::cerr <<
R"(COMMANDS:
    validate    Validate a configuration file
    benchmark   Run a benchmark
    debug       Run lsfg-vk on a set of images
    devices     List Vulkan devices visible to this process
    interop     Query external buffer and SYNC_FD capabilities
    interop-buffer-probe  Probe cross-device DMA_BUF buffer import/use
    help        Show this help text

GLOBAL OPTIONS:
        --help                          Show this help text

SUBCOMMAND OPTIONS:

    validate
        -c, --config <PATH>             Optional path to the configuration file
            --help                      Show this help text

    devices
            --help                      Show this help text

    benchmark & debug
        -d, --dll <PATH>                Path to Lossless.dll
        -a, --allow-fp16                Allow FP16 acceleration
        -w, --width <INT>               Width of the input frames
        -h, --height <INT>              Height of the input frames
        -f, --flow <FLOAT>              Flow scale
        -m, --multiplier <INT>          Backend interpolation multiplier (2+)
        -p, --performance-mode          Use performance mode
        -g, --gpu <STRING>              GPU to use
            --help                      Show this help text

    benchmark
        -t, --duration <SECONDS>        Benchmark duration in seconds

    debug
        <folder>                        Path to the debug frames)" << '\n';
    }

    /// parse the validate command options
    [[noreturn]] void on_validate(int argc, char** argv, const std::string& prog) {
        validate::Options opts{};

        const std::array<option, 3> GETOPT {{
            { "config", required_argument, nullptr, 'c' },
            { "help",          no_argument, nullptr, OPT_HELP },
            { nullptr,         no_argument, nullptr,  0  }
        }};

        int c{0};
        while ((c = getopt_long(argc, argv, "c:", GETOPT.data(), nullptr)) != -1) {
            switch (c) {
                case 'c':
                    opts.config.emplace(optarg);
                    break;
                case OPT_HELP:
                    usage(prog, "validate");
                    std::exit(EXIT_SUCCESS);
                case '?':
                default:
                    usage(prog, "validate");
                    std::exit(EXIT_FAILURE);
            }
        }

        if (optind < argc) {
            usage(prog, "validate");
            std::exit(EXIT_FAILURE);
        }

        std::exit(validate::run(opts));
    }

    /// parse the devices command options
    [[noreturn]] void on_devices(int argc, char** argv, const std::string& prog) {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage(prog, "devices");
            std::exit(EXIT_SUCCESS);
        }
        if (argc != 1) {
            usage(prog, "devices");
            std::exit(EXIT_FAILURE);
        }

        std::exit(devices::run());
    }

    /// parse the benchmark command options
    [[noreturn]] void on_benchmark(int argc, char** argv, const std::string& prog) {
        benchmark::Options opts{};

        const std::array<option, 11> GETOPT {{
            { "dll",              required_argument, nullptr, 'd' },
            { "allow-fp16",       no_argument,       nullptr, 'a' },
            { "width",            required_argument, nullptr, 'w' },
            { "height",           required_argument, nullptr, 'h' },
            { "flow",             required_argument, nullptr, 'f' },
            { "multiplier",       required_argument, nullptr, 'm' },
            { "performance-mode",       no_argument, nullptr, 'p' },
            { "gpu",              required_argument, nullptr, 'g' },
            { "duration",         required_argument, nullptr, 't' },
            { "help",                   no_argument, nullptr, OPT_HELP },
            { nullptr,                  no_argument, nullptr,  0  }
        }};

        int c{0};
        while ((c = getopt_long(argc, argv, "d:aw:h:f:m:pg:t:", GETOPT.data(), nullptr)) != -1) {
            switch (c) {
                case 'd':
                    opts.dll.emplace(optarg);
                    break;
                case 'a':
                    opts.allow_fp16 = true;
                    break;
                case 'w':
                    opts.width = std::stoi(optarg);
                    break;
                case 'h':
                    opts.height = std::stoi(optarg);
                    break;
                case 'f':
                    opts.flow = std::stof(optarg);
                    break;
                case 'm':
                    opts.multiplier = std::stoi(optarg);
                    break;
                case 'p':
                    opts.performance_mode = true;
                    break;
                case 'g':
                    opts.gpu.emplace(optarg);
                    break;
                case 't':
                    opts.duration = std::stoi(optarg);
                    break;
                case OPT_HELP:
                    usage(prog, "benchmark");
                    std::exit(EXIT_SUCCESS);
                case '?':
                default:
                    usage(prog, "benchmark");
                    std::exit(EXIT_FAILURE);
            }
        }

        if (optind < argc) {
            usage(prog, "benchmark");
            std::exit(EXIT_FAILURE);
        }

        std::exit(benchmark::run(opts));
    }

    /// parse the debug command options
    [[noreturn]] void on_debug(int argc, char** argv, const std::string& prog) {
        debug::Options opts{};

        const std::array<option, 10> GETOPT {{
            { "dll",              required_argument, nullptr, 'd' },
            { "allow-fp16",       no_argument,       nullptr, 'a' },
            { "width",            required_argument, nullptr, 'w' },
            { "height",           required_argument, nullptr, 'h' },
            { "flow",             required_argument, nullptr, 'f' },
            { "multiplier",       required_argument, nullptr, 'm' },
            { "performance-mode",       no_argument, nullptr, 'p' },
            { "gpu",              required_argument, nullptr, 'g' },
            { "help",                   no_argument, nullptr, OPT_HELP },
            { nullptr,                  no_argument, nullptr,  0  }
        }};

        int c{0};
        while ((c = getopt_long(argc, argv, "d:aw:h:f:m:pg:", GETOPT.data(), nullptr)) != -1) {
            switch (c) {
                case 'd':
                    opts.dll.emplace(optarg);
                    break;
                case 'a':
                    opts.allow_fp16 = true;
                    break;
                case 'w':
                    opts.width = std::stoi(optarg);
                    break;
                case 'h':
                    opts.height = std::stoi(optarg);
                    break;
                case 'f':
                    opts.flow = std::stof(optarg);
                    break;
                case 'm':
                    opts.multiplier = std::stoi(optarg);
                    break;
                case 'p':
                    opts.performance_mode = true;
                    break;
                case 'g':
                    opts.gpu.emplace(optarg);
                    break;
                case OPT_HELP:
                    usage(prog, "debug");
                    std::exit(EXIT_SUCCESS);
                case '?':
                default:
                    usage(prog, "debug");
                    std::exit(EXIT_FAILURE);
            }
        }

        if ((optind + 1) != argc) {
            usage(prog, "debug");
            std::exit(EXIT_FAILURE);
        }

        opts.path = argv[optind];

        std::exit(debug::run(opts));
    }
}

int main(int argc, char** argv) {
    const std::string prog{argv[0]};

    if (argc < 2) {
        usage(prog);
        return EXIT_FAILURE;
    }

    const std::string command{argv[1]};
    if (command == "--help" || command == "help") {
        usage(prog);
        return EXIT_SUCCESS;
    }
    if (command == "validate")
        on_validate(argc - 1, argv + 1, prog);
    else if (command == "devices")
        on_devices(argc - 1, argv + 1, prog);
    else if (command == "interop")
        return lsfgvk::cli::interop::run();
    else if (command == "interop-buffer-probe") {
        std::vector<std::string> args;
        for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);
        std::string error;
        const auto options = interop_buffer_probe::parse(args, error);
        if (!options) { std::cerr << "error: " << error << "\n"; return EXIT_FAILURE; }
        return interop_buffer_probe::run(*options);
    }
    else if (command == "benchmark")
        on_benchmark(argc - 1, argv + 1, prog);
    else if (command == "debug")
        on_debug(argc - 1, argv + 1, prog);

    usage(prog);
    return EXIT_FAILURE;
}
