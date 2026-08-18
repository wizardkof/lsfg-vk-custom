set(CLI_SOURCES
    "src/tools/benchmark.cpp"
    "src/tools/debug.cpp"
    "src/tools/devices.cpp"
    "src/tools/interop.cpp"
    "src/tools/interop_buffer_probe.cpp"
    "src/tools/interop_sync_fd_probe.cpp"
    "src/tools/validate.cpp"
    "src/main.cpp")

add_executable(lsfg-vk-cli ${CLI_SOURCES})

target_link_libraries(lsfg-vk-cli
    PUBLIC lsfg-vk-common
    PUBLIC lsfg-vk-backend)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GBM REQUIRED IMPORTED_TARGET gbm)
target_link_libraries(lsfg-vk-cli PRIVATE PkgConfig::GBM)

target_compile_options(lsfg-vk-cli PRIVATE
    -Wno-unknown-warning-option
    -Wno-unsafe-buffer-usage) # CLI parsing

install(TARGETS lsfg-vk-cli
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
