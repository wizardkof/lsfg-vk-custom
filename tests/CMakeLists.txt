# Regression tests for the Fixed/Adaptive integration.
#
# These are intentionally lightweight unit tests. They do not install or load
# the Vulkan layer and do not require Lossless.dll at runtime.

find_package(Threads REQUIRED)

add_executable(lsfg-vk-test-d3b3-production-state
    d3b3_production_state_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-test-d3b3-production-state PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-d3b3-production-state PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-d3b3-production-state PRIVATE
    lsfg-vk-common lsfg-vk-backend)
add_test(NAME lsfg-vk-test-d3b3-production-state
    COMMAND lsfg-vk-test-d3b3-production-state)

function(lsfgvk_add_layer_unit_test target test_source implementation_source)
    add_executable(${target}
        "${test_source}"
        "${implementation_source}")

    target_compile_options(${target} PRIVATE
        # CMake defines NDEBUG for Release builds. These regression tests use
        # assert(), so explicitly keep their checks enabled in CI as well.
        -UNDEBUG
        -Wno-unknown-warning-option)

    add_test(NAME ${target} COMMAND ${target})
endfunction()

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-fixed-frame-scheduler
    fixed_frame_scheduler_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/fixed_frame_scheduler.cpp")

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-fixed-output-pacer
    fixed_output_pacer_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/fixed_output_pacer.cpp")

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-fixed-present-mode
    fixed_present_mode_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/fixed_present_mode.cpp")

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-virtual-swapchain-state
    virtual_swapchain_state_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/virtual_swapchain_state.cpp")
target_link_libraries(lsfg-vk-test-virtual-swapchain-state
    PRIVATE Threads::Threads)

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-virtual-swapchain-image-spec
    virtual_swapchain_image_spec_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/virtual_swapchain_image_spec.cpp")
target_link_libraries(lsfg-vk-test-virtual-swapchain-image-spec
    PRIVATE lsfg-vk-common)

add_executable(lsfg-vk-test-runtime-generate-diagnostic-state
    runtime_generate_diagnostic_state_test.cpp)
target_include_directories(lsfg-vk-test-runtime-generate-diagnostic-state PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-backend/src")
target_compile_options(lsfg-vk-test-runtime-generate-diagnostic-state PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)
add_test(
    NAME lsfg-vk-test-runtime-generate-diagnostic-state
    COMMAND lsfg-vk-test-runtime-generate-diagnostic-state)

add_executable(lsfg-vk-test-temporal-source-contract
    temporal_source_contract_test.cpp)
target_compile_options(lsfg-vk-test-temporal-source-contract PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-temporal-source-contract PRIVATE lsfg-vk-backend)
add_test(NAME lsfg-vk-test-temporal-source-contract
    COMMAND lsfg-vk-test-temporal-source-contract)

add_executable(lsfg-vk-test-runtime-operation-authority
    runtime_operation_authority_test.cpp)
target_compile_options(lsfg-vk-test-runtime-operation-authority PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-runtime-operation-authority PRIVATE lsfg-vk-backend)
add_test(NAME lsfg-vk-test-runtime-operation-authority
    COMMAND lsfg-vk-test-runtime-operation-authority)

add_executable(lsfg-vk-test-generated-output-return-diagnostic
    generated_output_return_diagnostic_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/generated_output_return_diagnostic.cpp")
target_include_directories(lsfg-vk-test-generated-output-return-diagnostic PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-generated-output-return-diagnostic PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-generated-output-return-diagnostic PRIVATE
    lsfg-vk-common lsfg-vk-backend)
add_test(NAME lsfg-vk-test-generated-output-return-diagnostic
    COMMAND lsfg-vk-test-generated-output-return-diagnostic)

add_executable(lsfg-vk-test-runtime-foreign-handoff
    runtime_foreign_handoff_test.cpp)
target_compile_options(lsfg-vk-test-runtime-foreign-handoff PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-runtime-foreign-handoff PRIVATE lsfg-vk-common)
add_test(NAME lsfg-vk-test-runtime-foreign-handoff
    COMMAND lsfg-vk-test-runtime-foreign-handoff)

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-d3b1-present-path
    d3b1_present_path_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b1_present_path.cpp")
target_include_directories(lsfg-vk-test-d3b1-present-path PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_link_libraries(lsfg-vk-test-d3b1-present-path PRIVATE lsfg-vk-common)

lsfgvk_add_layer_unit_test(
    lsfg-vk-test-d3b2-insertion-path
    d3b2_insertion_path_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-test-d3b2-insertion-path PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_link_libraries(lsfg-vk-test-d3b2-insertion-path PRIVATE lsfg-vk-common)

add_executable(lsfg-vk-test-d3b3-production-seams
    d3b3_production_seams_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_compile_options(lsfg-vk-test-d3b3-production-seams PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
add_test(NAME lsfg-vk-test-d3b3-production-seams
    COMMAND lsfg-vk-test-d3b3-production-seams)
target_include_directories(lsfg-vk-test-d3b3-production-seams PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_link_libraries(lsfg-vk-test-d3b3-production-seams PRIVATE
    lsfg-vk-common lsfg-vk-backend)

add_executable(lsfg-vk-test-d3b3-normal-present-adapter
    d3b3_normal_present_adapter_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_normal_present_adapter.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_runtime.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-test-d3b3-normal-present-adapter PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-d3b3-normal-present-adapter PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-d3b3-normal-present-adapter PRIVATE
    lsfg-vk-common lsfg-vk-backend)
add_test(NAME lsfg-vk-test-d3b3-normal-present-adapter
    COMMAND lsfg-vk-test-d3b3-normal-present-adapter)

add_executable(lsfg-vk-test-graphics-final-queue graphics_final_queue_test.cpp)
target_include_directories(lsfg-vk-test-graphics-final-queue PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-graphics-final-queue PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
add_test(NAME lsfg-vk-test-graphics-final-queue
    COMMAND lsfg-vk-test-graphics-final-queue)

add_executable(lsfg-vk-test-config-frame-generation-mode
    config_frame_generation_mode_test.cpp)

target_link_libraries(lsfg-vk-test-config-frame-generation-mode
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-config-frame-generation-mode
    COMMAND lsfg-vk-test-config-frame-generation-mode)

add_executable(lsfg-vk-test-config-snapshot
    config_snapshot_test.cpp)

target_compile_options(lsfg-vk-test-config-snapshot PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-config-snapshot
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-config-snapshot
    COMMAND lsfg-vk-test-config-snapshot)

add_executable(lsfg-vk-test-physical-device-identity
    physical_device_identity_test.cpp)

target_compile_options(lsfg-vk-test-physical-device-identity PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-physical-device-identity
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-physical-device-identity
    COMMAND lsfg-vk-test-physical-device-identity)

add_executable(lsfg-vk-test-physical-device-diagnostics
    physical_device_diagnostics_test.cpp)

target_compile_options(lsfg-vk-test-physical-device-diagnostics PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-physical-device-diagnostics
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-physical-device-diagnostics
    COMMAND lsfg-vk-test-physical-device-diagnostics)

add_executable(lsfg-vk-test-runtime-device-pair
    runtime_device_pair_test.cpp)

target_compile_options(lsfg-vk-test-runtime-device-pair PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-runtime-device-pair
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-runtime-device-pair
    COMMAND lsfg-vk-test-runtime-device-pair)

add_executable(lsfg-vk-test-runtime-exchange-channel
    runtime_exchange_channel_test.cpp)

target_compile_options(lsfg-vk-test-runtime-exchange-channel PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-runtime-exchange-channel
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-runtime-exchange-channel
    COMMAND lsfg-vk-test-runtime-exchange-channel)

add_executable(lsfg-vk-test-runtime-payload-exchange
    runtime_payload_exchange_test.cpp)
target_compile_options(lsfg-vk-test-runtime-payload-exchange PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-runtime-payload-exchange PRIVATE lsfg-vk-common)
add_test(
    NAME lsfg-vk-test-runtime-payload-exchange
    COMMAND lsfg-vk-test-runtime-payload-exchange)

add_executable(lsfg-vk-test-runtime-image-observation
    runtime_image_observation_test.cpp)
target_compile_options(lsfg-vk-test-runtime-image-observation PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-runtime-image-observation PRIVATE lsfg-vk-common)
add_test(NAME lsfg-vk-test-runtime-image-observation
    COMMAND lsfg-vk-test-runtime-image-observation)

if(LSFGVK_BUILD_VK_LAYER)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(TEST_RUNTIME_GBM REQUIRED IMPORTED_TARGET gbm)

    add_executable(lsfg-vk-test-runtime-dma-buf-backing
        runtime_dma_buf_backing_test.cpp
        "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/runtime_dma_buf_backing.cpp")

    target_include_directories(lsfg-vk-test-runtime-dma-buf-backing PRIVATE
        "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")

    target_compile_options(lsfg-vk-test-runtime-dma-buf-backing PRIVATE
        -UNDEBUG
        -Wno-unknown-warning-option)

    target_link_libraries(lsfg-vk-test-runtime-dma-buf-backing PRIVATE
        lsfg-vk-common PkgConfig::TEST_RUNTIME_GBM)

    add_test(
        NAME lsfg-vk-test-runtime-dma-buf-backing
        COMMAND lsfg-vk-test-runtime-dma-buf-backing)
endif()

add_executable(lsfg-vk-test-external-capabilities
    external_capabilities_test.cpp)

target_compile_options(lsfg-vk-test-external-capabilities PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-external-capabilities
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-external-capabilities
    COMMAND lsfg-vk-test-external-capabilities)

add_executable(lsfg-vk-test-external-image-contract
    external_image_contract_test.cpp)

target_compile_options(lsfg-vk-test-external-image-contract PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-external-image-contract
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-external-image-contract
    COMMAND lsfg-vk-test-external-image-contract)

add_executable(lsfg-vk-test-external-image-transport
    external_image_transport_test.cpp)
target_compile_options(lsfg-vk-test-external-image-transport PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-external-image-transport PRIVATE lsfg-vk-common)
add_test(NAME lsfg-vk-test-external-image-transport
    COMMAND lsfg-vk-test-external-image-transport)

add_executable(lsfg-vk-test-exchange-image-sync
    exchange_image_sync_test.cpp)

target_compile_options(lsfg-vk-test-exchange-image-sync PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-exchange-image-sync
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-exchange-image-sync
    COMMAND lsfg-vk-test-exchange-image-sync)

add_executable(lsfg-vk-test-owned-fd
    owned_fd_test.cpp)

target_compile_options(lsfg-vk-test-owned-fd PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-owned-fd
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-owned-fd
    COMMAND lsfg-vk-test-owned-fd)

if(LSFGVK_BUILD_CLI)
add_executable(lsfg-vk-test-cli-devices
        cli_devices_test.cpp
        "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src/tools/devices.cpp")

    target_compile_options(lsfg-vk-test-cli-devices PRIVATE
        -UNDEBUG
        -Wno-unknown-warning-option)

    target_include_directories(lsfg-vk-test-cli-devices PRIVATE
        "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src")

    target_link_libraries(lsfg-vk-test-cli-devices
        PRIVATE lsfg-vk-common)

add_test(NAME lsfg-vk-test-cli-devices
    COMMAND lsfg-vk-test-cli-devices)

add_executable(lsfg-vk-test-cli-interop
    cli_interop_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src/tools/interop.cpp")

target_include_directories(lsfg-vk-test-cli-interop PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src")

target_compile_options(lsfg-vk-test-cli-interop PRIVATE
    -UNDEBUG
    -Wno-unknown-warning-option)

target_link_libraries(lsfg-vk-test-cli-interop PRIVATE lsfg-vk-common)

add_test(NAME lsfg-vk-test-cli-interop
    COMMAND lsfg-vk-test-cli-interop)

add_executable(lsfg-vk-test-interop-buffer-probe
    interop_buffer_probe_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src/tools/interop_buffer_probe.cpp")
find_package(PkgConfig REQUIRED)
pkg_check_modules(TEST_GBM REQUIRED IMPORTED_TARGET gbm)
target_include_directories(lsfg-vk-test-interop-buffer-probe PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src")
target_compile_options(lsfg-vk-test-interop-buffer-probe PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-interop-buffer-probe PRIVATE
    lsfg-vk-common PkgConfig::TEST_GBM)
add_test(NAME lsfg-vk-test-interop-buffer-probe
    COMMAND lsfg-vk-test-interop-buffer-probe)

add_executable(lsfg-vk-test-external-buffer-transport
    external_buffer_transport_test.cpp)
target_compile_options(lsfg-vk-test-external-buffer-transport PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-external-buffer-transport PRIVATE
    lsfg-vk-common)
add_test(NAME lsfg-vk-test-external-buffer-transport
    COMMAND lsfg-vk-test-external-buffer-transport)

add_executable(lsfg-vk-test-external-semaphore-sync
    external_semaphore_sync_test.cpp)
target_compile_options(lsfg-vk-test-external-semaphore-sync PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-external-semaphore-sync PRIVATE
    lsfg-vk-common)
add_test(NAME lsfg-vk-test-external-semaphore-sync
    COMMAND lsfg-vk-test-external-semaphore-sync)

add_executable(lsfg-vk-test-shadow-b-return-execution-harness
    shadow_b_return_execution_harness_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/generated_output_return_diagnostic.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_core_owner.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_normal_present_adapter.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_runtime.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-test-shadow-b-return-execution-harness PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-shadow-b-return-execution-harness PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-shadow-b-return-execution-harness PRIVATE
    lsfg-vk-common lsfg-vk-backend)
add_test(NAME lsfg-vk-test-shadow-b-return-execution-harness
    COMMAND lsfg-vk-test-shadow-b-return-execution-harness)

add_executable(lsfg-vk-test-d3b3-production-core-runner
    d3b3_production_core_runner_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/generated_output_return_diagnostic.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_core_owner.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_normal_present_adapter.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_runtime.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-test-d3b3-production-core-runner PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-test-d3b3-production-core-runner PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-d3b3-production-core-runner PRIVATE
    lsfg-vk-common lsfg-vk-backend)
add_test(NAME lsfg-vk-test-d3b3-production-core-runner
    COMMAND lsfg-vk-test-d3b3-production-core-runner)

add_executable(lsfg-vk-test-d3b3-hardware-runner-device-selector
    d3b3_hardware_runner_device_selector_test.cpp)
target_compile_options(lsfg-vk-test-d3b3-hardware-runner-device-selector PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-d3b3-hardware-runner-device-selector PRIVATE
    lsfg-vk-common)
add_test(NAME lsfg-vk-test-d3b3-hardware-runner-device-selector
    COMMAND lsfg-vk-test-d3b3-hardware-runner-device-selector)

# A6A5H is an explicitly invoked real-device executable.  It is built with
# the normal test/tool graph but is intentionally not a default CTest test:
# it requires live AMD->NVIDIA hardware and LSFGVK_DLL_PATH.
add_executable(lsfg-vk-d3b3-production-core-hardware-runner
    d3b3_production_core_hardware_runner.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/runtime_dma_buf_backing.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/generated_output_return_diagnostic.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_core_owner.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_runtime.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b3_production_seams.cpp"
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src/d3b2_insertion_path.cpp")
target_include_directories(lsfg-vk-d3b3-production-core-hardware-runner PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-layer/src")
target_compile_options(lsfg-vk-d3b3-production-core-hardware-runner PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-d3b3-production-core-hardware-runner PRIVATE
    lsfg-vk-common lsfg-vk-backend PkgConfig::TEST_GBM)

add_executable(lsfg-vk-test-interop-sync-fd-probe
    interop_sync_fd_probe_test.cpp
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src/tools/interop_sync_fd_probe.cpp")
target_include_directories(lsfg-vk-test-interop-sync-fd-probe PRIVATE
    "${PROJECT_SOURCE_DIR}/lsfg-vk-cli/src")
target_compile_options(lsfg-vk-test-interop-sync-fd-probe PRIVATE
    -UNDEBUG -Wno-unknown-warning-option)
target_link_libraries(lsfg-vk-test-interop-sync-fd-probe PRIVATE
    lsfg-vk-common PkgConfig::TEST_GBM)
add_test(NAME lsfg-vk-test-interop-sync-fd-probe
    COMMAND lsfg-vk-test-interop-sync-fd-probe)
    add_test(NAME lsfg-vk-test-cli-help
        COMMAND lsfg-vk-cli --help)
    add_test(NAME lsfg-vk-test-cli-devices-help
        COMMAND lsfg-vk-cli devices --help)
    add_test(NAME lsfg-vk-test-cli-devices-invalid-argument
        COMMAND lsfg-vk-cli devices invalid)
    set_tests_properties(lsfg-vk-test-cli-devices-invalid-argument
        PROPERTIES WILL_FAIL TRUE)
    add_test(NAME lsfg-vk-test-cli-validate-help
        COMMAND lsfg-vk-cli validate --help)
    add_test(NAME lsfg-vk-test-cli-benchmark-help
        COMMAND lsfg-vk-cli benchmark --help)
    add_test(NAME lsfg-vk-test-cli-debug-help
        COMMAND lsfg-vk-cli debug --help)
endif()

add_test(NAME lsfg-vk-test-documentation-packaging-consistency
    COMMAND ${CMAKE_COMMAND} -P
        "${CMAKE_CURRENT_SOURCE_DIR}/documentation_packaging_consistency.cmake")

if(LSFGVK_BUILD_UI)
    find_package(Qt6 REQUIRED COMPONENTS Core Qml)

    add_executable(lsfg-vk-test-ui-language-manager
        ui_language_manager_test.cpp
        "${PROJECT_SOURCE_DIR}/lsfg-vk-ui/src/language_manager.cpp")

    set_target_properties(lsfg-vk-test-ui-language-manager PROPERTIES
        AUTOMOC ON)

    target_compile_options(lsfg-vk-test-ui-language-manager PRIVATE
        -UNDEBUG
        -Wno-unknown-warning-option)

    target_include_directories(lsfg-vk-test-ui-language-manager PRIVATE
        "${PROJECT_SOURCE_DIR}/lsfg-vk-ui/src")

    target_link_libraries(lsfg-vk-test-ui-language-manager
        PRIVATE Qt6::Core
        PRIVATE Qt6::Qml)

    add_test(
        NAME lsfg-vk-test-ui-language-manager
        COMMAND lsfg-vk-test-ui-language-manager
            "$<TARGET_FILE_DIR:lsfg-vk-ui>")

    add_test(
        NAME lsfg-vk-test-ui-translation-consistency
        COMMAND ${CMAKE_COMMAND}
            -DROOT=${PROJECT_SOURCE_DIR}
            -DUI_BINARY_DIR=$<TARGET_FILE_DIR:lsfg-vk-ui>
            -P "${CMAKE_CURRENT_SOURCE_DIR}/ui_translation_consistency.cmake")
endif()
