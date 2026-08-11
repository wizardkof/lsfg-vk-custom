# Regression tests for the Fixed/Adaptive integration.
#
# These are intentionally lightweight unit tests. They do not install or load
# the Vulkan layer and do not require Lossless.dll at runtime.

find_package(Threads REQUIRED)

function(lsfgvk_add_layer_unit_test target test_source implementation_source)
    add_executable(${target}
        "${test_source}"
        "${implementation_source}")

    target_compile_options(${target} PRIVATE
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

add_executable(lsfg-vk-test-config-frame-generation-mode
    config_frame_generation_mode_test.cpp)

target_link_libraries(lsfg-vk-test-config-frame-generation-mode
    PRIVATE lsfg-vk-common)

add_test(
    NAME lsfg-vk-test-config-frame-generation-mode
    COMMAND lsfg-vk-test-config-frame-generation-mode)
