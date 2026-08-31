set(LAYER_SOURCES
    "src/aborted_present_semantics.cpp"
    "src/adaptive_1x_prepared_logical_final.cpp"
    "src/adaptive_1x_preparation_reservation.cpp"
    "src/adaptive_1x_recovery_authority.cpp"
    "src/batch_application_present_bridge_authority.cpp"
    "src/batch_present_transaction.cpp"
    "src/borrowed_present_fence_registry.cpp"
    "src/acquire_fence_proxy_registry.cpp"
    "src/device_retirement_reactor.cpp"
    "src/d2_vulkan_shadow_runtime.cpp"
    "src/d3b3_per_swapchain_runtime.cpp"
    "src/d3b3_production_seams.cpp"
    "src/d3b3_normal_present_adapter.cpp"
    "src/d3b3_terminal_context.cpp"
    "src/d3b3_production_runtime.cpp"
    "src/d3b3_production_core_owner.cpp"
    "src/d3b2_insertion_path.cpp"
    "src/d3b1_present_path.cpp"
    "src/entrypoint.cpp"
    "src/fixed_frame_scheduler.cpp"
    "src/fixed_output_pacer.cpp"
    "src/fixed_present_mode.cpp"
    "src/generated_output_return_diagnostic.cpp"
    "src/instance.cpp"
    "src/present_batch_projection.cpp"
    "src/presented_physical_image_lease.cpp"
    "src/runtime_dma_buf_backing.cpp"
    "src/swapchain.cpp"
    "src/terminal_pending_gpu_operation.cpp"
    "src/virtual_swapchain_state.cpp"
    "src/virtual_swapchain_image_spec.cpp"
    "src/virtual_swapchain_runtime.cpp")

add_library(lsfg-vk-layer SHARED ${LAYER_SOURCES})

target_link_libraries(lsfg-vk-layer
    PUBLIC lsfg-vk-common
    PUBLIC lsfg-vk-backend)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LAYER_GBM REQUIRED IMPORTED_TARGET gbm)
target_link_libraries(lsfg-vk-layer PRIVATE PkgConfig::LAYER_GBM)

target_compile_options(lsfg-vk-layer PRIVATE
    -Wno-unknown-warning-option
    -Wno-unsafe-buffer-usage) # Array indexing

set_target_properties(lsfg-vk-layer PROPERTIES
    CXX_VISIBILITY_PRESET hidden)

configure_file(
    "VkLayer_LSFGVK_frame_generation.json.in"
    "VkLayer_LSFGVK_frame_generation.json"
    @ONLY)

install(TARGETS lsfg-vk-layer
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VkLayer_LSFGVK_frame_generation.json"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/vulkan/implicit_layer.d")
