set(ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

function(assert_contains rel needle)
    file(READ "${ROOT}/${rel}" content)
    string(FIND "${content}" "${needle}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "${rel} is missing expected text: ${needle}")
    endif()
endfunction()

function(assert_not_contains rel needle)
    file(READ "${ROOT}/${rel}" content)
    string(FIND "${content}" "${needle}" pos)
    if(NOT pos EQUAL -1)
        message(FATAL_ERROR "${rel} still contains stale text: ${needle}")
    endif()
endfunction()

assert_contains("README.md" "lsfg-vk-2.0.0-linux.tar.xz")
assert_not_contains("README.md" "lsfg-vk-2.0.0-x86_64.tar.xz")

assert_contains(
    ".github/workflows/flatpak.yml"
    [=[bundle: "org.freedesktop.Platform.VulkanLayer.lsfgvk_${{ matrix.version }}.flatpak"]=])
assert_not_contains(
    ".github/workflows/flatpak.yml"
    [=[org.freedesktop.Platform.VulkanLayer.lsfg_vk_${{ matrix.version }}.flatpak]=])

foreach(version IN ITEMS 23.08 24.08 25.08)
    assert_contains(
        "docs/Flatpak-Guide.md"
        "org.freedesktop.Platform.VulkanLayer.lsfgvk_${version}.flatpak")
endforeach()

assert_contains("docs/Flatpak-Guide.md" "flatpak run gay.pancake.lsfg-vk-ui")
assert_contains(
    "docs/Flatpak-Guide.md"
    [=[$HOME/.local/share/Steam/steamapps/common:ro]=])
assert_not_contains("docs/Flatpak-Guide.md" "gay.pancake.lsfg_vk_ui")
assert_not_contains(
    "docs/Flatpak-Guide.md"
    [=[/home/$USER/local/share/Steam]=])

assert_contains(
    "docs/Building-From-Source.md"
    "CLI help behavior, and documentation/packaging consistency")
