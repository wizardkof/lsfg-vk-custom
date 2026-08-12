if(NOT DEFINED ROOT)
    message(FATAL_ERROR "ROOT is required")
endif()
if(NOT DEFINED UI_BINARY_DIR)
    message(FATAL_ERROR "UI_BINARY_DIR is required")
endif()

function(assert_exists path)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Expected file does not exist: ${path}")
    endif()
endfunction()

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
        message(FATAL_ERROR "${rel} contains forbidden text: ${needle}")
    endif()
endfunction()

set(UI_QML "lsfg-vk-ui/rsc/UI.qml")
set(UI_CMAKE "lsfg-vk-ui/CMakeLists.txt")
set(LANGUAGE_MANAGER_CPP "lsfg-vk-ui/src/language_manager.cpp")
set(LANGUAGE_MANAGER_HPP "lsfg-vk-ui/src/language_manager.hpp")
set(PT_TS "lsfg-vk-ui/translations/lsfg-vk-ui_pt_BR.ts")
set(ES_TS "lsfg-vk-ui/translations/lsfg-vk-ui_es.ts")

assert_exists("${ROOT}/${PT_TS}")
assert_exists("${ROOT}/${ES_TS}")
assert_exists("${UI_BINARY_DIR}/lsfg-vk-ui_pt_BR.qm")
assert_exists("${UI_BINARY_DIR}/lsfg-vk-ui_es.qm")

foreach(source IN ITEMS
        "lsfg-vk Configuration Window"
        "Create New Profile"
        "Choose a profile name"
        "Global Settings"
        "Language"
        "System Default"
        "Frame Generation Mode"
        "Adaptive"
        "Fixed"
        "Target FPS"
        "Pacing Mode"
        "None"
        "Default")
    assert_contains("${UI_QML}" "qsTr(\"${source}\")")
endforeach()

assert_contains("${PT_TS}" "<translation>Configuração do lsfg-vk</translation>")
assert_contains("${PT_TS}" "<translation>Perfis</translation>")
assert_contains("${PT_TS}" "<translation>Adaptativo</translation>")
assert_contains("${PT_TS}" "<translation>Fixo</translation>")
assert_contains("${PT_TS}" "<translation>Nenhum</translation>")
assert_contains("${PT_TS}" "<translation>Padrão</translation>")
assert_contains("${PT_TS}" "cadência de saída explícita")

assert_contains("${ES_TS}" "<translation>Configuración de lsfg-vk</translation>")
assert_contains("${ES_TS}" "<translation>Perfiles</translation>")
assert_contains("${ES_TS}" "<translation>Adaptativo</translation>")
assert_contains("${ES_TS}" "<translation>Fijo</translation>")
assert_contains("${ES_TS}" "<translation>Ninguno</translation>")
assert_contains("${ES_TS}" "<translation>Predeterminado</translation>")
assert_contains("${ES_TS}" "cadencia de salida explícita")

assert_not_contains("${PT_TS}" "type=\"unfinished\"")
assert_not_contains("${ES_TS}" "type=\"unfinished\"")
foreach(ts IN ITEMS "${PT_TS}" "${ES_TS}")
    assert_not_contains("${ts}" "frame_generation_mode")
    assert_not_contains("${ts}" "target_fps")
    assert_not_contains("${ts}" "flow_scale")
    assert_not_contains("${ts}" "performance_mode")
endforeach()

file(READ "${ROOT}/${UI_QML}" ui_qml_content)
file(READ "${ROOT}/lsfg-vk-ui/rsc/widgets/FileEdit.qml" file_edit_content)
string(APPEND ui_qml_content "\n${file_edit_content}")
string(REPLACE ";" "__SEMICOLON__" ui_qml_content "${ui_qml_content}")
string(REGEX MATCHALL "qsTr\\(\"[^\"]+\"\\)" qstr_calls "${ui_qml_content}")
set(qml_sources)
foreach(call IN LISTS qstr_calls)
    string(REGEX REPLACE "^qsTr\\(\"|\"\\)$" "" source "${call}")
    list(APPEND qml_sources "${source}")
endforeach()
list(REMOVE_DUPLICATES qml_sources)
list(SORT qml_sources)

foreach(ts IN ITEMS "${PT_TS}" "${ES_TS}")
    file(READ "${ROOT}/${ts}" ts_content)
    string(REPLACE ";" "__SEMICOLON__" ts_content "${ts_content}")
    string(REGEX MATCHALL "<source>[^<]+</source>" source_elements "${ts_content}")
    set(ts_sources)
    foreach(element IN LISTS source_elements)
        string(REGEX REPLACE "^<source>|</source>$" "" source "${element}")
        list(APPEND ts_sources "${source}")
    endforeach()
    list(REMOVE_DUPLICATES ts_sources)
    list(SORT ts_sources)
    if(NOT qml_sources STREQUAL ts_sources)
        message(FATAL_ERROR "${ts} does not cover the complete qsTr source set")
    endif()
endforeach()

assert_contains("${UI_CMAKE}" "LinguistTools")
assert_contains("${UI_CMAKE}" "qt_add_translations(lsfg-vk-ui")
assert_contains("${UI_CMAKE}" "RESOURCE_PREFIX \"/i18n\"")
assert_not_contains("${UI_CMAKE}" "share/lsfg-vk-ui/translations")

assert_contains("${UI_QML}" "model: [qsTr(\"Adaptive\"), qsTr(\"Fixed\")]")
assert_contains("${UI_QML}" "backend.frame_generation_mode = index")
assert_contains("${UI_QML}" "model: [qsTr(\"None\")]")
assert_contains("${UI_QML}" "backend.pacing_mode = index")
assert_contains("${UI_QML}" "model: backend.gpus")
assert_contains("${UI_QML}" "index === 0 ? qsTr(\"Default\") : modelData")
assert_contains("${UI_QML}" "backend.gpu = index")

assert_contains("lsfg-vk-ui/src/backend.hpp" "conf.frame_generation_mode = ls::FrameGenerationMode::Adaptive")
assert_contains("lsfg-vk-ui/src/backend.hpp" "conf.frame_generation_mode = ls::FrameGenerationMode::Fixed")
assert_contains("lsfg-vk-ui/src/backend.hpp" "conf.pacing = ls::Pacing::None")
assert_contains("lsfg-vk-ui/src/backend.hpp" "gpu == \"Default\"")
assert_contains("lsfg-vk-ui/src/utils.cpp" "QStringList list{\"Default\"}")

assert_contains("${LANGUAGE_MANAGER_HPP}" "QSettings m_settings")
assert_contains("${LANGUAGE_MANAGER_CPP}" "QStringLiteral(\"ui/language\")")
assert_not_contains("${LANGUAGE_MANAGER_CPP}" "ConfigFile")
assert_not_contains("${LANGUAGE_MANAGER_CPP}" "conf.toml")
