find_package(Qt6 REQUIRED COMPONENTS LinguistTools Quick QuickControls2)

set(UI_SOURCES
    "src/backend.cpp"
    "src/language_manager.cpp"
    "src/main.cpp"
    "src/utils.cpp")

set(UI_RESOURCES
    "rsc/dialogs/CenteredDialog.qml"
    "rsc/dialogs/LargeDialog.qml"
    "rsc/panes/Group.qml"
    "rsc/panes/GroupEntry.qml"
    "rsc/panes/Pane.qml"
    "rsc/widgets/FileEdit.qml"
    "rsc/widgets/FlowSlider.qml"
    "rsc/widgets/List.qml"
    "rsc/gay.pancake.lsfg-vk-ui.png"
    "rsc/UI.qml")

set(UI_TRANSLATION_SOURCES
    ${UI_SOURCES}
    "src/backend.hpp"
    "src/language_manager.hpp"
    "src/utils.hpp"
    "rsc/dialogs/CenteredDialog.qml"
    "rsc/dialogs/LargeDialog.qml"
    "rsc/panes/Group.qml"
    "rsc/panes/GroupEntry.qml"
    "rsc/panes/Pane.qml"
    "rsc/widgets/FileEdit.qml"
    "rsc/widgets/FlowSlider.qml"
    "rsc/widgets/List.qml"
    "rsc/UI.qml")

set(UI_TRANSLATIONS
    "translations/lsfg-vk-ui_pt_BR.ts"
    "translations/lsfg-vk-ui_es.ts")

qt_add_executable(lsfg-vk-ui ${UI_SOURCES})

qt_add_resources(lsfg-vk-ui RESOURCES
    PREFIX "/"
    FILES ${UI_RESOURCES})

# Qt 6.5-compatible translation integration. The generated catalogs are
# embedded so the UI remains self-contained after installation.
qt_add_translations(lsfg-vk-ui
    TS_FILES ${UI_TRANSLATIONS}
    SOURCES ${UI_TRANSLATION_SOURCES}
    RESOURCE_PREFIX "/i18n")

set_target_properties(lsfg-vk-ui PROPERTIES
    AUTOMOC ON
    AUTOUIC ON)

target_compile_options(lsfg-vk-ui PRIVATE # QT-codegen warnings
    -Wno-unknown-warning-option
    -Wno-ctad-maybe-unsupported
    -Wno-unsafe-buffer-usage-in-libc-call
    -Wno-global-constructors
    -Wno-unsafe-buffer-usage)

target_link_libraries(lsfg-vk-ui
    PRIVATE lsfg-vk-common
    PRIVATE lsfg-vk-backend
    PRIVATE Qt6::Quick
    PRIVATE Qt6::QuickControls2)

install(TARGETS lsfg-vk-ui
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
if(LSFGVK_INSTALL_XDG_FILES)
    install(FILES "rsc/gay.pancake.lsfg-vk-ui.png"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/256x256/apps")
    install(FILES "rsc/gay.pancake.lsfg-vk-ui.desktop"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications")
endif()
