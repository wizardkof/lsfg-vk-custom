import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "dialogs"
import "panes"
import "widgets"

ApplicationWindow {
    title: qsTr("lsfg-vk Configuration Window")
    width: 900
    height: 650
    minimumWidth: 700
    minimumHeight: 500
    visible: true
    // Keep the editor consistently dark instead of mixing the platform
    // window palette with dark custom Group/List backgrounds.
    color: "#151821"
    palette.window: "#151821"
    palette.windowText: "#f0f2f5"
    palette.base: "#10131a"
    palette.alternateBase: "#202432"
    palette.toolTipBase: "#202432"
    palette.toolTipText: "#f0f2f5"
    palette.text: "#f0f2f5"
    palette.button: "#272c39"
    palette.buttonText: "#f0f2f5"
    palette.brightText: "#ffffff"
    palette.light: "#3b4252"
    palette.midlight: "#343a48"
    palette.mid: "#2c3140"
    palette.dark: "#0d1016"
    palette.shadow: "#05070a"

    CenteredDialog {
        id: create_dialog

        name: qsTr("Create New Profile")
        onConfirm: backend.createProfile(create_name.text)

        TextField {
            id: create_name

            Layout.fillWidth: true
            placeholderText: qsTr("Choose a profile name")
            focus: true
        }

    }

    CenteredDialog {
        id: rename_dialog

        name: qsTr("Rename Profile")
        onConfirm: backend.renameProfile(rename_name.text)

        TextField {
            id: rename_name

            Layout.fillWidth: true
            placeholderText: qsTr("Choose a profile name")
            focus: true
        }

    }

    CenteredDialog {
        id: delete_dialog

        name: qsTr("Confirm Deletion")
        onConfirm: backend.deleteProfile()

        Label {
            Layout.fillWidth: true
            text: qsTr("Are you sure you want to delete the selected profile?")
            horizontalAlignment: Text.AlignHCenter
        }

    }

    LargeDialog {
        id: active_in_dialog

        name: qsTr("Active In")
        onConfirm: backend.createProfile(create_name.text)

        List {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: backend.active_in
            selected: backend.active_in_index
            onSelect: (index) => {
                backend.active_in_index = index;
                var idx = backend.active_in.index(index, 0);
                active_in_name.text = backend.active_in.data(idx);
            }
        }

        RowLayout {
            spacing: 8

            TextField {
                id: active_in_name

                Layout.fillWidth: true
                placeholderText: qsTr("Specify linux binary / exe file / process name")
                focus: true
            }

            Button {
                icon.name: "list-add"
                onClicked: backend.addActiveIn(active_in_name.text)
            }

            Button {
                icon.name: "list-remove"
                onClicked: backend.removeActiveIn()
            }

        }

    }

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Pane {
            SplitView.minimumWidth: 200
            SplitView.preferredWidth: 250
            SplitView.maximumWidth: 300

            Label {
                text: qsTr("Profiles")
                Layout.fillWidth: true
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
            }

            List {
                model: backend.profiles
                selected: backend.profile_index
                onSelect: (index) => {
                    return backend.profile_index = index;
                }
            }

            Button {
                Layout.fillWidth: true
                text: qsTr("Create New Profile")
                onClicked: {
                    create_name.text = "";
                    create_dialog.open();
                }
            }

            Button {
                Layout.fillWidth: true
                text: qsTr("Rename Profile")
                onClicked: {
                    var idx = backend.profiles.index(backend.profile_index, 0);
                    rename_name.text = backend.profiles.data(idx);
                    rename_dialog.open();
                }
            }

            Button {
                Layout.fillWidth: true
                text: qsTr("Delete Profile")
                onClicked: {
                    delete_dialog.open();
                }
            }

        }

        ScrollView {
            id: settings_scroll

            SplitView.fillWidth: true
            clip: true
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            ScrollBar.vertical.policy: ScrollBar.AsNeeded

            ColumnLayout {
                width: settings_scroll.availableWidth
                spacing: 4

                Group {
                    Layout.leftMargin: 12
                    Layout.rightMargin: 12
                    Layout.topMargin: 12
                    name: qsTr("Global Settings")

                    GroupEntry {
                        title: qsTr("Path to Lossless Scaling")
                        description: qsTr("Change the location of Lossless.dll")

                        FileEdit {
                            Layout.fillWidth: true
                            title: qsTr("Select Lossless.dll")
                            filter: qsTr("Dynamic Link Library Files (*.dll)")
                            text: backend.dll
                            onUpdate: (text) => {
                                return backend.dll = text;
                            }
                        }

                    }

                    GroupEntry {
                        title: qsTr("Allow half-precision")
                        description: qsTr("Allow acceleration through half-precision")

                        CheckBox {
                            Layout.alignment: Qt.AlignRight
                            checked: backend.allow_fp16
                            onToggled: backend.allow_fp16 = checked
                        }

                    }

                    GroupEntry {
                        title: qsTr("Language")
                        description: qsTr("Change the interface language")

                        ComboBox {
                            Layout.fillWidth: true
                            model: [qsTr("System Default"), qsTr("English"), qsTr("Português (Brasil)"), qsTr("Español")]
                            currentIndex: languageManager.current_language
                            onActivated: (index) => {
                                return languageManager.current_language = index;
                            }
                        }

                    }

                }

                Group {
                    Layout.leftMargin: 12
                    Layout.rightMargin: 12
                    name: qsTr("Profile Settings")
                    enabled: backend.available

                    GroupEntry {
                        title: qsTr("Active In")
                        description: qsTr("Specify which applications this profile is active in")

                        Button {
                            Layout.alignment: Qt.AlignRight
                            text: qsTr("Edit...")
                            onClicked: active_in_dialog.open()
                        }

                    }

                    GroupEntry {
                        title: qsTr("Frame Generation Mode")
                        description: qsTr("Adaptive follows the multiplier; Fixed targets an explicit output FPS")

                        ComboBox {
                            Layout.fillWidth: true
                            model: [qsTr("Adaptive"), qsTr("Fixed")]
                            currentIndex: backend.frame_generation_mode
                            onActivated: (index) => {
                                return backend.frame_generation_mode = index;
                            }
                        }

                    }

                    GroupEntry {
                        title: qsTr("Target FPS")
                        description: qsTr("Output FPS target used by Fixed mode")
                        visible: backend.frame_generation_mode === 1

                        SpinBox {
                            Layout.alignment: Qt.AlignRight
                            from: 1
                            to: 2147483647
                            value: backend.target_fps
                            onValueModified: backend.target_fps = value
                        }

                    }

                    GroupEntry {
                        title: qsTr("Multiplier")
                        description: qsTr("Adaptive only; 1x bypasses frame generation")

                        SpinBox {
                            Layout.alignment: Qt.AlignRight
                            enabled: backend.frame_generation_mode === 0
                            from: 1
                            to: backend.adaptive_multiplier_max
                            value: backend.multiplier
                            onValueModified: backend.multiplier = value
                        }

                    }

                    GroupEntry {
                        title: qsTr("Flow Scale")
                        description: qsTr("Lower the internal motion estimation resolution")

                        FlowSlider {
                            Layout.fillWidth: true
                            from: 0.25
                            to: 1
                            value: backend.flow_scale
                            onUpdate: (value) => {
                                return backend.flow_scale = value;
                            }
                        }

                    }

                    GroupEntry {
                        title: qsTr("Performance Mode")
                        description: qsTr("Use a significantly lighter frame generation model")

                        CheckBox {
                            Layout.alignment: Qt.AlignRight
                            checked: backend.performance_mode
                            onToggled: backend.performance_mode = checked
                        }

                    }

                    GroupEntry {
                        title: qsTr("Pacing Mode")
                        description: qsTr("Change how frames are presented to the display")

                        ComboBox {
                            Layout.fillWidth: true
                            model: [qsTr("None")]
                            currentIndex: backend.pacing_mode
                            onActivated: (index) => {
                                return backend.pacing_mode = index;
                            }
                        }

                    }

                    GroupEntry {
                        title: qsTr("GPU")
                        description: qsTr("Select which GPU to use for frame generation")

                        ComboBox {
                            id: gpu_combo_box

                            Layout.fillWidth: true
                            model: backend.gpus
                            displayText: currentIndex === 0 ? qsTr("Default") : currentText
                            currentIndex: backend.gpu
                            onActivated: (index) => {
                                return backend.gpu = index;
                            }

                            delegate: ItemDelegate {
                                width: ListView.view ? ListView.view.width : implicitWidth
                                text: index === 0 ? qsTr("Default") : modelData
                                highlighted: gpu_combo_box.highlightedIndex === index
                            }

                        }

                    }

                }

                Item {
                    Layout.fillHeight: true
                    Layout.minimumHeight: 12
                }

            }

        }

    }

}
