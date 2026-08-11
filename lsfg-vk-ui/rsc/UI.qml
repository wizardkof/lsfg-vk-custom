import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "dialogs"
import "panes"
import "widgets"

ApplicationWindow {
    title: "lsfg-vk Configuration Window"
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
        name: "Create New Profile"
        onConfirm: backend.createProfile(create_name.text)

        TextField {
            Layout.fillWidth: true
            id: create_name
            placeholderText: "Choose a profile name"
            focus: true
        }
    }

    CenteredDialog {
        id: rename_dialog
        name: "Rename Profile"
        onConfirm: backend.renameProfile(rename_name.text)

        TextField {
            Layout.fillWidth: true
            id: rename_name
            placeholderText: "Choose a profile name"
            focus: true
        }
    }

    CenteredDialog {
        id: delete_dialog
        name: "Confirm Deletion"
        onConfirm: backend.deleteProfile()

        Label {
            Layout.fillWidth: true
            text: "Are you sure you want to delete the selected profile?"
            horizontalAlignment: Text.AlignHCenter
        }
    }

    LargeDialog {
        id: active_in_dialog
        onConfirm: backend.createProfile(create_name.text)

        List {
            Layout.fillWidth: true
            Layout.fillHeight: true

            model: backend.active_in
            selected: backend.active_in_index
            onSelect: (index) => {
                backend.active_in_index = index
                var idx = backend.active_in.index(index, 0);
                active_in_name.text = backend.active_in.data(idx);
            }
        }

        RowLayout {
            spacing: 8

            TextField {
                Layout.fillWidth: true
                id: active_in_name
                placeholderText: "Specify linux binary / exe file / process name"
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
                text: "Profiles"
                Layout.fillWidth: true
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
            }

            List {
                model: backend.profiles
                selected: backend.profile_index
                onSelect: (index) => backend.profile_index = index
            }

            Button {
                Layout.fillWidth: true
                text: "Create New Profile"
                onClicked: {
                    create_name.text = ""
                    create_dialog.open()
                }
            }
            Button {
                Layout.fillWidth: true
                text: "Rename Profile"
                onClicked: {
                    var idx = backend.profiles.index(backend.profile_index, 0);
                    rename_name.text = backend.profiles.data(idx);
                    rename_dialog.open()
                }
            }
            Button {
                Layout.fillWidth: true
                text: "Delete Profile"
                onClicked: {
                    delete_dialog.open()
                }
            }
        }

        Pane {
            SplitView.fillWidth: true

            Group {
                name: "Global Settings"

                GroupEntry {
                    title: "Path to Lossless Scaling"
                    description: "Change the location of Lossless.dll"

                    FileEdit {
                        Layout.fillWidth: true

                        title: "Select Lossless.dll"
                        filter: "Dynamic Link Library Files (*.dll)"

                        text: backend.dll
                        onUpdate: (text) => backend.dll = text
                    }
                }

                GroupEntry {
                    title: "Allow half-precision"
                    description: "Allow acceleration through half-precision"

                    CheckBox {
                        Layout.alignment: Qt.AlignRight

                        checked: backend.allow_fp16
                        onToggled: backend.allow_fp16 = checked
                    }
                }
            }

            Group {
                name: "Profile Settings"
                enabled: backend.available

                GroupEntry {
                    title: "Active In"
                    description: "Specify which applications this profile is active in"

                    Button {
                        Layout.alignment: Qt.AlignRight

                        text: "Edit..."
                        onClicked: active_in_dialog.open()
                    }
                }

                GroupEntry {
                    title: "Frame Generation Mode"
                    description: "Adaptive follows the multiplier; Fixed targets an explicit output FPS"

                    ComboBox {
                        Layout.fillWidth: true

                        model: ["Adaptive", "Fixed"]
                        currentIndex: backend.frame_generation_mode
                        onActivated: (index) => backend.frame_generation_mode = index
                    }
                }

                GroupEntry {
                    title: "Target FPS"
                    description: "Output FPS target used by Fixed mode"
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
                    title: "Multiplier"
                    description: "Adaptive only; 1x bypasses frame generation"

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
                    title: "Flow Scale"
                    description: "Lower the internal motion estimation resolution"

                    FlowSlider {
                        Layout.fillWidth: true

                        from: 0.25
                        to: 1.00

                        value: backend.flow_scale
                        onUpdate: (value) => backend.flow_scale = value
                    }
                }

                GroupEntry {
                    title: "Performance Mode"
                    description: "Use a significantly lighter frame generation model"

                    CheckBox {
                        Layout.alignment: Qt.AlignRight

                        checked: backend.performance_mode
                        onToggled: backend.performance_mode = checked
                    }
                }

                GroupEntry {
                    title: "Pacing Mode"
                    description: "Change how frames are presented to the display"

                    ComboBox {
                        Layout.fillWidth: true

                        model: ["None"]
                        currentIndex: backend.pacing_mode
                        onActivated: (index) => backend.pacing_mode = index
                    }
                }

                GroupEntry {
                    title: "GPU"
                    description: "Select which GPU to use for frame generation"

                    ComboBox {
                        Layout.fillWidth: true

                        model: backend.gpus
                        currentIndex: backend.gpu
                        onActivated: (index) => backend.gpu = index
                    }
                }
            }

            Item {
                Layout.fillHeight: true
            }
        }
    }
}
