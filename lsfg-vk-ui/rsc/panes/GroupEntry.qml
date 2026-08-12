import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

RowLayout {
    property string title
    property string description
    default property alias content: inner.children

    id: root
    spacing: 12

    ColumnLayout {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        clip: true

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Label {
                Layout.fillWidth: true
                text: root.title
                font.bold: true
            }

            Label {
                Layout.fillWidth: true
                text: root.description
                wrapMode: Text.WordWrap
                color: Qt.rgba(
                    palette.text.r,
                    palette.text.g,
                    palette.text.b,
                    0.7
                )
            }
        }
    }

    RowLayout {
        Layout.minimumWidth: 220
        Layout.preferredWidth: 260

        Item {
            Layout.fillWidth: true
        }

        ColumnLayout {
            spacing: 0
            id: inner
        }
    }


}
