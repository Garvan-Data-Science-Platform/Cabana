import QtQuick 2.15

// On/off switch with its state written beside it ("On" / "Off").
Item {
    id: s
    property bool checked: true
    width: label.implicitWidth + 8 + track.width
    height: 20

    Text {
        id: label
        anchors.verticalCenter: parent.verticalCenter
        text: s.checked ? "On" : "Off"
        font { family: t.font; pixelSize: t.small + 1 }
        color: s.checked ? t.accentText : t.faint
    }
    Rectangle {
        id: track
        x: label.implicitWidth + 8; anchors.verticalCenter: parent.verticalCenter
        width: 30; height: 17; radius: 8.5
        color: s.checked ? t.accent : t.track
        Rectangle {
            x: s.checked ? parent.width - width - 2 : 2; y: 2
            width: 13; height: 13; radius: 6.5
            color: s.checked ? t.onAccent : t.handle
            Behavior on x { NumberAnimation { duration: 120 } }
        }
    }
    MouseArea { anchors.fill: parent; onClicked: s.checked = !s.checked }
}
