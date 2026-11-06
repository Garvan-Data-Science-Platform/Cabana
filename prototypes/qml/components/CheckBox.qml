import QtQuick 2.15

Item {
    id: c
    property string text
    property bool checked: false
    property bool enabledLook: true

    width: box.width + 9 + label.implicitWidth
    height: 20
    opacity: enabledLook ? 1 : 0.5

    Rectangle {
        id: box
        anchors.verticalCenter: parent.verticalCenter
        width: 15; height: 15; radius: t.boxRadius
        color: c.checked ? t.accent : (t.fieldStyle === "underline" ? "transparent" : t.field)
        border.color: c.checked ? t.accent : t.borderHover
        Icon { anchors.centerIn: parent; width: 13; height: 13; name: "check"; color: t.onAccent; visible: c.checked }
    }
    Text {
        id: label
        x: box.width + 9; anchors.verticalCenter: parent.verticalCenter
        text: c.text
        font { family: t.font; pixelSize: t.size }
        color: t.text
    }
    MouseArea { anchors.fill: parent; onClicked: c.checked = !c.checked }
}
