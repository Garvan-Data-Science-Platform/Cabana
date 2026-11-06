import QtQuick 2.15

// Drop-down: current item and a chevron (the popup is not prototyped).
FieldBox {
    id: c
    property var items: []
    property int index: 0

    anchors.fill: parent
    hot: area.containsMouse

    Text {
        x: c.textX; width: parent.width - x - 30; height: parent.height
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        text: c.items.length ? c.items[c.index] : ""
        font { family: t.font; pixelSize: t.size }
        color: t.text
    }
    Icon { anchors { right: parent.right; rightMargin: c.underline ? 2 : 9; verticalCenter: parent.verticalCenter }
           name: "chevron-down"; color: t.label }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                onClicked: c.index = (c.index + 1) % c.items.length }
}
