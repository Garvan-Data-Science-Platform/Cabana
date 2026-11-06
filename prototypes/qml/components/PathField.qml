import QtQuick 2.15

// File or folder picker: name, then the shortened parent path, then icons.
FieldBox {
    id: p
    property string name
    property string folder
    property string icon: "folder"
    property bool set: true
    property bool clearable: false
    property string placeholder: "Optional"
    signal clear()
    signal choose()

    anchors.fill: parent
    hot: area.containsMouse

    Text {
        id: nameText
        x: p.textX; height: parent.height
        verticalAlignment: Text.AlignVCenter
        text: p.set ? p.name : p.placeholder
        font { family: t.font; pixelSize: t.size; weight: p.set ? Font.Medium : Font.Normal }
        color: p.set ? t.text : t.placeholder
    }
    Text {
        anchors { left: nameText.right; leftMargin: 6; right: tools.left; rightMargin: 8 }
        height: parent.height
        verticalAlignment: Text.AlignVCenter
        visible: p.set
        elide: Text.ElideLeft
        text: p.folder
        font { family: t.font; pixelSize: t.small }
        color: t.faint
    }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                onClicked: p.choose() }
    Row {
        id: tools
        anchors { right: parent.right; rightMargin: p.underline ? 2 : 9; verticalCenter: parent.verticalCenter }
        spacing: 8
        Icon {
            name: "clear"; color: t.faint
            visible: p.clearable && p.set
            MouseArea { anchors { fill: parent; margins: -4 } onClicked: p.clear() }
        }
        Icon { name: p.icon; color: t.label }
    }
}
