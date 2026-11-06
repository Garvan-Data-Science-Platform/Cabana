import QtQuick 2.15

// A titled inset list of on/off options, one per row: checkbox and name on
// the left, a short note on what it produces on the right.
Column {
    id: list
    property string title
    property var options: []      // [{ text, note, checked }]

    width: parent ? parent.width : 0
    spacing: 6

    Text {
        visible: list.title !== ""
        text: list.title
        font { family: t.font; pixelSize: t.small + 1 }
        color: t.label
    }

    Rectangle {
        width: parent.width
        height: rows.height
        radius: t.radius
        color: t.field
        border.width: t.fieldStyle === "outline" ? 1 : 0
        border.color: t.border

        Column {
            id: rows
            width: parent.width
            Repeater {
                model: list.options
                Item {
                    width: rows.width; height: t.rowH + 2
                    CheckBox {
                        x: 10; anchors.verticalCenter: parent.verticalCenter
                        text: modelData.text; checked: modelData.checked
                    }
                    Text {
                        anchors { right: parent.right; rightMargin: 10; verticalCenter: parent.verticalCenter }
                        text: modelData.note
                        font { family: t.font; pixelSize: t.small }
                        color: t.faint
                    }
                    Rectangle {
                        visible: index < list.options.length - 1
                        anchors.bottom: parent.bottom
                        x: 35; width: parent.width - x; height: 1
                        color: t.cardBorderColor
                    }
                }
            }
        }
    }
}
