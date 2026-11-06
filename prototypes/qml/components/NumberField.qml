import QtQuick 2.15

// Numeric entry: value and unit, with step buttons on the right.
FieldBox {
    id: n
    property real value
    property real step: 1
    property real from: 0
    property real to: 1e9
    property int decimals: 0
    property string suffix
    property string special        // shown instead of the value when value === from
    property real fieldWidth: 132

    width: fieldWidth
    height: t.rowH
    hot: area.containsMouse

    Text {
        x: n.textX; height: parent.height
        verticalAlignment: Text.AlignVCenter
        text: n.special !== "" && n.value === n.from ? n.special
              : n.value.toFixed(n.decimals) + (n.suffix !== "" ? " " + n.suffix : "")
        font { family: t.font; pixelSize: t.size }
        color: t.text
    }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true }
    Row {
        anchors { right: parent.right; rightMargin: n.underline ? 0 : 4; verticalCenter: parent.verticalCenter }
        Repeater {
            model: [-1, 1]
            Item {
                width: 22; height: n.height
                Icon { anchors.centerIn: parent; width: 11; height: 11
                       name: modelData < 0 ? "minus" : "plus"; color: t.label }
                MouseArea { anchors.fill: parent
                            onClicked: n.value = Math.max(n.from, Math.min(n.to, n.value + modelData * n.step)) }
            }
        }
    }
}
