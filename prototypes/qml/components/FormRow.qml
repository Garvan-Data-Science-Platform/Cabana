import QtQuick 2.15

// A labelled row. Labels sit left of the control, or above it when
// t.labels === "top". An optional value is shown right-aligned (sliders).
Item {
    id: r
    property string label
    property string value
    property real controlHeight: t.rowH
    property real indent: 0          // room for a data-flow rail
    default property alias control: slot.data
    property alias decoration: decor.data   // drawn under the row (FlowRow rail)
    readonly property bool stacked: t.labels === "top"
    readonly property real valueWidth: value !== "" && !stacked ? 58 : 0

    width: parent ? parent.width : 0
    height: stacked ? 22 + controlHeight : controlHeight

    Item { id: decor; anchors.fill: parent }

    // hairline in the gap below the row, unless it is the last child
    Rectangle {
        visible: t.rowLines && r.parent && r.parent.children[r.parent.children.length - 1] !== r
        y: r.height + t.rowGap / 2 - 0.5
        width: parent.width; height: 1
        color: t.cardBorderColor
    }

    Text {
        x: r.indent
        width: r.stacked ? parent.width - r.indent - 70 : t.labelWidth
        // tall controls (a column of checkboxes) keep the label on the first line
        height: r.stacked ? 16 : (r.controlHeight > t.rowH ? 20 : r.controlHeight)
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        text: r.label
        font { family: t.font; pixelSize: r.stacked ? t.small + 1 : t.size }
        color: t.label
    }
    Text {
        visible: r.value !== ""
        anchors.right: parent.right
        width: 56
        height: r.stacked ? 16 : r.controlHeight
        horizontalAlignment: Text.AlignRight
        verticalAlignment: Text.AlignVCenter
        text: r.value
        font { family: t.font; pixelSize: t.small + 1 }
        color: t.faint
    }
    Item {
        id: slot
        x: r.stacked ? r.indent : r.indent + t.labelWidth
        y: r.stacked ? 22 : 0
        width: parent.width - x - r.valueWidth
        height: r.controlHeight
    }
}
