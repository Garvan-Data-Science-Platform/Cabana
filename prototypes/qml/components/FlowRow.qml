import QtQuick 2.15

// A path row on a data-flow rail: node (filled when set, hollow when empty)
// and a segment down to the next row unless this is the last one.
FormRow {
    id: fr
    property bool set: true
    property bool lastNode: false
    indent: t.railOn ? 22 : 0

    readonly property real nodeY: stacked ? 8 : controlHeight / 2

    decoration: [
    Rectangle {
        visible: t.railOn && !fr.lastNode
        x: 6; y: fr.nodeY + 6; width: 2
        height: fr.height + t.rowGap - 12
        color: t.rail; opacity: 0.55
    },
    Rectangle {
        visible: t.railOn
        x: 2.5; y: fr.nodeY - 4.5; width: 9; height: 9; radius: 4.5
        color: fr.set ? t.rail : (t.sections === "cards" ? t.card : t.panel)
        border.color: fr.set ? t.rail : t.faint
        border.width: 1.5
    }
    ]
}
