import QtQuick 2.15

// Single-value slider: thin track, accent fill, round handle.
Item {
    id: s
    property real from: 0
    property real to: 100
    property real value: 0
    property real stepSize: 1
    property color fill: t.accent

    anchors.fill: parent
    readonly property real span: width - 14
    function pos(v) { return 7 + (v - from) / (to - from) * span }
    function val(x) {
        var v = from + Math.max(0, Math.min(1, (x - 7) / span)) * (to - from)
        return Math.round(v / stepSize) * stepSize
    }

    Rectangle { x: 7; width: s.span; height: 4; radius: 2; anchors.verticalCenter: parent.verticalCenter; color: t.track }
    Rectangle { x: 7; width: s.pos(s.value) - 7; height: 4; radius: 2; anchors.verticalCenter: parent.verticalCenter; color: s.fill }
    Rectangle {
        x: s.pos(s.value) - 7; anchors.verticalCenter: parent.verticalCenter
        width: 14; height: 14; radius: 7
        color: t.handle; border.color: s.fill; border.width: 2
    }
    MouseArea {
        anchors.fill: parent
        onPressed: s.value = s.val(mouse.x)
        onPositionChanged: s.value = s.val(mouse.x)
    }
}
