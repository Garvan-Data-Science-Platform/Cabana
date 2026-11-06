import QtQuick 2.15

// Two-handle slider; the nearer handle follows the pointer.
Item {
    id: s
    property real from: 0
    property real to: 100
    property real low: 25
    property real high: 75
    property real stepSize: 1
    property color fill: t.accent

    anchors.fill: parent
    readonly property real span: width - 14
    function pos(v) { return 7 + (v - from) / (to - from) * span }
    function val(x) {
        var v = from + Math.max(0, Math.min(1, (x - 7) / span)) * (to - from)
        return Math.round(v / stepSize) * stepSize
    }
    property bool dragLow: true

    Rectangle { x: 7; width: s.span; height: 4; radius: 2; anchors.verticalCenter: parent.verticalCenter; color: t.track }
    Rectangle { x: s.pos(s.low); width: s.pos(s.high) - s.pos(s.low); height: 4
                anchors.verticalCenter: parent.verticalCenter; color: s.fill }
    Repeater {
        model: [s.low, s.high]
        Rectangle {
            x: s.pos(modelData) - 7; anchors.verticalCenter: parent.verticalCenter
            width: 14; height: 14; radius: 7
            color: t.handle; border.color: s.fill; border.width: 2
        }
    }
    MouseArea {
        anchors.fill: parent
        onPressed: {
            var v = s.val(mouse.x)
            s.dragLow = Math.abs(v - s.low) <= Math.abs(v - s.high)
            onPositionChanged(mouse)
        }
        onPositionChanged: {
            var v = s.val(mouse.x)
            if (s.dragLow) s.low = Math.min(v, s.high)
            else s.high = Math.max(v, s.low)
        }
    }
}
