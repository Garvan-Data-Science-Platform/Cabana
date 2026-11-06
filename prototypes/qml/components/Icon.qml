import QtQuick 2.15

// Small line icons drawn on a 14 x 14 grid.
Canvas {
    property string name
    property color color: "black"
    width: 14; height: 14

    onNameChanged: requestPaint()
    onColorChanged: requestPaint()
    onVisibleChanged: requestPaint()

    onPaint: {
        var c = getContext("2d")
        c.reset()
        c.scale(width / 14, height / 14)
        c.strokeStyle = color; c.fillStyle = color
        c.lineWidth = 1.3; c.lineJoin = "round"; c.lineCap = "round"
        c.beginPath()
        switch (name) {
        case "chevron-down":
            c.moveTo(4, 5.5); c.lineTo(7, 8.5); c.lineTo(10, 5.5); c.stroke(); break
        case "chevron-right":
            c.moveTo(5.5, 4); c.lineTo(8.5, 7); c.lineTo(5.5, 10); c.stroke(); break
        case "folder":
            c.moveTo(1.5, 3); c.lineTo(5.3, 3); c.lineTo(6.8, 4.6); c.lineTo(12.5, 4.6)
            c.lineTo(12.5, 11.5); c.lineTo(1.5, 11.5); c.closePath(); c.stroke(); break
        case "file":
            c.moveTo(3.2, 1.5); c.lineTo(8.3, 1.5); c.lineTo(11, 4.2); c.lineTo(11, 12.5)
            c.lineTo(3.2, 12.5); c.closePath(); c.stroke()
            c.beginPath(); c.moveTo(8.3, 1.5); c.lineTo(8.3, 4.2); c.lineTo(11, 4.2); c.stroke(); break
        case "clear":
            c.arc(7, 7, 5.5, 0, 2 * Math.PI); c.fill()
            c.strokeStyle = "white"; c.lineWidth = 1.2
            c.beginPath(); c.moveTo(5, 5); c.lineTo(9, 9); c.moveTo(9, 5); c.lineTo(5, 9); c.stroke(); break
        case "play":
            c.moveTo(3.5, 2); c.lineTo(12.5, 7); c.lineTo(3.5, 12); c.closePath(); c.fill(); break
        case "stop":
            c.moveTo(4, 2); c.arcTo(12, 2, 12, 12, 2); c.arcTo(12, 12, 2, 12, 2)
            c.arcTo(2, 12, 2, 2, 2); c.arcTo(2, 2, 12, 2, 2); c.closePath(); c.fill(); break
        case "check":
            c.lineWidth = 1.8
            c.moveTo(3.3, 7.3); c.lineTo(5.9, 9.8); c.lineTo(10.7, 4.4); c.stroke(); break
        case "reset":
            c.arc(7, 7.5, 4.3, -2.6, 2.0); c.stroke()
            c.beginPath(); c.moveTo(2.2, 2.6); c.lineTo(2.6, 5.4); c.lineTo(5.4, 5.0); c.stroke(); break
        case "more":
            for (var i = 0; i < 3; i++) { c.beginPath(); c.arc(3 + i * 4, 7, 1.1, 0, 2 * Math.PI); c.fill() }
            break
        case "minus":
            c.moveTo(3.5, 7); c.lineTo(10.5, 7); c.stroke(); break
        case "plus":
            c.moveTo(3.5, 7); c.lineTo(10.5, 7); c.moveTo(7, 3.5); c.lineTo(7, 10.5); c.stroke(); break
        }
    }
}
