import QtQuick 2.15

// The shared input surface: outlined, filled or underlined (t.fieldStyle),
// with an optional bottom stroke (Fluent style).
Rectangle {
    id: f
    property bool hot: false
    property bool enabledLook: true
    readonly property bool underline: t.fieldStyle === "underline"
    readonly property bool filled: t.fieldStyle === "filled"
    readonly property bool quiet: t.fieldStyle === "quiet"
    readonly property int textX: underline ? 0 : (quiet ? 6 : 10)

    radius: underline ? 0 : t.radius
    color: underline || (quiet && !hot) ? "transparent" : t.field
    border.width: t.fieldStyle === "outline" || (filled && hot) ? 1 : 0
    border.color: hot ? t.borderHover : t.border
    opacity: enabledLook ? 1 : 0.55

    Rectangle {
        visible: f.underline || t.bottomStroke !== ""
        x: f.underline ? 0 : f.radius / 2
        width: parent.width - 2 * x
        height: f.underline && f.hot ? 2 : 1
        anchors.bottom: parent.bottom
        color: f.underline ? (f.hot ? t.accent : t.border) : t.bottomStroke
    }
}
