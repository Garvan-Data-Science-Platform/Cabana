import QtQuick 2.15

// Primary (filled) or secondary (outlined) button with an optional icon.
Rectangle {
    id: b
    property string text
    property string icon
    property bool primary: false
    property bool enabledLook: true
    signal clicked()

    width: parent ? parent.width : 120
    height: primary ? t.btnH : t.rowH
    radius: primary ? t.btnRadius : Math.min(t.btnRadius, t.radius + 1)
    color: primary ? (area.containsMouse ? t.buttonHover : t.buttonBg)
                   : (area.containsMouse ? t.field : t.secondaryBg)
    border.width: primary ? 0 : 1
    border.color: t.border
    opacity: enabledLook ? 1 : 0.45

    Row {
        anchors.centerIn: parent
        spacing: 7
        Icon { visible: b.icon !== ""; anchors.verticalCenter: parent.verticalCenter
               width: 13; height: 13; name: b.icon; color: b.primary ? t.buttonText : t.text }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: b.text
            font { family: t.font; pixelSize: t.size; weight: b.primary ? Font.DemiBold : Font.Medium }
            color: b.primary ? t.buttonText : t.text
        }
    }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                onClicked: b.clicked() }
}
