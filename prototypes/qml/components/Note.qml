import QtQuick 2.15

// Small wrapped status or help text.
Text {
    width: parent ? parent.width : 200
    wrapMode: Text.WordWrap
    lineHeight: 1.15
    font { family: t.font; pixelSize: t.small + 0.5 }
    color: t.faint
}
