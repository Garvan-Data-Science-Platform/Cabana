import QtQuick 2.15

// Collapsible section. Flat sections are separated by a rule; with
// t.sections === "cards" each sits on a card whose look is set by tokens:
// cardBorder, cardShadow, headerBand, footerBand, footerDivider, accentEdge,
// topHighlight, titleOutside, showDesc, headerIcons, cardPad.
Item {
    id: s
    property string title
    property string description
    property bool open: true
    property bool last: false
    default property alias content: body.data
    property alias headerRight: right.data
    property alias footer: foot.data

    readonly property bool cards: t.sections === "cards"
    readonly property bool outside: cards && t.titleOutside
    readonly property bool hasFooter: foot.children.length > 0
    readonly property bool desc: t.showDesc && description !== ""
    readonly property int pad: cards ? t.cardPad : 0

    width: parent ? parent.width : 0
    height: outTitle.height + card.height + (cards ? 0 : 1)

    // group title above the card (grouped-settings style)
    Text {
        id: outTitle
        visible: s.outside
        height: visible ? 24 : 0
        leftPadding: 4
        verticalAlignment: Text.AlignTop
        text: s.title
        font { family: t.font; pixelSize: t.small + 1; weight: Font.DemiBold }
        color: t.label
    }

    // soft shadow: a few widening translucent layers under the card
    Repeater {
        model: s.cards && t.cardShadow ? 4 : 0
        Rectangle {
            x: card.x - index; y: card.y + 1 + index * 0.75
            width: card.width + 2 * index; height: card.height + index
            radius: card.radius + index
            color: Qt.rgba(0, 0, 0, t.shadowAlpha / (index + 1))
        }
    }

    Rectangle {
        id: card
        y: outTitle.height
        width: parent.width
        height: col.implicitHeight
        color: s.cards ? t.card : "transparent"
        radius: s.cards ? t.cardRadius : 0
        border.width: s.cards && t.cardBorder ? 1 : 0
        border.color: t.cardBorderColor

        // header band (Blender-style): tinted strip behind the header
        Rectangle {
            visible: s.cards && t.headerBand && !s.outside
            x: card.border.width; y: card.border.width
            width: card.width - 2 * x; height: header.height
            radius: card.radius - x
            color: t.cardHeader
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: parent.radius; color: parent.color }
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: t.cardBorderColor
                        visible: s.open }
        }
        // footer band: tinted strip behind the actions
        Rectangle {
            visible: s.cards && t.footerBand && s.hasFooter && s.open
            x: card.border.width
            y: footArea.y
            width: card.width - 2 * x; height: card.height - footArea.y - card.border.width
            radius: card.radius - x
            color: t.cardHeader
            Rectangle { width: parent.width; height: parent.radius; color: parent.color }
            Rectangle { width: parent.width; height: 1; color: t.cardBorderColor }
        }
        // accent edge on the left
        Rectangle {
            visible: s.cards && t.accentEdge
            x: 0; y: card.radius; width: 3; height: card.height - 2 * card.radius
            radius: 1.5; color: t.accent
        }
        // 1 px highlight along the top edge (bevel on dark themes)
        Rectangle {
            visible: s.cards && t.topHighlight
            x: card.radius; y: 1; width: card.width - 2 * card.radius; height: 1
            color: t.highlightLine
        }

        Column {
            id: col
            x: s.pad + (s.cards && t.accentEdge ? 3 : 0)
            width: parent.width - 2 * s.pad - (s.cards && t.accentEdge ? 3 : 0)

            Item {
                id: header
                visible: !s.outside
                x: -s.pad; width: parent.width + 2 * s.pad
                height: visible ? (s.desc ? 52 : 38) + (s.cards && t.headerBand ? 2 : 0) : (s.cards ? t.cardPad : 0)

                Item {
                    x: s.pad; width: parent.width - 2 * s.pad; height: 38
                    Icon { name: s.open ? "chevron-down" : "chevron-right"; color: t.label
                           anchors.verticalCenter: parent.verticalCenter }
                    Text {
                        x: 20; anchors.verticalCenter: parent.verticalCenter
                        text: s.title
                        font { family: t.font; pixelSize: t.head; weight: Font.DemiBold }
                        color: t.text
                    }
                    MouseArea { anchors.fill: parent; onClicked: s.open = !s.open }
                    Row {
                        anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                        spacing: 12
                        Row { id: right }
                        Icon { visible: t.headerIcons; name: "reset"; color: t.faint; anchors.verticalCenter: parent.verticalCenter }
                        Icon { visible: t.headerIcons; name: "more"; color: t.faint; anchors.verticalCenter: parent.verticalCenter }
                    }
                }
                Text {
                    visible: s.desc
                    x: s.pad + 20; y: 30
                    width: parent.width - x - s.pad
                    elide: Text.ElideRight
                    text: s.description
                    font { family: t.font; pixelSize: t.small + 0.5 }
                    color: t.faint
                }
            }

            Item { width: 1; height: s.open && s.cards && t.headerBand && !s.outside ? 10 : 0 }

            Column {
                id: body
                visible: s.open
                width: parent.width
                spacing: t.rowGap
            }

            Item {
                id: footArea
                visible: s.open && s.hasFooter
                width: parent.width
                height: visible ? foot.implicitHeight + (s.cards && t.footerBand ? 2 * t.cardPad : t.rowGap)
                                  + (s.cards && t.footerDivider ? t.rowGap : 0) : 0
                Rectangle {
                    visible: s.cards && t.footerDivider && !t.footerBand
                    x: -s.pad; width: parent.width + 2 * s.pad; y: t.rowGap / 2 + 2; height: 1
                    color: t.cardBorderColor
                }
                Column {
                    id: foot
                    y: s.cards && t.footerBand ? t.cardPad : t.rowGap + (s.cards && t.footerDivider ? t.rowGap : 0)
                    width: parent.width
                    spacing: t.rowGap
                }
            }

            Item { width: 1; height: s.open ? (s.cards ? (t.footerBand && s.hasFooter ? 0 : t.cardPad + 2) : 16)
                                            : (s.cards ? 2 : 0) }
        }
    }

    Rectangle {
        visible: !s.cards && !s.last
        anchors.bottom: parent.bottom
        width: parent.width; height: 1
        color: t.rule
    }
}
