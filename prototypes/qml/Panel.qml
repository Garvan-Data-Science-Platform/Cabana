import QtQuick 2.15
import "components"

// The side panel: page title and one page. Tokens come from the `t`
// context property set by preview.py.
Rectangle {
    id: root
    property int page: 0
    readonly property real contentHeight: col.height + 24
    signal key(int k)
    readonly property var pages: [
        { title: "TMA Dearrayer", source: "pages/Tma.qml" },
        { title: "Birefringence", source: "pages/Birefringence.qml" },
        { title: "Segmentation", source: "pages/Segmentation.qml", toggle: true },
        { title: "Fibre Detection", source: "pages/Detection.qml" },
        { title: "Gap Analysis", source: "pages/Gaps.qml", toggle: true },
        { title: "Batch Run", source: "pages/BatchRun.qml" }
    ]

    color: t.panel
    focus: true
    Keys.onPressed: root.key(event.key)

    Flickable {
        anchors.fill: parent
        contentHeight: col.height + 24
        clip: true
        Column {
            id: col
            x: 12; y: 10
            width: root.width - 24
            Item {
                width: parent.width; height: title.height
                Text {
                    id: title
                    bottomPadding: t.sections === "cards" ? 10 : 2
                    text: root.pages[root.page].title
                    font { family: t.font; pixelSize: t.head + 2; weight: Font.DemiBold }
                    color: t.text
                }
                // pages that can be switched off in the pipeline
                Switch { visible: !!root.pages[root.page].toggle; anchors.right: parent.right; y: 1 }
            }
            Loader {
                width: parent.width
                height: item ? item.implicitHeight : 0
                source: root.pages[root.page].source
            }
        }
    }
}
