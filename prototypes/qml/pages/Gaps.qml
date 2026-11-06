import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Gaps"
        description: "Spaces between and inside fibres."
        FormRow { label: "Min gap diameter"; value: Math.round(minGap.value) + " px"
                  Slider { id: minGap; from: 5; to: 100; value: 20 } }
        Item {
            width: parent.width; height: 20
            CheckBox { anchors.right: parent.right; text: "Overlay gaps" }
        }
        footer: [
            Button { primary: true; text: "Analyze"; icon: "play" },
            Note { text: "42 gaps (30 intra-fibre), median Ø 12.3 µm (45 px), 8 % of the image." }
        ]
    }

    Section {
        title: "Hematoxylin density (HDM)"
        description: "Dark-area measurement; stays on when gaps are off."
        last: true
        FormRow { label: "Max display HDM"; value: Math.round(hdm.value)
                  Slider { id: hdm; from: 100; to: 255; value: 230 } }
        FormRow { label: "Saturation"; value: Math.round(sat.value) + " %"
                  Slider { id: sat; from: 0; to: 50; value: 10 } }
    }
}
