import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Colour"
        description: "The stain colour that marks collagen."
        FormRow {
            label: "Colour of interest"
            Row {
                spacing: 10; anchors.verticalCenter: parent.verticalCenter
                Rectangle { width: 22; height: 22; radius: 5; color: "#F53282"; border.color: t.border }
                Text { anchors.verticalCenter: parent.verticalCenter; text: "Hue 0.96"
                       font { family: t.font; pixelSize: t.size; weight: Font.Medium } color: t.text }
            }
        }
        FormRow { label: "Colour threshold"; value: (thr.value / 100).toFixed(2)
                  Slider { id: thr; from: 0; to: 100; value: 20 } }
    }

    Section {
        title: "Training"
        description: "More labels and fewer iterations give finer regions."
        FormRow { label: "Labels"; value: Math.round(labels.value)
                  Slider { id: labels; from: 8; to: 96; value: 32 } }
        FormRow { label: "Max iterations"; value: Math.round(iters.value)
                  Slider { id: iters; from: 10; to: 100; value: 30 } }
    }

    Section {
        title: "Large images"
        description: "How big images are split for training and batches."
        last: true
        FormRow { label: "Patch size"
                  NumberField { value: 1024; step: 256; to: 8192; suffix: "px"; special: "Off" } }
        FormRow { label: "Max size"
                  NumberField { value: 5000; step: 1000; to: 20000; suffix: "px"; special: "No limit" } }
        Item {
            width: parent.width; height: 20
            CheckBox { text: "White background"; checked: true }
            CheckBox { anchors.right: parent.right; text: "Overlay original" }
        }
        footer: Button { primary: true; text: "Segment"; icon: "play" }
    }
}
