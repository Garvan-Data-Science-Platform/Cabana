import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Bands"
        description: "Sort polarised-light pixels into colour bands."
        FormRow { label: "Min brightness"; value: Math.round(bright.value)
                  Slider { id: bright; from: 0; to: 255; value: 10 } }
        FormRow { label: "Min saturation"; value: Math.round(sat.value)
                  Slider { id: sat; from: 0; to: 255; value: 0 } }
        FormRow { label: "Red-orange hue"; value: red.low + ", " + red.high
                  RangeSlider { id: red; from: -64; to: 191; low: 0; high: 27; fill: "#E0533D" } }
        FormRow { label: "Yellow hue"; value: yellow.low + ", " + yellow.high
                  RangeSlider { id: yellow; from: -64; to: 191; low: 28; high: 47; fill: "#E3B931" } }
        FormRow { label: "Green hue"; value: green.low + ", " + green.high
                  RangeSlider { id: green; from: -64; to: 191; low: 48; high: 140; fill: "#3FAF5A" } }
        FormRow {
            label: "View"
            Item {
                anchors.fill: parent
                // the drop-down takes the room the checkbox leaves
                Item { width: parent.width - overlay.width - 12; height: parent.height
                       Combo { items: ["Grid", "Total", "Red-Orange", "Yellow", "Green", "Class map"] } }
                CheckBox { id: overlay; anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                           text: "Overlay" }   // tooltip: Overlay the original image
            }
        }
        footer: [
            Button { primary: true; text: "Separate"; icon: "play" },
            Note { text: "Red-orange 62.1 %, yellow 20.3 %, green 17.6 % of the signal. 41.2 % of the image is in the bands, 3.1 % unclassified." }
        ]
    }

    Section {
        title: "Folder run"
        description: "Separate every image in a folder."
        last: true
        FlowRow { label: "Input folder"
                  PathField { name: "stroma"; folder: "~/Studies/BRF/export" } }
        FlowRow { label: "ROI masks"
                  PathField { name: "stroma_mask"; folder: "~/Studies/BRF/export"; clearable: true } }
        FlowRow { label: "Output folder"; lastNode: true
                  PathField { name: "BRF_stroma"; folder: "~/Studies/BRF" } }
        Item {
            width: parent.width; height: 20
            CheckBox { text: "Sort into folders"; checked: true }
            CheckBox { anchors.right: parent.right; text: "Save images"; checked: true }
        }
        footer: Button { primary: true; text: "Run Folder"; icon: "play" }
    }
}
