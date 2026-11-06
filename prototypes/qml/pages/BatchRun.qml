import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Data flow"
        description: "What goes in, and where the results go."
        FlowRow { label: "Images"
                  PathField { name: "Images"; folder: "~/Studies/Stroma/TMA4_cores/BF/Patients" } }
        FlowRow { label: "ROI masks"; set: false
                  PathField { set: false } }
        FlowRow { label: "Parameters"
                  PathField { name: "Parameters_BF.yml"; folder: "~/Studies/Stroma"; icon: "file" } }
        FlowRow { label: "Results"; lastNode: true
                  PathField { name: "Output"; folder: "~/Studies/Stroma/TMA4_cores/BF/Patients" } }
    }

    Section {
        title: "Run"
        description: "Settings for this run only."
        last: true
        FormRow {
            label: "Batch size"
            Row {
                spacing: 12
                NumberField { id: bs; value: 5; from: 1; to: 100; fieldWidth: 96 }
                Text { anchors.verticalCenter: parent.verticalCenter
                       text: Math.ceil(248 / bs.value) + " batches"
                       font { family: t.font; pixelSize: t.small + 1 } color: t.faint }
            }
        }
        FormRow { label: "Max size"; controlHeight: 20
                  Text { text: "5000 px"; anchors.verticalCenter: parent.verticalCenter
                         font { family: t.font; pixelSize: t.size } color: t.text } }
        OptionList {
            title: "Save, besides the measurements"
            options: [
                { text: "Visualisations", note: "colour maps, gap images", checked: true },
                { text: "Per-patient statistics", note: "mean, STD, SEM", checked: true },
                { text: "Risk scores", note: "rigidity, bundling", checked: false }
            ]
        }
        footer: Button { primary: true; text: "Process Batch"; icon: "play" }
    }
}
