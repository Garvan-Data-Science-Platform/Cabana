import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Fit"
        description: "Find the cores and match them to the array map."
        FormRow { label: "Slide"
                  PathField { name: "TMA4_PicRed.vsi"; folder: "~/Studies/Stroma/slides"; icon: "file" } }
        FormRow { label: "Array map"
                  Combo { items: ["ICGC Array 4", "None (grid position only)", "Load map file…"] } }
        FormRow { label: "Orientation"
                  Combo { items: ["Auto", "0", "90", "180", "270", "0+flip", "90+flip", "180+flip", "270+flip"] } }
        FormRow { label: "Pixel size"
                  NumberField { value: 0.2738; decimals: 4; step: 0.01; suffix: "µm" } }
        FormRow {
            label: "Core Ø"
            Row {
                spacing: 12
                NumberField { value: 1250; step: 50; suffix: "µm"; fieldWidth: 112; enabledLook: !autoCore.checked }
                CheckBox { id: autoCore; text: "Auto"; checked: true; anchors.verticalCenter: parent.verticalCenter }
            }
        }
        FormRow {
            label: "Sensitivity"
            Row {
                spacing: 12
                NumberField { value: 15; from: 2; to: 60; fieldWidth: 112 }
                CheckBox { text: "Recover faint"; checked: true; anchors.verticalCenter: parent.verticalCenter }
            }
        }
        footer: Button { primary: true; text: "Fit Cores"; icon: "play" }
    }

    Section {
        title: "Filter"
        description: "Leave cores out of the export."
        FormRow { label: "Grid offset"
                  NumberField { value: 0.35; decimals: 2; step: 0.05; suffix: "pitch" } }
        FormRow { label: "Min stain"
                  NumberField { value: 2; suffix: "%" } }
        FormRow {
            label: "Core size"
            Row {
                spacing: 8
                NumberField { value: 80; step: 5; suffix: "%"; fieldWidth: 96 }
                Text { anchors.verticalCenter: parent.verticalCenter; text: "to"
                       font { family: t.font; pixelSize: t.size } color: t.label }
                NumberField { value: 120; step: 5; suffix: "%"; fieldWidth: 96 }
            }
        }
    }

    Section {
        title: "Edit"
        description: "Correct the fit by hand."
        Item {
            width: parent.width; height: t.rowH
            Row {
                spacing: 18; anchors.verticalCenter: parent.verticalCenter
                CheckBox { text: "Edit cores"; enabledLook: false }
                CheckBox { text: "Grid lines"; checked: true }
            }
            Button { width: 78; anchors.right: parent.right; text: "Undo"; enabledLook: false }
        }
        Row {
            width: parent.width; spacing: 8
            Button { width: (parent.width - 8) / 2; text: "Load Edits…" }
            Button { width: (parent.width - 8) / 2; text: "Save Edits…" }
        }
    }

    Section {
        title: "Export"
        description: "One image and mask per core, named by patient."
        last: true
        FormRow { label: "Margin"
                  NumberField { value: 30; step: 10; suffix: "µm" } }
        FormRow { label: "Mask shrink"
                  NumberField { value: 8; suffix: "px" } }
        FormRow {
            label: "Channels"; controlHeight: 20
            Row { spacing: 18
                  CheckBox { text: "BF"; checked: true }
                  CheckBox { text: "POL"; checked: true } }
        }
        FormRow { label: "Output folder"
                  PathField { name: "TMA4_PicRed_cores"; folder: "~/Studies/Stroma" } }
        footer: [
            Note { text: "96 cores on a 12 x 8 grid, 90 to export. Orientation 90, chosen by the Brain control." },
            Button { primary: true; text: "Export Cores" }
        ]
    }
}
