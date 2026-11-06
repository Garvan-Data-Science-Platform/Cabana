import QtQuick 2.15
import "../components"

Column {
    spacing: t.sections === "cards" ? 10 : 0

    Section {
        title: "Detection"
        description: "Ridge detection of individual fibres."
        FormRow { label: "Line width"; value: width_.low + ", " + width_.high + " px"
                  RangeSlider { id: width_; from: 1; to: 15; low: 5; high: 7 } }
        FormRow { label: "Line step"; value: Math.round(step.value) + " px"
                  Slider { id: step; from: 1; to: 5; value: 2 } }
        FormRow { label: "Contrast"; value: contrast.low + ", " + contrast.high
                  RangeSlider { id: contrast; from: 0; to: 255; low: 100; high: 200 } }
        FormRow { label: "Min line length"; value: Math.round(minLen.value) + " px"
                  Slider { id: minLen; from: 1; to: 50; value: 5 } }
        Flow {
            width: parent.width; spacing: 18
            CheckBox { text: "Dark line"; checked: true }
            CheckBox { text: "Extend line" }
            CheckBox { text: "Overlay fibres"; checked: true }
        }
        footer: Button { primary: true; text: "Detect"; icon: "play" }
    }

    Section {
        title: "Measurements"
        description: "Used by Batch Run, not by Detect."
        last: true
        Note { visible: !t.showDesc; height: visible ? implicitHeight : 0; text: "Used by Batch Run, not by Detect." }
        FormRow { label: "Min branch length"; value: Math.round(branch.value) + " px"
                  Slider { id: branch; from: 1; to: 100; value: 5 } }
        FormRow { label: "Curvature window"; value: curv.low + ", " + curv.high + " px"
                  RangeSlider { id: curv; from: 2; to: 100; low: 10; high: 30 } }
        FormRow { label: "Window step"; value: Math.round(wstep.value) + " px"
                  Slider { id: wstep; from: 1; to: 100; value: 10 } }
    }
}
