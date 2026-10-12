// Standalone dialog for the Topology Optimize extension.
//
// This replaces the old Tool side panel (TopologyOptimizeTool.qml) with a
// UM.Dialog opened from Cura's Extensions menu, so it works on an empty
// build plate -- a Tool's toolbar button only shows up when a scene object
// is selected, which was a real usability wart for something that
// generates a brand-new model rather than editing an existing one.
//
// Binding pattern follows Cura's own PostProcessingPlugin.qml: the Python
// side is passed in as the plain context property `manager` (see
// TopologyOptimizeExtension.py's showPopup(): createQmlComponent(path,
// {"manager": self})), so everything here calls manager.getX() /
// manager.setX(value) directly -- there is no UM.Controller.properties
// mechanism outside an actual Tool subclass.
//
// Editable number fields set their initial text once (Component.onCompleted)
// and only write back on editingFinished, rather than binding `text` live --
// the Python side emits statusChanged every ~1.5s while a run is in
// progress (to push status/progress updates), and a live two-way text
// binding would fight the user's typing every time that fires.
// Status/progress labels use manager.statusText etc. directly, which is a
// real pyqtProperty(NOTIFY=statusChanged), so those stay live safely.
//
// NOTE: text elements use UM.Label (not the plain QtQuick.Controls Label).
// UM.Label sets color: UM.Theme.getColor("text"); a plain Label has no
// color set here and Cura's global style leaves it effectively invisible
// against this background (this bit us once already on the old Tool panel).

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import UM 1.5 as UM
import Cura 1.0 as Cura

UM.Dialog
{
    id: dialog
    title: "Shaft-coupling topology optimizer"
    width: 380 * screenScaleFactor
    height: 620 * screenScaleFactor
    minimumWidth: 340 * screenScaleFactor
    minimumHeight: 420 * screenScaleFactor
    backgroundColor: UM.Theme.getColor("main_background")

    property bool busy: manager.isBusy

    function numField(setterName, value)
    {
        manager[setterName](value)
    }

    ScrollView
    {
        anchors.fill: parent
        clip: true

        Column
        {
            id: column
            width: dialog.width - 2 * UM.Theme.getSize("default_margin").width
            spacing: UM.Theme.getSize("default_margin").height / 2

            RowLayout
            {
                width: parent.width
                UM.Label { text: "Server URL"; Layout.preferredWidth: 120 }
                TextField
                {
                    id: serverUrlField
                    Layout.fillWidth: true
                    enabled: !dialog.busy
                    Component.onCompleted: text = manager.getServerUrl()
                    onEditingFinished: manager.setServerUrl(text)
                }
            }

            UM.Label { text: "Domain size (m)"; font.bold: true }
            RowLayout
            {
                width: parent.width
                TextField { id: lxField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getDomainLx()
                    onEditingFinished: manager.setDomainLx(parseFloat(text)) }
                TextField { id: lyField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getDomainLy()
                    onEditingFinished: manager.setDomainLy(parseFloat(text)) }
                TextField { id: lzField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getDomainLz()
                    onEditingFinished: manager.setDomainLz(parseFloat(text)) }
            }
            UM.Label { text: "1 m here becomes 1000 mm in the printed model -- keep this near your real part size"; font.italic: true; font.pixelSize: 10; wrapMode: Text.WordWrap; width: parent.width }

            UM.Label { text: "Mesh resolution (nx, ny, nz)"; font.bold: true }
            RowLayout
            {
                width: parent.width
                TextField { id: nxField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getMeshNx()
                    onEditingFinished: manager.setMeshNx(parseInt(text)) }
                TextField { id: nyField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getMeshNy()
                    onEditingFinished: manager.setMeshNy(parseInt(text)) }
                TextField { id: nzField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getMeshNz()
                    onEditingFinished: manager.setMeshNz(parseInt(text)) }
            }
            UM.Label { text: "Normal Run website limit: 20 x 15 x 15, 100 iterations"; font.italic: true; font.pixelSize: 10 }

            RowLayout
            {
                width: parent.width
                UM.Label { text: "Coupling radius (m)"; Layout.preferredWidth: 140 }
                TextField { id: radiusField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getCouplingRadius()
                    onEditingFinished: manager.setCouplingRadius(parseFloat(text)) }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Solid interface length (m)"; Layout.preferredWidth: 140 }
                TextField { id: interfaceField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getInterfaceLength()
                    onEditingFinished: manager.setInterfaceLength(parseFloat(text)) }
            }

            UM.Label { text: "Loading"; font.bold: true }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Force (N)"; Layout.preferredWidth: 140 }
                TextField { id: forceField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getForceMagnitude()
                    onEditingFinished: manager.setForceMagnitude(parseFloat(text)) }
            }
            RowLayout
            {
                width: parent.width
                CheckBox
                {
                    id: torqueCheck
                    text: "Torque enabled"
                    enabled: !dialog.busy
                    Component.onCompleted: checked = manager.getTorqueEnabled() === true
                    onCheckedChanged: manager.setTorqueEnabled(checked)
                }
                TextField { id: torqueField; Layout.fillWidth: true; enabled: !dialog.busy && torqueCheck.checked
                    Component.onCompleted: text = manager.getTorqueMagnitude()
                    onEditingFinished: manager.setTorqueMagnitude(parseFloat(text)) }
            }

            UM.Label { text: "SIMP settings"; font.bold: true }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Volume fraction"; Layout.preferredWidth: 140 }
                TextField { id: volFracField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getVolumeFraction()
                    onEditingFinished: manager.setVolumeFraction(parseFloat(text)) }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Penalty"; Layout.preferredWidth: 140 }
                TextField { id: penaltyField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getPenalty()
                    onEditingFinished: manager.setPenalty(parseFloat(text)) }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Filter radius (m)"; Layout.preferredWidth: 140 }
                TextField { id: filterField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getFilterRadius()
                    onEditingFinished: manager.setFilterRadius(parseFloat(text)) }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Iterations"; Layout.preferredWidth: 140 }
                TextField { id: iterField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getIterations()
                    onEditingFinished: manager.setIterations(parseInt(text)) }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Run mode"; Layout.preferredWidth: 140 }
                ComboBox
                {
                    id: runModeBox
                    Layout.fillWidth: true
                    enabled: !dialog.busy
                    model: ["quick", "normal"]
                    Component.onCompleted:
                    {
                        var v = manager.getRunMode()
                        currentIndex = model.indexOf(v) >= 0 ? model.indexOf(v) : 1
                    }
                    onActivated: manager.setRunMode(model[currentIndex])
                }
            }

            UM.Label { text: "STL export"; font.bold: true }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Density threshold"; Layout.preferredWidth: 140 }
                ComboBox
                {
                    id: thresholdBox
                    Layout.fillWidth: true
                    enabled: !dialog.busy
                    model: ["0.1", "0.3", "0.5"]
                    Component.onCompleted: currentIndex = 1
                    onActivated: manager.setStlThreshold(parseFloat(model[currentIndex]))
                }
            }
            RowLayout
            {
                width: parent.width
                UM.Label { text: "Smoothing iterations"; Layout.preferredWidth: 140 }
                TextField { id: smoothField; Layout.fillWidth: true; enabled: !dialog.busy
                    Component.onCompleted: text = manager.getStlSmoothIterations()
                    onEditingFinished: manager.setStlSmoothIterations(parseInt(text)) }
            }

            Item { width: 1; height: 8 }

            Button
            {
                text: dialog.busy ? "Running..." : "Run Optimization + Load Result"
                enabled: !dialog.busy
                onClicked: manager.runRequested()
            }

            UM.Label
            {
                width: parent.width
                wrapMode: Text.WordWrap
                text: manager.statusText
            }
            UM.Label
            {
                width: parent.width
                wrapMode: Text.WordWrap
                visible: manager.progressTotal > 0 && dialog.busy
                text: "Progress: " + manager.progressCurrent + " / " + manager.progressTotal
            }
            UM.Label
            {
                width: parent.width
                wrapMode: Text.WordWrap
                color: "#c0392b"
                visible: manager.errorText !== ""
                text: manager.errorText
            }

            Item { width: 1; height: 8 }
        }
    }
}
