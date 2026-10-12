# Converted from a UM.Tool.Tool subclass to a UM.Extension.Extension.
#
# Why: a Tool's toolbar button only appears in Cura when a scene object is
# selected (Cura's own translate/scale/rotate/mirror tools work the same
# way) -- fine for tools that manipulate an existing model, but this plugin
# GENERATES a new model, so requiring an unrelated placeholder object on the
# plate first is a real usability wart. An Extension instead adds an entry
# under Cura's "Extensions" menu that is always available, empty scene or
# not, and shows its own standalone dialog (UM.Dialog) rather than a
# selection-bound tool panel.
#
# Reference pattern verified against Ultimaker/Cura's own
# plugins/PostProcessingPlugin (PostProcessingPlugin.py + __init__.py +
# PostProcessingPlugin.qml) on GitHub:
#   - __init__(self, parent=None): QObject.__init__(self, parent);
#     Extension.__init__(self)
#   - self.addMenuItem(name, callback) adds the Extensions-menu entry.
#   - showPopup() builds the qml path via
#     PluginRegistry.getInstance().getPluginPath(<plugin id>) and creates
#     the window with CuraApplication.getInstance().createQmlComponent(
#     path, {"manager": self}), then self._dialog.show().
#   - The QML side talks to this object as `manager.<method>()` (a plain
#     context property), not UM.Controller.properties.getValue(...) --
#     that binding only exists for the active Tool.
#   - getMetaData() in __init__.py returns {} and register() returns
#     {"extension": TopologyOptimizeExtension()} (see PostProcessingPlugin's
#     own __init__.py -- an Extension needs no "tool"/"extension" keys of
#     its own in plugin.json).
#
# This does NOT reimplement the SIMP optimizer or the marching-cubes STL
# export inside Cura's own Python -- it stays a thin HTTP client to the
# already-tested Flask API (/api/optimize/start, /api/optimize/status/<id>,
# /api/optimize/export_stl/<id>), same as web/app.js and the previous Tool
# version. Once the STL is ready it downloads it and hands it to Cura's own
# file loader (CuraApplication.readLocalFile), which ADDS the result to the
# scene rather than replacing anything already there.

import json
import os
import tempfile
import urllib.error
import urllib.request
from typing import cast

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtProperty, pyqtSignal, pyqtSlot

from UM.Extension import Extension
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry

from cura.CuraApplication import CuraApplication

DEFAULT_SERVER_URL = "http://127.0.0.1:18080"
POLL_INTERVAL_MS = 1500
HTTP_TIMEOUT_S = 15

# Must match this plugin's folder name under Cura's plugins directory --
# that folder name is the plugin id PluginRegistry.getPluginPath() expects.
PLUGIN_ID = "TopologyOptimizeTool"


class TopologyOptimizeExtension(QObject, Extension):
    statusChanged = pyqtSignal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        Extension.__init__(self)

        self.setMenuName("Topology Optimize (SIMP)")
        self.addMenuItem("Open panel...", self.showPopup)

        self._dialog = None

        # ---- parameters (mirror web/index.html + app.py's _default_config) ----
        # Scaled to a printable shaft coupling (~60mm long, ~12mm across),
        # keeping the web app's original proportions -- see the comment in
        # the previous Tool version for the full reasoning. 1m of domain
        # size becomes 1000mm in the exported STL (mesh_export.export_stl's
        # unit_scale), so these need to stay near the real part size.
        self._server_url = DEFAULT_SERVER_URL
        self._domain_lx = 0.06
        self._domain_ly = 0.012
        self._domain_lz = 0.012
        self._mesh_nx = 20
        self._mesh_ny = 15
        self._mesh_nz = 15
        self._coupling_radius = 0.006
        self._interface_length = 0.0048
        self._force_magnitude = -10000.0
        self._torque_enabled = True
        self._torque_magnitude = 100.0
        self._volume_fraction = 0.2
        self._penalty = 3.0
        self._filter_radius = 0.0012
        self._iterations = 60
        self._run_mode = "normal"
        self._stl_threshold = 0.3
        self._stl_smooth_iterations = 0

        # ---- run state (read-only from QML, exposed as real pyqtProperty
        # so the dialog's status/progress labels update live) ----
        self._is_busy = False
        self._status_text = "Ready"
        self._error_text = ""
        self._progress_current = 0
        self._progress_total = 0
        self._task_id = None

        self._poll_timer = QTimer()
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------
    # Menu entry -> standalone dialog (not tied to any scene selection).
    # ------------------------------------------------------------------

    def showPopup(self):
        if self._dialog is None:
            plugin_path = PluginRegistry.getInstance().getPluginPath(PLUGIN_ID)
            if plugin_path is None:
                Logger.log("e", "TopologyOptimizeExtension: could not resolve plugin path")
                return
            path = os.path.join(cast(str, plugin_path), "TopologyOptimizeExtension.qml")
            self._dialog = CuraApplication.getInstance().createQmlComponent(path, {"manager": self})
            if self._dialog is None:
                Logger.log("e", "TopologyOptimizeExtension: failed to create dialog from " + path)
                return
        self._dialog.show()

    # ------------------------------------------------------------------
    # Read-only run state, exposed as real Qt properties (NOTIFY =
    # statusChanged) so the dialog's bindings update live while a run is
    # in progress.
    # ------------------------------------------------------------------

    @pyqtProperty(bool, notify=statusChanged)
    def isBusy(self):
        return self._is_busy

    @pyqtProperty(str, notify=statusChanged)
    def statusText(self):
        return self._status_text

    @pyqtProperty(str, notify=statusChanged)
    def errorText(self):
        return self._error_text

    @pyqtProperty(int, notify=statusChanged)
    def progressCurrent(self):
        return self._progress_current

    @pyqtProperty(int, notify=statusChanged)
    def progressTotal(self):
        return self._progress_total

    # ------------------------------------------------------------------
    # Plain get/set slot pairs for every editable parameter. The QML side
    # calls these explicitly (Component.onCompleted / onEditingFinished),
    # the same one-shot-read / write-on-commit pattern as the previous
    # Tool version -- deliberately NOT a live two-way property binding,
    # since propertyChanged-style live updates would fight the user's
    # typing while a run is in progress.
    # ------------------------------------------------------------------

    @pyqtSlot(result=str)
    def getServerUrl(self):
        return self._server_url

    @pyqtSlot(str)
    def setServerUrl(self, value):
        self._server_url = str(value)

    @pyqtSlot(result=float)
    def getDomainLx(self):
        return self._domain_lx

    @pyqtSlot(float)
    def setDomainLx(self, value):
        self._domain_lx = float(value)

    @pyqtSlot(result=float)
    def getDomainLy(self):
        return self._domain_ly

    @pyqtSlot(float)
    def setDomainLy(self, value):
        self._domain_ly = float(value)

    @pyqtSlot(result=float)
    def getDomainLz(self):
        return self._domain_lz

    @pyqtSlot(float)
    def setDomainLz(self, value):
        self._domain_lz = float(value)

    @pyqtSlot(result=int)
    def getMeshNx(self):
        return self._mesh_nx

    @pyqtSlot(int)
    def setMeshNx(self, value):
        self._mesh_nx = int(value)

    @pyqtSlot(result=int)
    def getMeshNy(self):
        return self._mesh_ny

    @pyqtSlot(int)
    def setMeshNy(self, value):
        self._mesh_ny = int(value)

    @pyqtSlot(result=int)
    def getMeshNz(self):
        return self._mesh_nz

    @pyqtSlot(int)
    def setMeshNz(self, value):
        self._mesh_nz = int(value)

    @pyqtSlot(result=float)
    def getCouplingRadius(self):
        return self._coupling_radius

    @pyqtSlot(float)
    def setCouplingRadius(self, value):
        self._coupling_radius = float(value)

    @pyqtSlot(result=float)
    def getInterfaceLength(self):
        return self._interface_length

    @pyqtSlot(float)
    def setInterfaceLength(self, value):
        self._interface_length = float(value)

    @pyqtSlot(result=float)
    def getForceMagnitude(self):
        return self._force_magnitude

    @pyqtSlot(float)
    def setForceMagnitude(self, value):
        self._force_magnitude = float(value)

    @pyqtSlot(result=bool)
    def getTorqueEnabled(self):
        return self._torque_enabled

    @pyqtSlot(bool)
    def setTorqueEnabled(self, value):
        self._torque_enabled = bool(value)

    @pyqtSlot(result=float)
    def getTorqueMagnitude(self):
        return self._torque_magnitude

    @pyqtSlot(float)
    def setTorqueMagnitude(self, value):
        self._torque_magnitude = float(value)

    @pyqtSlot(result=float)
    def getVolumeFraction(self):
        return self._volume_fraction

    @pyqtSlot(float)
    def setVolumeFraction(self, value):
        self._volume_fraction = float(value)

    @pyqtSlot(result=float)
    def getPenalty(self):
        return self._penalty

    @pyqtSlot(float)
    def setPenalty(self, value):
        self._penalty = float(value)

    @pyqtSlot(result=float)
    def getFilterRadius(self):
        return self._filter_radius

    @pyqtSlot(float)
    def setFilterRadius(self, value):
        self._filter_radius = float(value)

    @pyqtSlot(result=int)
    def getIterations(self):
        return self._iterations

    @pyqtSlot(int)
    def setIterations(self, value):
        self._iterations = int(value)

    @pyqtSlot(result=str)
    def getRunMode(self):
        return self._run_mode

    @pyqtSlot(str)
    def setRunMode(self, value):
        self._run_mode = str(value)

    @pyqtSlot(result=float)
    def getStlThreshold(self):
        return self._stl_threshold

    @pyqtSlot(float)
    def setStlThreshold(self, value):
        self._stl_threshold = float(value)

    @pyqtSlot(result=int)
    def getStlSmoothIterations(self):
        return self._stl_smooth_iterations

    @pyqtSlot(int)
    def setStlSmoothIterations(self, value):
        self._stl_smooth_iterations = int(value)

    # ---- the "Run" button calls this directly now instead of setting a
    # trigger property (there's no Tool/Controller property mechanism to
    # piggyback on outside a Tool subclass). ----

    @pyqtSlot()
    def runRequested(self):
        if not self._is_busy:
            self._startRun()

    # ------------------------------------------------------------------
    # HTTP helpers -- thin wrappers over the Flask API app.py exposes.
    # ------------------------------------------------------------------

    def _api_post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._server_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _api_get(self, path):
        req = urllib.request.Request(self._server_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download(self, path, dest_path):
        req = urllib.request.Request(self._server_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S * 2) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)

    def _extract_error(self, exc):
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = json.loads(exc.read().decode("utf-8"))
                return body.get("error", str(exc))
            except Exception:
                return str(exc)
        return str(exc)

    # ------------------------------------------------------------------
    # Run sequence: start -> poll -> export_stl -> download -> load into scene.
    # Mirrors web/app.js's startOptimization()/pollTask()/exportStl() flow
    # one-for-one, just driven by a QTimer instead of setInterval().
    # ------------------------------------------------------------------

    def _buildConfig(self):
        return {
            "mesh": {
                "Lx": self._domain_lx, "Ly": self._domain_ly, "Lz": self._domain_lz,
                "nx": self._mesh_nx, "ny": self._mesh_ny, "nz": self._mesh_nz,
            },
            "geometry": {
                "type": "cylindrical_coupling",
                "radius": self._coupling_radius,
                "interface_length": self._interface_length,
            },
            "boundary_conditions": [{"axis": "x", "coord_mode": "min"}],
            "loading_condition": {
                "type": "distributed", "axis": "x", "coord_mode": "max",
                "direction": "y", "magnitude": self._force_magnitude,
            },
            "torque_condition": {
                "enabled": self._torque_enabled, "axis": "x", "coord_mode": "max",
                "magnitude": self._torque_magnitude,
            },
            "optimizer": {
                "method": "shaft_simp", "initial_density": 0.2,
                "volume_fraction": self._volume_fraction, "penalty": self._penalty,
                "filter_radius": self._filter_radius, "iterations": self._iterations,
            },
            "run_mode": self._run_mode,
        }

    def _startRun(self):
        self._is_busy = True
        self._error_text = ""
        self._status_text = f"Submitting to {self._server_url} ..."
        self.statusChanged.emit()

        try:
            task = self._api_post("/api/optimize/start", self._buildConfig())
        except Exception as exc:
            self._fail(
                f"Could not reach {self._server_url} -- is `python app.py` "
                f"running in the shaft conda env? ({self._extract_error(exc)})"
            )
            return

        self._task_id = task.get("task_id")
        self._status_text = f"Running (task {self._task_id}) ..."
        self.statusChanged.emit()
        self._poll_timer.start()

    def _poll(self):
        try:
            data = self._api_get(f"/api/optimize/status/{self._task_id}")
        except Exception as exc:
            self._poll_timer.stop()
            self._fail(f"Lost connection while polling: {self._extract_error(exc)}")
            return

        status = data.get("status")
        if status == "running":
            progress = data.get("progress") or {}
            self._progress_current = progress.get("current", 0)
            self._progress_total = progress.get("total", 0)
            self._status_text = f"Optimizing... iteration {self._progress_current}/{self._progress_total}"
            self.statusChanged.emit()
        elif status == "completed":
            self._poll_timer.stop()
            self._status_text = "Optimization complete -- exporting STL..."
            self.statusChanged.emit()
            self._exportAndLoad()
        elif status == "error":
            self._poll_timer.stop()
            self._fail(data.get("error") or "Optimization failed on the server")

    def _exportAndLoad(self):
        try:
            result = self._api_post(
                f"/api/optimize/export_stl/{self._task_id}",
                {"threshold": self._stl_threshold, "smooth_iterations": self._stl_smooth_iterations},
            )
        except Exception as exc:
            self._fail(f"STL export failed: {self._extract_error(exc)}")
            return

        if not result.get("is_watertight", False):
            Logger.log(
                "w",
                "TopologyOptimizeExtension: exported STL is not watertight -- "
                "loading it anyway, inspect it in Cura before slicing.",
            )

        dest_dir = tempfile.mkdtemp(prefix="topo_opt_")
        dest_path = os.path.join(dest_dir, result["file_name"])
        try:
            self._download(result["download_url"], dest_path)
        except Exception as exc:
            self._fail(f"Could not download the STL: {self._extract_error(exc)}")
            return

        CuraApplication.getInstance().readLocalFile(QUrl.fromLocalFile(dest_path))

        watertight_note = "watertight" if result.get("is_watertight") else "NOT watertight -- check before printing"
        self._status_text = (
            f"Loaded {result['file_name']} into the scene "
            f"({result['n_vertices']} vertices, {result['n_faces']} faces, {watertight_note})"
        )
        self._is_busy = False
        self.statusChanged.emit()

    def _fail(self, message):
        Logger.log("e", f"TopologyOptimizeExtension: {message}")
        self._error_text = message
        self._status_text = "Error"
        self._is_busy = False
        self.statusChanged.emit()
