"""Adapter between the shaft-coupling website and the teacher-provided baseline code.

IMPORTANT: Files under teacher_core are intentionally left unchanged. This adapter
translates website parameters into the APIs exposed by those baseline modules.
"""
from __future__ import annotations

import math
import numpy as np


def _axis_index(value):
    if isinstance(value, int):
        return value
    return {"x": 0, "y": 1, "z": 2}[str(value).lower()]


def _face_center(mesh, axis, mode="max"):
    lengths = [float(mesh["Lx"]), float(mesh["Ly"]), float(mesh["Lz"])]
    c = [lengths[0] / 2.0, lengths[1] / 2.0, lengths[2] / 2.0]
    a = _axis_index(axis)
    c[a] = 0.0 if str(mode).lower() == "min" else lengths[a]
    return tuple(c)


def _coord(mesh, axis, mode):
    a = _axis_index(axis)
    lengths = [float(mesh["Lx"]), float(mesh["Ly"]), float(mesh["Lz"])]
    return 0.0 if str(mode).lower() == "min" else lengths[a]


def _apply_teacher_loads(fem, cfg, torch_mode=False):
    mesh = cfg["mesh"]
    load = cfg["loading_condition"]
    direction = _axis_index(load.get("direction", "y"))
    load_type = str(load.get("type", "distributed")).lower()
    if load_type == "distributed":
        a = _axis_index(load.get("axis", "x"))
        coord = _coord(mesh, a, load.get("coord_mode", "max"))
        fem.add_distributed_load(axis=a, coord=coord, direction=direction, total=float(load["magnitude"]))
    elif load_type == "point":
        loc = load.get("location") or [mesh["Lx"], mesh["Ly"] / 2.0, mesh["Lz"] / 2.0]
        fem.add_point_load(location=tuple(map(float, loc)), direction=direction, magnitude=float(load["magnitude"]))
    else:
        raise ValueError("Teacher baseline supports distributed or point force loading")

    torque = cfg.get("torque_condition", {})
    if torque.get("enabled", False) and abs(float(torque.get("magnitude", 0.0))) > 0.0:
        axis = _axis_index(torque.get("axis", "x"))
        mode = torque.get("coord_mode", "max")
        center = _face_center(mesh, axis, mode)
        # Preserve teacher implementation/signature: center, axis, magnitude.
        fem.add_torque_load(center=center, axis=axis, magnitude=float(torque["magnitude"]))


def run_teacher_numpy(cfg, progress_callback=None):
    from teacher_core.fem3d_numpy import HexFEMSolver3D
    from teacher_core.simp_numpy import SIMPOptimizer

    mesh, mat, opt = cfg["mesh"], cfg["material"], cfg["optimizer"]
    fem = HexFEMSolver3D(E_mod=float(mat["E_mod"]), nu=float(mat["nu"]), Emin=float(mat["Emin"]), penalty=float(opt["penalty"]))
    fem.set_mesh(Lx=float(mesh["Lx"]), Ly=float(mesh["Ly"]), Lz=float(mesh["Lz"]), nx=int(mesh["nx"]), ny=int(mesh["ny"]), nz=int(mesh["nz"]))
    for bc in cfg["boundary_conditions"]:
        a = _axis_index(bc.get("axis", "x"))
        fem.fix_face(axis=a, coord=_coord(mesh, a, bc.get("coord_mode", "min")))
    _apply_teacher_loads(fem, cfg)

    optimizer = SIMPOptimizer(
        fem_solver=fem,
        initial_density=float(opt["initial_density"]),
        volume_fraction=float(opt["volume_fraction"]),
        penalty=float(opt["penalty"]),
        filter_radius=float(opt["filter_radius"]),
    )
    # Teacher SIMP does not expose a callback. Run one iteration at a time while
    # retaining optimizer state; this does not alter teacher_core itself.
    n = int(opt["iterations"])
    for i in range(n):
        optimizer.optimize(n_iterations=1, verbose=False)
        if progress_callback:
            progress_callback(i)
    history = dict(optimizer.history)
    history["iteration"] = list(range(len(history.get("compliance", []))))
    density = np.asarray(optimizer.density, dtype=float)
    return {
        "engine": "teacher_numpy_simp",
        "engine_label": "Teacher NumPy SIMP (baseline rectangular mesh)",
        "density": density,
        "history": history,
        "final_compliance": float(history["compliance"][-1]),
        "final_volume": float(history["volume"][-1]),
        "nodes": np.asarray(fem.nodes_np),
        "elems": np.asarray(fem.elems_t),
        "notes": [
            "Uses teacher_core/fem3d_numpy.py and teacher_core/simp_numpy.py without modifying them.",
            "This teacher baseline uses its original rectangular structured mesh; the website cylindrical mask is not injected into the teacher code.",
        ],
    }


def run_teacher_nn(cfg, progress_callback=None):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Teacher NN mode requires PyTorch in the active conda environment") from exc

    from teacher_core.fem3d_torch import HexFEMSolver3D_Torch
    from teacher_core.neural_network_topology_optimization import NNOptimizer

    class SimpleDensityNetwork(nn.Module):
        def __init__(self, n_elems, device):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(n_elems, device=device))
        def forward(self):
            return torch.sigmoid(self.logits)

    mesh, mat, opt = cfg["mesh"], cfg["material"], cfg["optimizer"]
    requested_device = str(opt.get("nn_device", "auto")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch.cuda.is_available() is False")
    device = "cuda" if (requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available())) else "cpu"

    fem = HexFEMSolver3D_Torch(E_mod=float(mat["E_mod"]), nu=float(mat["nu"]), Emin=float(mat["Emin"]), penalty=float(opt["penalty"]), device=device)
    fem.set_mesh(Lx=float(mesh["Lx"]), Ly=float(mesh["Ly"]), Lz=float(mesh["Lz"]), nx=int(mesh["nx"]), ny=int(mesh["ny"]), nz=int(mesh["nz"]))
    for bc in cfg["boundary_conditions"]:
        a = _axis_index(bc.get("axis", "x"))
        fem.fix_face(axis=a, coord=_coord(mesh, a, bc.get("coord_mode", "min")))
    _apply_teacher_loads(fem, cfg, torch_mode=True)

    model = SimpleDensityNetwork(fem.n_elems, device=device)
    optimizer = NNOptimizer(
        fem_solver=fem,
        structure_model=model,
        volume_fraction=float(opt["volume_fraction"]),
        penalty=float(opt["penalty"]),
    )
    # Preserve the teacher optimizer itself. It runs the requested iteration count.
    result = optimizer.optimize(n_iterations=int(opt["iterations"]), verbose=False)
    if progress_callback:
        progress_callback(int(opt["iterations"]) - 1)
    density = np.asarray(result["density"], dtype=float)
    hist = result["history"]
    # NN teacher history has no density_change; add a NaN series only for website plotting.
    if "density_change" not in hist:
        hist = dict(hist)
        hist["density_change"] = [float("nan")] * len(hist.get("iteration", []))
    nodes = fem.nodes_torch.detach().cpu().numpy()
    elems = np.asarray(fem.elems_t, dtype=int)
    return {
        "engine": "teacher_nn",
        "engine_label": f"Teacher NN / PyTorch ({device})",
        "density": density,
        "history": hist,
        "final_compliance": float(result["final_compliance"]),
        "final_volume": float(result["final_volume"]),
        "nodes": nodes,
        "elems": elems,
        "notes": [
            "Uses teacher_core/fem3d_torch.py and teacher_core/neural_network_topology_optimization.py without modifying them.",
            f"Device used: {device}.",
            "This teacher baseline uses its original rectangular structured mesh; the website cylindrical mask is not injected into the teacher code.",
        ],
    }
