import ast
import datetime as dt
import io
import os
import pickle
import random
import sys
import threading
import traceback
import types
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request, send_from_directory
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fem3d_numpy import HexFEMSolver3D
from simp_numpy import SIMPOptimizer
from teacher_adapter import run_teacher_numpy, run_teacher_nn

try:
    import mesh_export
    _MESH_EXPORT_OK = True
except ImportError:
    _MESH_EXPORT_OK = False

_OPTIONAL_IMPORTS_OK = True
try:
    import pandas as pd
    import torch
    import torch.storage
except ImportError:
    _OPTIONAL_IMPORTS_OK = False


class CustomDataset:
    def __setstate__(self, state):
        self.__dict__.update(state)


class PytorchMinMaxScaler:
    def __setstate__(self, state):
        self.__dict__.update(state)


def _cpu_load_from_bytes(binary):
    return torch.load(io.BytesIO(binary), map_location="cpu", weights_only=False)


def _prepare_unpickle_environment():
    pkg = types.ModuleType("Data_Plotting")
    sub = types.ModuleType("Data_Plotting.Shape_Plotting")
    CustomDataset.__module__ = "Data_Plotting.Shape_Plotting"
    sub.CustomDataset = CustomDataset
    pkg.Shape_Plotting = sub
    sys.modules["Data_Plotting"] = pkg
    sys.modules["Data_Plotting.Shape_Plotting"] = sub
    torch.storage._load_from_bytes = _cpu_load_from_bytes


class DataModelService:
    """Optional service for dataset/VAE browsing. Loads gracefully."""

    def __init__(self, dataset_path, vae_path):
        self.dataset_path = dataset_path
        self.vae_path = vae_path
        self.dataset_obj = None
        self.train_subset = None
        self.test_subset = None
        self.dataset_df = None
        self.tensor_matrix = None
        self.model_state = OrderedDict()
        self.scaler_min = None
        self.scaler_max = None
        self.available = False
        self._try_load()

    def _try_load(self):
        if not _OPTIONAL_IMPORTS_OK:
            return
        if not os.path.isfile(self.dataset_path) or not os.path.isfile(self.vae_path):
            return
        try:
            self.load()
            self.available = True
        except Exception:
            traceback.print_exc()

    def load(self):
        _prepare_unpickle_environment()
        with open(self.dataset_path, "rb") as f:
            self.dataset_obj = pickle.load(f)
        self.train_subset = self.dataset_obj["train"]
        self.test_subset = self.dataset_obj["test"]
        self.dataset_df = self.train_subset.dataset.df
        if hasattr(self.train_subset.dataset, "all_stiffness_values"):
            values = self.train_subset.dataset.all_stiffness_values
            self.tensor_matrix = np.array(values, dtype=np.float32).reshape(-1, 9)
        else:
            self.tensor_matrix = self._build_tensor_matrix_from_df()
        with open(self.vae_path, "rb") as f:
            vae_obj = pickle.load(f)
        self.model_state = vae_obj["model"]
        scaler = vae_obj.get("scaler")
        if scaler is not None and hasattr(scaler, "min_vals") and hasattr(scaler, "max_vals"):
            self.scaler_min = scaler.min_vals.detach().cpu().numpy().astype(np.float32)
            self.scaler_max = scaler.max_vals.detach().cpu().numpy().astype(np.float32)

    def _build_tensor_matrix_from_df(self):
        tensors = []
        for item in self.dataset_df["Elasticity Tensor"].tolist():
            arr = np.array(ast.literal_eval(item), dtype=np.float32).reshape(9)
            tensors.append(arr)
        return np.vstack(tensors)

    def _get_subset(self, split):
        return self.train_subset if split == "train" else self.test_subset

    def _parse_shape(self, value):
        return np.array(ast.literal_eval(value), dtype=np.float32)

    def _parse_tensor(self, value):
        return np.array(ast.literal_eval(value), dtype=np.float32)

    def stats(self):
        return {
            "dataset_rows": int(len(self.dataset_df)),
            "train_size": int(len(self.train_subset)),
            "test_size": int(len(self.test_subset)),
            "columns": list(self.dataset_df.columns),
            "shape_param_count": 784,
            "tensor_param_count": 9,
            "model_param_count": int(sum(v.numel() for v in self.model_state.values())),
            "model_layer_count": int(len(self.model_state)),
        }

    def model_info(self):
        layers = []
        for name, tensor in self.model_state.items():
            layers.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        info = {"layer_count": len(layers), "layers": layers}
        if self.scaler_min is not None and self.scaler_max is not None:
            info["scaler"] = {"min": self.scaler_min.tolist(), "max": self.scaler_max.tolist()}
        return info

    def sample(self, split, subset_index):
        subset = self._get_subset(split)
        subset_index = int(subset_index)
        if subset_index < 0 or subset_index >= len(subset):
            raise IndexError("index out of range")
        source_index = int(subset.indices[subset_index])
        row = self.dataset_df.iloc[source_index]
        shape_grid = self._parse_shape(row["Array"])
        tensor = self._parse_tensor(row["Elasticity Tensor"])
        return {
            "split": split,
            "subset_index": subset_index,
            "source_index": source_index,
            "shape": shape_grid.tolist(),
            "shape_size": list(shape_grid.shape),
            "tensor": tensor.tolist(),
        }

    def random_sample(self, split):
        subset = self._get_subset(split)
        idx = random.randint(0, len(subset) - 1)
        return self.sample(split, idx)

    def nearest_by_tensor(self, tensor_input, split):
        subset = self._get_subset(split)
        flat = np.array(tensor_input, dtype=np.float32).reshape(9)
        candidate_indices = np.array(subset.indices, dtype=np.int64)
        candidate_tensor = self.tensor_matrix[candidate_indices]
        dist = np.linalg.norm(candidate_tensor - flat[None, :], axis=1)
        best_local = int(np.argmin(dist))
        best_distance = float(dist[best_local])
        result = self.sample(split, best_local)
        result["distance"] = best_distance
        return result


def _axis_to_index(value):
    if isinstance(value, int):
        if value in (0, 1, 2):
            return value
    if isinstance(value, str):
        low = value.lower()
        if low in ("x", "0"):
            return 0
        if low in ("y", "1"):
            return 1
        if low in ("z", "2"):
            return 2
    raise ValueError(f"invalid axis: {value}")


def _coord_from_mode(mode, axis_index, mesh_cfg, value=None):
    mode = str(mode).lower()
    lengths = [mesh_cfg["Lx"], mesh_cfg["Ly"], mesh_cfg["Lz"]]
    if mode == "min":
        return 0.0
    if mode == "max":
        return float(lengths[axis_index])
    if mode == "value":
        return float(value)
    raise ValueError(f"invalid coord_mode: {mode}")


def _plot_3d_design(nodes, elems, density, threshold, title, output_path):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    mask = density > threshold
    active_elems = np.where(mask)[0]
    if len(active_elems) == 0:
        ax.text(0.5, 0.5, 0.5, f"No elements with ρ > {threshold}", transform=ax.transAxes, ha="center", fontsize=14)
        ax.set_title(title)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=threshold, vmax=1.0)
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    for elem_idx in active_elems:
        elem_nodes = elems[elem_idx]
        elem_coords = nodes[elem_nodes]
        elem_density = density[elem_idx]
        color = cmap(norm(elem_density))
        for face in faces:
            ax.add_collection3d(Poly3DCollection([elem_coords[face]], facecolors=color, edgecolors="k", linewidths=0.5))
    all_active_coords = nodes[elems[active_elems]].reshape(-1, 3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.set_xlim([all_active_coords[:, 0].min(), all_active_coords[:, 0].max()])
    ax.set_ylim([all_active_coords[:, 1].min(), all_active_coords[:, 1].max()])
    ax.set_zlim([all_active_coords[:, 2].min(), all_active_coords[:, 2].max()])
    ax.set_box_aspect([np.ptp(all_active_coords[:, 0]), np.ptp(all_active_coords[:, 1]), np.ptp(all_active_coords[:, 2])])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Material Density", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class OptimizationService:
    def __init__(self, runs_dir):
        self.runs_dir = runs_dir
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.tasks = {}
        os.makedirs(self.runs_dir, exist_ok=True)

    def _default_config(self):
        return {
            "mesh": {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2, "nx": 20, "ny": 10, "nz": 10},
            "geometry": {"type": "cylindrical_coupling", "radius": 0.1, "interface_length": 0.08},
            "material": {"E_mod": 200e9, "nu": 0.3, "Emin": 1e-3},
            "boundary_conditions": [{"axis": "x", "coord_mode": "min"}],
            "loading_condition": {"type": "distributed", "axis": "x", "coord_mode": "max", "direction": "y", "magnitude": -10000.0},
            "torque_condition": {"enabled": True, "axis": "x", "coord_mode": "max", "magnitude": 100.0},
            "optimizer": {"method": "shaft_simp", "nn_device": "auto", "initial_density": 0.2, "volume_fraction": 0.2, "penalty": 3.0, "filter_radius": 0.02, "iterations": 60},
            "visualization": {"thresholds": [0.1, 0.3, 0.5]},
            "run_mode": "normal",
        }

    def _merge_dict(self, base, updates):
        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _normalize_config(self, config):
        cfg = self._merge_dict(self._default_config(), config or {})
        mesh = cfg["mesh"]
        mesh["Lx"] = float(mesh["Lx"])
        mesh["Ly"] = float(mesh["Ly"])
        mesh["Lz"] = float(mesh["Lz"])
        mesh["nx"] = int(mesh["nx"])
        mesh["ny"] = int(mesh["ny"])
        mesh["nz"] = int(mesh["nz"])
        if mesh["nx"] <= 0 or mesh["ny"] <= 0 or mesh["nz"] <= 0:
            raise ValueError("网格分辨率必须为正整数")
        opt = cfg["optimizer"]
        opt["method"] = str(opt.get("method", "shaft_simp")).lower()
        if opt["method"] not in ("shaft_simp", "teacher_numpy", "teacher_nn"):
            raise ValueError("optimizer.method 必须是 shaft_simp, teacher_numpy 或 teacher_nn")
        opt["nn_device"] = str(opt.get("nn_device", "auto")).lower()
        if opt["nn_device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("nn_device 必须是 auto, cpu 或 cuda")
        opt["initial_density"] = float(opt["initial_density"])
        opt["volume_fraction"] = float(opt["volume_fraction"])
        opt["penalty"] = float(opt["penalty"])
        opt["filter_radius"] = float(opt["filter_radius"])
        opt["iterations"] = int(opt["iterations"])
        if opt["volume_fraction"] <= 0 or opt["volume_fraction"] >= 1:
            raise ValueError("volume_fraction 必须在 (0,1)")
        if opt["iterations"] <= 0:
            raise ValueError("iterations 必须为正整数")

        # V6 website run modes. These are website safety limits, not FEM theoretical limits.
        run_mode = str(cfg.get("run_mode", "normal")).lower()
        if run_mode not in ("quick", "normal"):
            raise ValueError("run_mode 必须是 quick 或 normal")
        cfg["run_mode"] = run_mode
        if run_mode == "quick":
            mesh["nx"] = min(mesh["nx"], 8)
            mesh["ny"] = min(mesh["ny"], 6)
            mesh["nz"] = min(mesh["nz"], 6)
            opt["iterations"] = min(opt["iterations"], 10)
        else:
            violations = []
            if mesh["nx"] > 20:
                violations.append(f"Elements X={mesh['nx']} > 20")
            if mesh["ny"] > 15:
                violations.append(f"Elements Y={mesh['ny']} > 15")
            if mesh["nz"] > 15:
                violations.append(f"Elements Z={mesh['nz']} > 15")
            if opt["iterations"] > 100:
                violations.append(f"Iterations={opt['iterations']} > 100")
            if violations:
                raise ValueError("Normal Run 超过网站安全上限: " + ", ".join(violations) + ". 最大为 20×15×15, 100 iterations")

        geometry = cfg.get("geometry", {})
        geometry["type"] = str(geometry.get("type", "cylindrical_coupling"))
        geometry["radius"] = float(geometry.get("radius", min(mesh["Ly"], mesh["Lz"]) / 2.0))
        geometry["interface_length"] = float(geometry.get("interface_length", 0.08))
        max_radius = min(mesh["Ly"], mesh["Lz"]) / 2.0
        if geometry["radius"] <= 0 or geometry["radius"] > max_radius + 1e-12:
            raise ValueError(f"coupling radius 必须在 (0, {max_radius:.6g}] m")
        if geometry["interface_length"] < 0 or geometry["interface_length"] >= mesh["Lx"] / 2.0:
            raise ValueError("interface_length 必须 >= 0 且小于轴长的一半")
        cfg["geometry"] = geometry

        material = cfg["material"]
        material["E_mod"] = float(material["E_mod"])
        material["nu"] = float(material["nu"])
        material["Emin"] = float(material["Emin"])
        if not isinstance(cfg["boundary_conditions"], list) or len(cfg["boundary_conditions"]) == 0:
            raise ValueError("至少需要一个边界条件")
        load = cfg["loading_condition"]
        load["magnitude"] = float(load["magnitude"])
        torque = cfg.get("torque_condition", {})
        torque["enabled"] = bool(torque.get("enabled", False))
        torque["magnitude"] = float(torque.get("magnitude", 0.0))
        cfg["torque_condition"] = torque
        thresholds = cfg.get("visualization", {}).get("thresholds", [0.1, 0.3, 0.5])
        cfg["visualization"]["thresholds"] = [float(t) for t in thresholds]
        cfg["save_intermediate"] = bool(cfg.get("save_intermediate", False))
        interval = int(cfg.get("snapshot_interval", 10))
        cfg["snapshot_interval"] = max(1, interval)
        return cfg

    def start(self, config):
        cfg = self._normalize_config(config)
        task_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        task = {
            "task_id": task_id,
            "status": "queued",
            "created_at": dt.datetime.now().isoformat(),
            "started_at": None,
            "finished_at": None,
            "input_summary": cfg,
            "result": None,
            "error": None,
        }
        with self.lock:
            self.tasks[task_id] = task
        self.executor.submit(self._run_task, task_id, cfg)
        return task

    def get(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            return dict(task)

    def _update_task(self, task_id, fields):
        with self.lock:
            task = self.tasks[task_id]
            task.update(fields)

    def _run_task(self, task_id, cfg):
        self._update_task(task_id, {"status": "running", "started_at": dt.datetime.now().isoformat(), "progress": {"current": 0, "total": cfg["optimizer"]["iterations"]}})
        try:
            result = self._execute(task_id, cfg)
            self._update_task(task_id, {"status": "completed", "finished_at": dt.datetime.now().isoformat(), "result": result, "progress": None})
        except Exception as exc:
            self._update_task(task_id, {"status": "error", "finished_at": dt.datetime.now().isoformat(), "error": str(exc), "progress": None})

    def _execute(self, task_id, cfg):
        run_dir = os.path.join(self.runs_dir, task_id)
        os.makedirs(run_dir, exist_ok=True)
        mesh = cfg["mesh"]
        material = cfg["material"]
        optimizer_cfg = cfg["optimizer"]
        geometry = cfg["geometry"]
        method = optimizer_cfg.get("method", "shaft_simp")
        total_iters = optimizer_cfg["iterations"]
        save_intermediate = cfg.get("save_intermediate", False)
        snap_interval = cfg.get("snapshot_interval", 10)
        snapshot_files = []

        def progress_only(it):
            self._update_task(task_id, {"progress": {"current": min(it + 1, total_iters), "total": total_iters}})

        if method == "shaft_simp":
            fem = HexFEMSolver3D(E_mod=material["E_mod"], nu=material["nu"], Emin=material["Emin"], penalty=optimizer_cfg["penalty"])
            fem.set_mesh(Lx=mesh["Lx"], Ly=mesh["Ly"], Lz=mesh["Lz"], nx=mesh["nx"], ny=mesh["ny"], nz=mesh["nz"])
            if geometry.get("type") == "cylindrical_coupling":
                fem.set_cylindrical_coupling_domain(radius=geometry["radius"], interface_length=geometry["interface_length"], axis=0)
            for bc in cfg["boundary_conditions"]:
                axis_idx = _axis_to_index(bc.get("axis", "x"))
                coord = _coord_from_mode(bc.get("coord_mode", "min"), axis_idx, mesh, bc.get("coord_value"))
                fem.fix_face(axis=axis_idx, coord=coord)
            load = cfg["loading_condition"]
            direction_idx = _axis_to_index(load.get("direction", "y"))
            load_type = str(load.get("type", "distributed")).lower()
            if load_type == "distributed":
                axis_idx = _axis_to_index(load.get("axis", "x"))
                coord = _coord_from_mode(load.get("coord_mode", "max"), axis_idx, mesh, load.get("coord_value"))
                fem.add_distributed_load(axis=axis_idx, coord=coord, direction=direction_idx, total=load["magnitude"])
            elif load_type == "point":
                location = load.get("location", [mesh["Lx"], mesh["Ly"] / 2.0, mesh["Lz"] / 2.0])
                fem.add_point_load(location=[float(v) for v in location], direction=direction_idx, magnitude=load["magnitude"])
            else:
                raise ValueError("loading_condition.type 必须是 distributed 或 point")

            torque_cfg = cfg.get("torque_condition", {})
            if torque_cfg.get("enabled", False) and abs(float(torque_cfg.get("magnitude", 0.0))) > 0.0:
                torque_axis = _axis_to_index(torque_cfg.get("axis", "x"))
                torque_coord = _coord_from_mode(torque_cfg.get("coord_mode", "max"), torque_axis, mesh, torque_cfg.get("coord_value"))
                fem.add_torque_load(axis=torque_axis, coord=torque_coord, torque=float(torque_cfg.get("magnitude", 0.0)))

            simp = SIMPOptimizer(
                fem_solver=fem,
                initial_density=optimizer_cfg["initial_density"],
                volume_fraction=optimizer_cfg["volume_fraction"],
                penalty=optimizer_cfg["penalty"],
                filter_radius=optimizer_cfg["filter_radius"],
                design_mask=fem.design_elem_mask,
                passive_solid_mask=fem.passive_solid_mask,
                passive_void_mask=~fem.physical_elem_mask,
            )

            def on_iteration(it):
                progress_only(it)
                iter_num = it + 1
                if save_intermediate and iter_num % snap_interval == 0 and iter_num < total_iters:
                    snap_name = f"snapshot_iter_{iter_num:04d}.png"
                    snap_path = os.path.join(run_dir, snap_name)
                    _plot_3d_design(fem.nodes_np, fem.elems_t, simp.density, 0.3, f"Iteration {iter_num}", snap_path)
                    snapshot_files.append({"iteration": iter_num, "filename": snap_name})

            raw = simp.optimize(n_iterations=total_iters, verbose=False, callback=on_iteration)
            run_data = {
                "engine": "shaft_simp",
                "engine_label": "Shaft SIMP (website cylindrical coupling)",
                "density": np.asarray(raw["density"]),
                "history": raw["history"],
                "final_compliance": float(raw["final_compliance"]),
                "final_volume": float(raw["final_volume"]),
                "nodes": np.asarray(fem.nodes_np),
                "elems": np.asarray(fem.elems_t),
                "density_mean_mask": np.asarray(fem.design_elem_mask),
                "coupling_geometry": fem.coupling_geometry,
                "notes": ["Uses the website shaft-coupling cylindrical design mask with force + torque superposition."],
            }
        elif method == "teacher_numpy":
            run_data = run_teacher_numpy(cfg, progress_callback=progress_only)
            run_data["density_mean_mask"] = np.ones(len(run_data["density"]), dtype=bool)
            run_data["coupling_geometry"] = None
        elif method == "teacher_nn":
            # Teacher NN controls its own full iteration loop, so status will jump to complete at the end.
            run_data = run_teacher_nn(cfg, progress_callback=progress_only)
            run_data["density_mean_mask"] = np.ones(len(run_data["density"]), dtype=bool)
            run_data["coupling_geometry"] = None
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        density = np.asarray(run_data["density"], dtype=float)
        history = run_data["history"]
        nodes = np.asarray(run_data["nodes"])
        elems = np.asarray(run_data["elems"], dtype=int)
        compliance = np.asarray(history.get("compliance", []), dtype=float)
        volume = np.asarray(history.get("volume", []), dtype=float)
        iterations = np.asarray(history.get("iteration", list(range(len(compliance)))), dtype=int)
        density_change = np.asarray(history.get("density_change", [np.nan] * len(iterations)), dtype=float)
        if len(compliance) == 0:
            raise RuntimeError("Optimizer returned an empty compliance history")
        compliance0 = float(compliance[0])
        compliancef = float(run_data["final_compliance"])
        improvement = float((1 - compliancef / compliance0) * 100.0) if abs(compliance0) > 0 else 0.0

        conv_path = os.path.join(run_dir, "convergence.png")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].semilogy(iterations, compliance, "b-o", linewidth=2, markersize=3)
        axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Compliance"); axes[0].set_title("Compliance Convergence"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(iterations, volume, "g-o", linewidth=2, markersize=3, label="Actual")
        axes[1].axhline(optimizer_cfg["volume_fraction"], color="r", linestyle="--", linewidth=2, label="Target")
        axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Volume Fraction"); axes[1].set_title("Volume Constraint"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        finite_change = np.isfinite(density_change) & (density_change > 0)
        if np.any(finite_change):
            axes[2].semilogy(iterations[finite_change], density_change[finite_change], "m-o", linewidth=2, markersize=3)
            axes[2].set_ylabel("Max Density Change")
        else:
            axes[2].text(0.5, 0.5, "Not reported by this optimizer", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_ylabel("Density Change")
        axes[2].set_xlabel("Iteration"); axes[2].set_title("Convergence Indicator"); axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); fig.savefig(conv_path, dpi=150, bbox_inches="tight"); plt.close(fig)

        hist_path = os.path.join(run_dir, "density_histogram.png")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(density, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(density), color="red", linestyle="--", linewidth=2, label=f"Mean {np.mean(density):.3f}")
        ax.axvline(optimizer_cfg["volume_fraction"], color="green", linestyle="--", linewidth=2, label=f"Target {optimizer_cfg['volume_fraction']:.3f}")
        ax.set_xlabel("Density"); ax.set_ylabel("Count"); ax.set_title("Density Distribution"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); fig.savefig(hist_path, dpi=150, bbox_inches="tight"); plt.close(fig)

        structure_files = []
        for threshold in cfg["visualization"]["thresholds"]:
            filename = f"optimized_design_threshold_{threshold:.2f}.png"
            _plot_3d_design(nodes, elems, density, threshold, f"{run_data['engine_label']} (ρ > {threshold:.2f})", os.path.join(run_dir, filename))
            structure_files.append(filename)

        npz_name = "result_data.npz"
        np.savez(
            os.path.join(run_dir, npz_name), density=density, compliance=compliance,
            volume=volume, density_change=density_change, iteration=iterations,
            nodes=nodes, elems=elems, engine=np.array(run_data["engine"]),
        )
        all_filenames = ["convergence.png", "density_histogram.png"] + structure_files + [s["filename"] for s in snapshot_files] + [npz_name]
        mask = np.asarray(run_data["density_mean_mask"], dtype=bool)
        return {
            "task_id": task_id,
            "status": "Optimization complete",
            "engine": run_data["engine"],
            "engine_label": run_data["engine_label"],
            "run_mode": cfg.get("run_mode", "normal"),
            "notes": ([f"Run mode: {'Quick Preview' if cfg.get('run_mode') == 'quick' else 'Normal Run'}; actual mesh {mesh['nx']}×{mesh['ny']}×{mesh['nz']}, iterations {optimizer_cfg['iterations']}."] + run_data.get("notes", [])),
            "metrics": {
                "initial_compliance": compliance0,
                "final_compliance": compliancef,
                "compliance_improvement_percent": improvement,
                "final_volume": float(run_data["final_volume"]),
                "density_min": float(np.min(density)),
                "density_max": float(np.max(density)),
                "density_mean": float(np.mean(density[mask])) if np.any(mask) else float(np.mean(density)),
                "coupling_geometry": run_data.get("coupling_geometry"),
            },
            "input_summary": cfg,
            "files": [{"name": name, "url": f"/runs/{task_id}/{name}"} for name in all_filenames],
            "structure_image_urls": [f"/runs/{task_id}/{name}" for name in structure_files],
            "snapshot_urls": [{"iteration": s["iteration"], "url": f"/runs/{task_id}/{s['filename']}"} for s in snapshot_files],
            "convergence_url": f"/runs/{task_id}/convergence.png",
            "histogram_url": f"/runs/{task_id}/density_histogram.png",
            "download_url": f"/runs/{task_id}/{npz_name}",
        }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(ROOT_DIR, "dataset_strut.pkl")
VAE_PATH = os.path.join(ROOT_DIR, "vae.pickle")
STATIC_DIR = os.path.join(BASE_DIR, "web")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

service = DataModelService(DATASET_PATH, VAE_PATH)
optimization_service = OptimizationService(RUNS_DIR)
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "dataset_loaded": service.available,
        "model_loaded": service.available and len(service.model_state) > 0,
        "optimization_ready": True,
    })


def _require_dataset(fn):
    """Decorator that returns 503 when dataset is unavailable."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not service.available:
            return jsonify({"error": "Dataset / VAE not loaded (pickle files missing)"}), 503
        return fn(*args, **kwargs)
    return wrapper


@app.get("/api/dataset/stats")
@_require_dataset
def dataset_stats():
    return jsonify(service.stats())


@app.get("/api/dataset/sample")
@_require_dataset
def dataset_sample():
    split = request.args.get("split", "train")
    idx = int(request.args.get("index", "0"))
    return jsonify(service.sample(split, idx))


@app.get("/api/dataset/random")
@_require_dataset
def dataset_random():
    split = request.args.get("split", "train")
    return jsonify(service.random_sample(split))


@app.post("/api/dataset/nearest")
@_require_dataset
def dataset_nearest():
    payload = request.get_json(force=True)
    split = payload.get("split", "train")
    tensor = payload.get("tensor")
    return jsonify(service.nearest_by_tensor(tensor, split))


@app.get("/api/model/info")
@_require_dataset
def model_info():
    return jsonify(service.model_info())


@app.post("/api/optimize/start")
def optimize_start():
    payload = request.get_json(silent=True) or {}
    try:
        task = optimization_service.start(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(task), 202


@app.get("/api/optimize/status/<task_id>")
def optimize_status(task_id):
    task = optimization_service.get(task_id)
    if task is None:
        return jsonify({"error": "task not found"}), 404
    return jsonify(task)


@app.post("/api/optimize/export_stl/<task_id>")
def optimize_export_stl(task_id):
    """Turn a completed run's density field into a printable STL.

    This is the bridge between the SIMP result (a per-voxel density array)
    and any slicer (Cura, PrusaSlicer, Bambu Studio, ...): marching cubes
    reconstructs a triangle surface at the requested density threshold, then
    mesh_export.py cleans it up (merged vertices, fixed normals, filled
    holes) so the output is watertight and ready to slice, not just a
    picture like the existing structure PNGs.
    """
    if not _MESH_EXPORT_OK:
        return jsonify({
            "error": "STL export requires scikit-image and trimesh. "
                     "Install with: pip install -r requirements_shaft.txt"
        }), 503

    task = optimization_service.get(task_id)
    if task is None:
        return jsonify({"error": "task not found"}), 404
    if task.get("status") != "completed":
        return jsonify({"error": f"task is not completed (status={task.get('status')})"}), 400

    payload = request.get_json(silent=True) or {}
    try:
        threshold = float(payload.get("threshold", 0.3))
        smooth_iterations = int(payload.get("smooth_iterations", 0))
        smooth_lambda = float(payload.get("smooth_lambda", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold/smooth_iterations/smooth_lambda must be numbers"}), 400
    if not (0.0 < threshold < 1.0):
        return jsonify({"error": "threshold must be between 0 and 1"}), 400
    if smooth_iterations < 0 or smooth_iterations > 50:
        return jsonify({"error": "smooth_iterations must be between 0 and 50"}), 400

    run_dir = os.path.join(RUNS_DIR, task_id)
    npz_path = os.path.join(run_dir, "result_data.npz")
    if not os.path.isfile(npz_path):
        return jsonify({"error": "result_data.npz not found for this task"}), 404

    mesh_cfg = task["input_summary"]["mesh"]
    data = np.load(npz_path)
    nodes = data["nodes"]
    elems = data["elems"]
    density = data["density"]

    smooth_tag = f"_smooth{smooth_iterations}" if smooth_iterations > 0 else ""
    file_name = f"optimized_model_threshold_{threshold:.2f}{smooth_tag}.stl"
    output_path = os.path.join(run_dir, file_name)

    try:
        diag = mesh_export.export_stl(
            nodes, elems, density, mesh_cfg, output_path,
            threshold=threshold, smooth_iterations=smooth_iterations, smooth_lambda=smooth_lambda,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"STL export failed: {exc}"}), 500

    diag.pop("output_path", None)
    diag["file_name"] = file_name
    diag["download_url"] = f"/runs/{task_id}/{file_name}"
    return jsonify(diag)


@app.get("/runs/<task_id>/<path:filename>")
def runs_file(task_id, filename):
    directory = os.path.join(RUNS_DIR, task_id)
    return send_from_directory(directory, filename, as_attachment=False)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "18080"))
    print(f"Starting 3D Topology Optimization server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
