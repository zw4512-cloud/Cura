"""
mesh_export.py

Bridges the SIMP topology-optimization density field (defined on a
structured Hex8 voxel mesh, see fem3d_numpy.py / simp_numpy.py) into a
printable surface mesh (STL), so a run's result can be opened directly in a
slicer (Cura, PrusaSlicer, Bambu Studio, ...) instead of only being viewed
as the density PNG plots app.py already produces.

Pipeline
--------
density (n_elem,) on the structured hex grid
  -> resample onto a regular 3D array (nx, ny, nz), by element-center
     coordinates (so it does not depend on how fem3d_numpy.py orders elems)
  -> pad with an empty (density = 0) 1-voxel border, so marching cubes has
     something to interpolate against at the domain boundary and the
     resulting surface is closed there too
  -> marching-cubes iso-surface at `threshold`
  -> trimesh cleanup (merge vertices, drop degenerate/duplicate faces, fix
     normals/winding, fill any remaining holes, optional Laplacian smoothing)
  -> STL export

This module has no dependency on Flask/torch/the FEM solver itself -- it
only needs nodes, elems, density and the box-mesh config (Lx,Ly,Lz,nx,ny,nz),
so it can be called from app.py, from a standalone script, or later from a
Cura-plugin subprocess bridge without dragging those dependencies along.

Requires: numpy, scikit-image, trimesh
    pip install scikit-image trimesh
"""

from __future__ import annotations

import numpy as np
from skimage import measure
import trimesh


def _density_to_grid(nodes, elems, density, Lx, Ly, Lz, nx, ny, nz):
    """Resample the (n_elem,) density array back onto the regular
    (nx, ny, nz) grid it actually came from, using element-center
    coordinates.

    Going by coordinates (rather than assuming a fixed loop order inside
    fem3d_numpy.py's mesh generator) makes this robust to however that
    solver actually orders `elems` -- it will still land in the right voxel
    even if that ordering ever changes.
    """
    elems = np.asarray(elems, dtype=int)
    nodes = np.asarray(nodes, dtype=float)
    density = np.asarray(density, dtype=float)

    if elems.ndim != 2 or elems.shape[1] != 8:
        raise ValueError(f"expected Hex8 elems with shape (n_elem, 8), got {elems.shape}")
    if len(density) != len(elems):
        raise ValueError(f"density has {len(density)} entries but elems has {len(elems)}")

    centers = nodes[elems].mean(axis=1)  # (n_elem, 3)
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    ix = np.clip(np.floor(centers[:, 0] / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(centers[:, 1] / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.floor(centers[:, 2] / dz).astype(int), 0, nz - 1)

    grid = np.zeros((nx, ny, nz), dtype=float)
    counts = np.zeros((nx, ny, nz), dtype=int)
    np.add.at(grid, (ix, iy, iz), density)
    np.add.at(counts, (ix, iy, iz), 1)

    filled = counts > 0
    if not np.all(filled):
        missing = int(np.sum(~filled))
        raise ValueError(
            f"{missing} of {nx * ny * nz} grid cells received no element -- "
            "nodes/elems do not match the given nx,ny,nz box-mesh config."
        )
    if np.any(counts > 1):
        grid[filled] /= counts[filled]

    return grid, (dx, dy, dz)


def density_to_surface(
    nodes,
    elems,
    density,
    mesh_cfg,
    threshold=0.3,
    smooth_iterations=0,
    smooth_lambda=0.5,
):
    """Turn a SIMP density field into a closed triangle surface.

    Parameters
    ----------
    nodes, elems, density : arrays as saved by app.py's OptimizationService
        into result_data.npz (`nodes`, `elems`, `density`).
    mesh_cfg : dict with Lx, Ly, Lz, nx, ny, nz -- the same box-mesh config
        used to run the optimization (cfg["mesh"] in app.py).
    threshold : density iso-level. Matches the "structure" PNGs app.py
        already produces (defaults there: 0.1 / 0.3 / 0.5).
    smooth_iterations : 0 keeps the exact voxel (staircase) surface, which
        is the most faithful to what SIMP actually computed. A small value
        (5-10) with a modest smooth_lambda visually softens the stair-steps
        for printing, at the cost of slightly blurring thin load-bearing
        members -- do not use a large value on structurally thin parts.
    """
    Lx, Ly, Lz = mesh_cfg["Lx"], mesh_cfg["Ly"], mesh_cfg["Lz"]
    nx, ny, nz = int(mesh_cfg["nx"]), int(mesh_cfg["ny"]), int(mesh_cfg["nz"])

    grid, (dx, dy, dz) = _density_to_grid(nodes, elems, density, Lx, Ly, Lz, nx, ny, nz)

    # Pad with a 1-voxel border of density=0 on every side. Without this,
    # marching_cubes has no "outside" to interpolate against at the domain
    # boundary, and the resulting mesh is open (not watertight) wherever the
    # part touches the edge of the box -- which it usually does here, since
    # the passive-solid coupling interfaces sit right at the ends of the box.
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=0.0)

    if padded.max() <= threshold:
        raise ValueError(
            f"No material above threshold={threshold} (max density in grid "
            f"= {padded.max():.4f}). Lower the threshold or check the run."
        )

    verts, faces, _normals, _values = measure.marching_cubes(
        padded, level=threshold, spacing=(dx, dy, dz)
    )
    # Undo the padding offset (marching_cubes coordinates are in
    # padded-grid space, i.e. shifted by one voxel on each axis).
    verts = verts - np.array([dx, dy, dz])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # process=True already merges duplicate vertices and drops degenerate
    # faces on construction; nondegenerate_faces() + update_faces is kept
    # explicit here (idempotent) since that automatic pass is easy to lose
    # across trimesh versions.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(
            mesh, lamb=smooth_lambda, iterations=int(smooth_iterations)
        )

    return mesh


def export_stl(
    nodes,
    elems,
    density,
    mesh_cfg,
    output_path,
    threshold=0.3,
    smooth_iterations=0,
    smooth_lambda=0.5,
    unit_scale=1000.0,
):
    """Build the surface and write it straight to an STL file.

    unit_scale: the FEM/SIMP side of this project works in SI meters (see
    app.py's default mesh Lx/Ly/Lz and material.E_mod in Pa = N/m^2). STL
    files carry no unit metadata, and every slicer (Cura, PrusaSlicer, Bambu
    Studio) treats the raw numbers in an STL as millimeters. Writing the
    meter-valued coordinates straight to the file would make a 1 m shaft
    load as a 1 mm speck. unit_scale=1000.0 (the default) converts meters to
    millimeters at export time, so the file opens at the correct real-world
    size; only pass a different value if you are feeding this a mesh_cfg
    that is not already in meters.

    Returns a small diagnostics dict -- surface this back to the caller (the
    website, or the Cura plugin) so a non-watertight or oddly-sized result
    does not silently get sent to a slicer/printer. The diagnostics
    (volume_m3, bounds_m) describe the true physical part in meters,
    independent of unit_scale; only the exported file itself is rescaled.
    """
    mesh = density_to_surface(
        nodes,
        elems,
        density,
        mesh_cfg,
        threshold=threshold,
        smooth_iterations=smooth_iterations,
        smooth_lambda=smooth_lambda,
    )
    is_watertight = bool(mesh.is_watertight)
    volume_m3 = float(mesh.volume) if is_watertight else None
    bounds_m = mesh.bounds.tolist()
    n_vertices = int(len(mesh.vertices))
    n_faces = int(len(mesh.faces))

    export_mesh = mesh.copy()
    if unit_scale != 1.0:
        export_mesh.apply_scale(unit_scale)
    export_mesh.export(output_path)

    return {
        "output_path": output_path,
        "threshold": threshold,
        "smooth_iterations": smooth_iterations,
        "unit_scale": unit_scale,
        "file_units": "mm" if unit_scale == 1000.0 else f"input units x{unit_scale}",
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "is_watertight": bool(mesh.is_watertight),
        "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
        "bounds_m": mesh.bounds.tolist(),
    }
