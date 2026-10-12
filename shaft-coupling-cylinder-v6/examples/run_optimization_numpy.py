#!/usr/bin/env python3
"""
Run SIMP topology optimization with NumPy FEM solver (no PyTorch required).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import sys

# Use pure NumPy implementations (no PyTorch or SciPy required)
from fem3d_numpy import HexFEMSolver3D
from simp_numpy import SIMPOptimizer


def plot_3d_design(nodes, elems, density, threshold=0.5, title="3D Structure"):
    """Create 3D visualization of optimized structure."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Find elements above threshold
    mask = density > threshold
    active_elems = np.where(mask)[0]

    print(f"   {len(active_elems)} elements with ρ > {threshold}")

    if len(active_elems) == 0:
        ax.text(
            0.5,
            0.5,
            0.5,
            f"No elements with ρ > {threshold}",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
        )
        ax.set_title(title)
        return fig, ax

    # Color map for density
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=threshold, vmax=1.0)

    # Plot each active face as solid with color based on density
    for elem_idx in active_elems:
        elem_nodes = elems[elem_idx]
        elem_coords = nodes[elem_nodes]
        elem_density = density[elem_idx]
        color = cmap(norm(elem_density))

        # Plot the 6 faces of the hexahedral element
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5],  # Right
        ]
        for face in faces:
            ax.add_collection3d(
                Poly3DCollection([elem_coords[face]], facecolors=color, edgecolors="k", linewidths=0.5)
            )

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Set limits
    all_active_coords = nodes[elems[active_elems]].reshape(-1, 3)
    ax.set_xlim([all_active_coords[:, 0].min(), all_active_coords[:, 0].max()])
    ax.set_ylim([all_active_coords[:, 1].min(), all_active_coords[:, 1].max()])
    ax.set_zlim([all_active_coords[:, 2].min(), all_active_coords[:, 2].max()])
    ## set the aspect ratio realize the true geometric ratios of the 3D structure
    ax.set_box_aspect([
        np.ptp(all_active_coords[:, 0]),
        np.ptp(all_active_coords[:, 1]),
        np.ptp(all_active_coords[:, 2])
    ])

    # Add colorbar to show density scale
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Material Density", fontsize=10)

    plt.tight_layout()
    return fig, ax


def main():
    print("\n" + "=" * 80)
    print("3D TOPOLOGY OPTIMIZATION WITH SIMP")
    print("=" * 80)

    # Create FEM solver
    print("\n1. Creating FEM solver (NumPy-based)...")
    fem = HexFEMSolver3D(E_mod=200e9, nu=0.3)
    fem.set_mesh(Lx=1.0, Ly=1, Lz=1, nx=10, ny=10, nz=10)
    print(f"   ✓ Mesh: {fem.nodes_np.shape[0]} nodes, {fem.n_elems} elements")

    # Apply boundary conditions
    print("\n2. Applying boundary conditions...")
    fem.fix_face(axis=0, coord=0.0)
    # Apply point load at center of right face (cantilever tip)
    #fem.add_point_load(location=(1.0, 0.1, 0.05), direction=1, magnitude=1e4)
    # Apply distributed load on right face (more realistic for 3D cantilever)
    #fem.add_distributed_load(axis=0, coord=1.0, direction=-1, total=1e4 / (fem.ny * fem.nz))  # Total load divided by number of nodes on the face
    fem.add_torque_load(center=(1.0, 0.5, 0.5), axis=0, magnitude=1000.0) 
    print("   ✓ Fixed left face, applied point load at right face center")

    # Create optimizer
    print("\n3. Setting up SIMP optimizer...")
    volume_fraction = 0.2  # Target volume fraction (20% of the design space)
    optimizer = SIMPOptimizer(
        fem_solver=fem,
        initial_density=0.2,
        volume_fraction=volume_fraction,
        penalty=3.0,
        filter_radius=0.02,
    )

    # Run optimization
    print("\n4. Running optimization (100 iterations for full convergence)...")
    print("   This may take a few minutes...")
    result = optimizer.optimize(n_iterations=100, verbose=True)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    # Print results
    compliance_improvement = (
        1 - result["final_compliance"] / result["history"]["compliance"][0]
    ) * 100
    print(f"\nCompliance improvement: {compliance_improvement:.1f}%")
    print(f"Initial compliance: {result['history']['compliance'][0]:.6e}")
    print(f"Final compliance:   {result['final_compliance']:.6e}")
    print(f"Final volume: {result['final_volume']:.3f}")

    # Analyze densities
    print("\n" + "=" * 80)
    print("DENSITY ANALYSIS")
    print("=" * 80)
    density = result["density"]
    print(f"Min density: {np.min(density):.4f}")
    print(f"Max density: {np.max(density):.4f}")
    print(f"Mean density: {np.mean(density):.4f}")
    print(f"Std dev: {np.std(density):.4f}")
    print(f"\nElements by density range:")
    print(f"  ρ > 0.9:  {np.sum(density > 0.9):3d}")
    print(f"  ρ > 0.7:  {np.sum(density > 0.7):3d}")
    print(f"  ρ > 0.5:  {np.sum(density > 0.5):3d}")
    print(f"  ρ > 0.3:  {np.sum(density > 0.3):3d}")
    print(f"  ρ > 0.2:  {np.sum(density > 0.2):3d}")
    print(f"  ρ > 0.1:  {np.sum(density > 0.1):3d}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Convergence plot
    print("\n1. Convergence plot...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    history = result["history"]
    iterations = history["iteration"]
    compliance = history["compliance"]
    volume = history["volume"]
    density_change = history["density_change"]

    axes[0].semilogy(iterations, compliance, "b-o", linewidth=2, markersize=4)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Compliance (log scale)")
    axes[0].set_title("Compliance Convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, volume, "g-o", linewidth=2, markersize=4, label="Actual")
    axes[1].axhline(volume_fraction, color="r", linestyle="--", linewidth=2, label=f"Target ({volume_fraction:.3f})")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Volume Fraction")
    axes[1].set_title("Volume Constraint")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(iterations, density_change, "r-o", linewidth=2, markersize=4)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Max Density Change (log scale)")
    axes[2].set_title("Convergence Indicator")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_final.png", dpi=150, bbox_inches="tight")
    print("   ✓ Saved to convergence_final.png")
    plt.close(fig)

    # 2. Density histogram
    print("\n2. Density histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(density, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(density), color="red", linestyle="--", linewidth=2,
              label=f"Mean: {np.mean(density):.3f}")
    ax.axvline(volume_fraction, color="green", linestyle="--", linewidth=2, label=f"Target: {volume_fraction:.3f}")

    ax.set_xlabel("Density (ρ)")
    ax.set_ylabel("Number of Elements")
    ax.set_title("Density Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("density_histogram_final.png", dpi=150, bbox_inches="tight")
    print("   ✓ Saved to density_histogram_final.png")
    plt.close(fig)

    # 3. 3D design visualizations
    print("\n3. 3D structure visualizations...")
    for threshold in [0.1, 0.3, 0.5]:
        if np.sum(density > threshold) > 0:
            print(f"\n   Creating visualization for ρ > {threshold}...")
            fig, ax = plot_3d_design(
                fem.nodes_np,
                fem.elems_t,
                density,
                threshold=threshold,
                title=f"SIMP Optimized Design (ρ > {threshold})",
            )
            filename = f"optimized_design_threshold_{threshold:.1f}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"   ✓ Saved to {filename}")
            plt.close(fig)
        else:
            print(f"   No elements with ρ > {threshold}, skipping")

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - convergence_final.png")
    print("  - density_histogram_final.png")
    print("  - optimized_design_threshold_0.1.png")
    print("  - optimized_design_threshold_0.3.png")
    print("  - optimized_design_threshold_0.5.png")

    return optimizer, result, fem


if __name__ == "__main__":
    optimizer, result, fem = main()
    print("\n✓ Optimization complete! Check the PNG files for visualizations.")
