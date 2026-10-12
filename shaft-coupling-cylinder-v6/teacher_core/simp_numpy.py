"""
Pure NumPy SIMP Topology Optimization (no PyTorch required).
Simple implementation for density-based topology optimization.
"""

import numpy as np
from itertools import product as iproduct


class DensityFilter:
    """Density filter to prevent checkerboard patterns."""

    def __init__(self, nodes, elems, radius=1.5):
        """Initialize filter with connectivity information."""
        self.nodes = nodes
        self.elems = elems
        self.radius = radius
        self.n_elem = len(elems)

        # Precompute element centers
        self.elem_centers = np.mean(nodes[elems], axis=1)

        # Precompute weight matrix
        self._compute_weights()

    def _compute_weights(self):
        """Precompute filter weights."""
        self.H = np.zeros((self.n_elem, self.n_elem))
        self.H_sum = np.zeros(self.n_elem)

        for i in range(self.n_elem):
            for j in range(self.n_elem):
                dist = np.linalg.norm(self.elem_centers[i] - self.elem_centers[j])
                if dist < self.radius:
                    weight = self.radius - dist
                    self.H[i, j] = weight
                    self.H_sum[i] += weight

    def apply(self, density):
        """Apply density filter."""
        filtered = np.zeros_like(density)
        for i in range(self.n_elem):
            if self.H_sum[i] > 0:
                filtered[i] = np.sum(self.H[i, :] * density) / self.H_sum[i]
            else:
                filtered[i] = density[i]
        return filtered


class SIMPOptimizer:
    """Pure NumPy SIMP topology optimizer."""

    def __init__(
        self, fem_solver, initial_density=0.5, volume_fraction=0.3, penalty=3.0, filter_radius=1.5
    ):
        """Initialize optimizer.

        NOTE: Boundary conditions should be applied to fem_solver BEFORE creating the optimizer.
        The optimizer will save and reapply them each iteration.
        """
        self.fem_solver = fem_solver
        self.n_elem = fem_solver.n_elems
        self.density = np.ones(self.n_elem) * initial_density
        self.volume_fraction = volume_fraction
        self.penalty = penalty

        # Create filter
        self.filter = DensityFilter(fem_solver.nodes_np, fem_solver.elems_t, filter_radius)

        # Save boundary conditions from FEM solver
        # (they should have been applied before creating the optimizer)
        self.fixed_dofs_saved = fem_solver.fixed_dofs.copy()
        self.F_global_saved = fem_solver.F_global.copy()

        # Storage for history
        self.history = {
            "compliance": [],
            "volume": [],
            "density_change": [],
            "iteration": [],
        }


    def update_density(self, sensitivities):
        """
        Update densities using Optimality Criteria (OC) method with bisection.

        Based on standard SIMP topology optimization method:
        - Uses bisection to find optimal Lagrange multiplier
        - Applies move limits for stability
        - Enforces volume constraint
        - Applies sensitivity filtering for checkerboard prevention
        """
        # Volume constraint
        V_max = self.volume_fraction * self.n_elem

        # OC method parameters
        move_limit = 0.2  # Maximum change per iteration (standard: 0.2)
        tol = 1e-3  # Convergence tolerance for bisection
        max_iter = 100  # Max bisection iterations

        # Bisection for Lagrange multiplier
        lam_low = 0.0
        lam_high = 1e9

        for lam_iter in range(max_iter):
            # Geometric mean for better convergence
            if lam_low == 0:
                lam_mid = lam_high / 2.0  # Arithmetic mean when lam_low = 0
            else:
                lam_mid = np.sqrt(lam_low * lam_high)

            # Optimality Criteria update (vectorized)
            # xnew = max(0, max(x-move, min(1, min(x+move, x*sqrt(-dc/dv/lam)))))
            # where dv = 1 for volume constraint

            # Compute the OC factor: x * sqrt(-dc / dv / lam)
            # For volume constraint, dv = 1
            # Handle negative sensitivities (standard for compliance minimization)
            safe_sens = np.where(sensitivities < 0, sensitivities, -1e-10)
            oc_factor = self.density * np.sqrt(-safe_sens / lam_mid)

            # Apply move limits: x ± move_limit
            density_new = np.maximum(
                0.0,  # Lower bound: 0
                np.maximum(
                    self.density - move_limit,  # Lower move limit
                    np.minimum(
                        1.0,  # Upper bound: 1
                        np.minimum(
                            self.density + move_limit,  # Upper move limit
                            oc_factor  # OC factor
                        )
                    )
                )
            )

            # Apply density filter to prevent checkerboard patterns
            density_new = self.filter.apply(density_new)

            # Check volume constraint
            current_vol = np.sum(density_new)
            volume_error = current_vol - V_max

            # Convergence check
            if abs(volume_error) < tol * V_max:
                break

            # Bisection update
            if volume_error > 0:
                # Too much material, increase lambda (decrease density)
                lam_low = lam_mid
            else:
                # Too little material, decrease lambda (increase density)
                lam_high = lam_mid

        return density_new

    def optimize(self, n_iterations=50, verbose=True):
        """Run topology optimization.

        Uses boundary conditions that were applied before the optimizer was created.
        """

        for iteration in range(n_iterations):
            ### compute compliance and sensitivities with current densities
            results = self.fem_solver.solve(self.density)

            # Compute compliance and sensitivities
            compliance = results["compliance"]
            sensitivities = results["sensitivities"]

            # Update densities
            density_new = self.update_density(sensitivities)
            density_change = np.max(np.abs(density_new - self.density))
            self.density = density_new

            # Store history
            volume = np.sum(self.density) / self.n_elem
            self.history["compliance"].append(compliance)
            self.history["volume"].append(volume)
            self.history["density_change"].append(density_change)
            self.history["iteration"].append(iteration)

            if verbose:
                print(
                    f"  Iter {iteration:2d} | C: {compliance:.6e} | V: {volume:.4f} | ΔρMax: {density_change:.6e}"
                )

        return {
            "density": self.density,
            "final_compliance": self.history["compliance"][-1],
            "final_volume": self.history["volume"][-1],
            "history": self.history,
        }
