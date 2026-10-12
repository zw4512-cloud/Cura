"""
Nural Network based Topology Optimization .
"""

import numpy as np
import torch


class NNOptimizer:
    """Neural Network based topology optimizer."""

    def __init__(
        self, fem_solver, structure_model, volume_fraction=0.3, penalty=3.0
    ):
        """Initialize optimizer.

        NOTE: Boundary conditions should be applied to fem_solver BEFORE creating the optimizer.
        The optimizer will save and reapply them each iteration.
        """
        self.fem_solver = fem_solver
        self.structure_model = structure_model

        self.n_elem = fem_solver.n_elems
        self.volume_fraction = volume_fraction
        self.penalty = penalty
        # Save boundary conditions from FEM solver
        # (they should have been applied before creating the optimizer)
        self.fixed_dofs_saved = fem_solver.fixed_dofs
        self.F_global_saved = fem_solver.F_global

        # Storage for history
        self.history = {
            "compliance": [],
            "volume": [],
            "density_change": [],
            "iteration": [],
        }

    def optimize(self, n_iterations=50, verbose=True):
        """Run topology optimization.

        Uses boundary conditions that were applied before the optimizer was created.
        """
        #self.optimizer = torch.optim.Adam([self.structure_model.densities], lr=0.01)
        self.optimizer = torch.optim.Adam(self.structure_model.parameters(), lr=0.01)
        alphaMax = 100*self.volume_fraction 
        alphaIncrement= 0.08
        alpha = alphaIncrement  # start
        for iteration in range(n_iterations):
            self.optimizer.zero_grad()
            self.densities = self.structure_model()
            ### compute compliance and sensitivities with current densities
            results = self.fem_solver.solve(self.densities)

            # Compute compliance 
            compliance = results["compliance"]
            if(iteration == 0): 
                self.obj0 = compliance.detach()
            objective = compliance/self.obj0
            volConstraint = ((torch.mean(self.densities)/self.volume_fraction) - 1.0) 
            loss = objective+ alpha*(pow(volConstraint,2))
            alpha = min(alphaMax, alpha + alphaIncrement) 
            loss.backward()
            self.optimizer.step()

            # Store history
            volume = torch.mean(self.densities).item()
            self.history["compliance"].append(compliance.item())
            self.history["volume"].append(volume)
            self.history["iteration"].append(iteration)

            if verbose:
                print(
                    f"  Iter {iteration:2d} | C: {compliance:.6e} | V: {volume:.4f}"
                )

        return {
            "density": self.densities.detach().cpu().numpy(),
            "final_compliance": self.history["compliance"][-1],
            "final_volume": self.history["volume"][-1],
            "history": self.history,
        }
