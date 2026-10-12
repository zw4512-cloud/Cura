"""
3D Finite Element Method — Hex8 (PyTorch, Differentiable)
==========================================================
PyTorch implementation with full autograd support for neural network-based
topology optimization. All operations are differentiable.
"""

import torch
import torch.nn as nn
from itertools import product as iproduct
import numpy as np


class HexFEMSolver3D_Torch(nn.Module):
    """3D FEM Solver using Hex8 elements with PyTorch (fully differentiable)."""

    def __init__(self, E_mod=200e9, nu=0.3, Emin=1e-3, penalty=3.0, device='cpu'):
        """
        Initialize FEM solver with material properties.

        Parameters
        ----------
        E_mod : float
            Young's modulus (base material)
        nu : float
            Poisson's ratio
        Emin : float
            Minimum Young's modulus (for SIMP, prevents singularity)
        penalty : float
            SIMP penalization exponent (default 3.0)
        device : str
            'cpu' or 'cuda'
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.E_mod = torch.tensor(E_mod, dtype=torch.float32, device=self.device)
        self.nu = torch.tensor(nu, dtype=torch.float32, device=self.device)
        self.Emin = torch.tensor(Emin, dtype=torch.float32, device=self.device)
        self.penalty = penalty

        self.nodes_torch = None
        self.elems_t = None  # Keep as numpy for indexing
        self.n_dofs = None
        self.K_global = None
        self.F_global = None
        self.fixed_dofs = set()
        self.n_elems = 0

        # Pre-computed for vectorized assembly
        self.KE = None  # Reference element stiffness (24, 24)
        self.iK = None  # Row indices for assembly
        self.jK = None  # Column indices for assembly
        self.edofMat = None  # Element DOF matrix (n_elems, 24)

    def set_mesh(self, Lx=1.0, Ly=0.1, Lz=0.1, nx=10, ny=3, nz=3):
        """Create structured mesh of Hex8 elements."""
        # Create nodes
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        x = torch.linspace(0, Lx, nx + 1, device=self.device, dtype=torch.float32)
        y = torch.linspace(0, Ly, ny + 1, device=self.device, dtype=torch.float32)
        z = torch.linspace(0, Lz, nz + 1, device=self.device, dtype=torch.float32)

        nodes = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        self.nodes_torch = nodes.reshape(-1, 3)

        # Create elements
        elems = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                    n1 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                    n2 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                    n3 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                    n4 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                    n5 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                    n6 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
                    n7 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
                    elems.append([n0, n1, n2, n3, n4, n5, n6, n7])

        self.elems_t = elems  # Keep as list for NumPy-style indexing
        self.n_elems = len(elems)
        self.n_dofs = len(self.nodes_torch) * 3
        self.density = torch.ones(self.n_elems, device=self.device, dtype=torch.float32)  # Default: uniform density
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # Initialize force vector
        self.F_global = torch.zeros(self.n_dofs, device=self.device, dtype=torch.float32)
        ## do not compute dirrevative of F_global since it is not a function of density
        self.F_global.requires_grad = False

        # Pre-compute reference element stiffness and assembly indices
        if self.KE is None:
            self._precompute_assembly_data()

    def _constitutive_matrix(self):
        """Compute 6×6 constitutive matrix for 3D elasticity."""
        lam = self.E_mod * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E_mod / (2 * (1 + self.nu))

        D = torch.zeros(6, 6, device=self.device, dtype=torch.float32)
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
        D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def _hex8_stiffness(self, coords, D):
        """Compute element stiffness matrix for Hex8 element (differentiable)."""
        # 2×2×2 Gauss quadrature
        gp = torch.tensor([-1, 1], device=self.device, dtype=torch.float32) / torch.sqrt(torch.tensor(3.0))
        Ke = torch.zeros(24, 24, device=self.device, dtype=torch.float32)

        for gxi, geta, gzeta in iproduct(gp, repeat=3):
            gxi_val = gxi.item() if hasattr(gxi, 'item') else gxi
            geta_val = geta.item() if hasattr(geta, 'item') else geta
            gzeta_val = gzeta.item() if hasattr(gzeta, 'item') else gzeta

            # Shape function derivatives
            N_xi = torch.tensor([
                -(1 - geta_val) * (1 - gzeta_val),
                (1 - geta_val) * (1 - gzeta_val),
                (1 + geta_val) * (1 - gzeta_val),
                -(1 + geta_val) * (1 - gzeta_val),
                -(1 - geta_val) * (1 + gzeta_val),
                (1 - geta_val) * (1 + gzeta_val),
                (1 + geta_val) * (1 + gzeta_val),
                -(1 + geta_val) * (1 + gzeta_val),
            ], device=self.device, dtype=torch.float32) * 0.125

            N_eta = torch.tensor([
                -(1 - gxi_val) * (1 - gzeta_val),
                -(1 + gxi_val) * (1 - gzeta_val),
                (1 + gxi_val) * (1 - gzeta_val),
                (1 - gxi_val) * (1 - gzeta_val),
                -(1 - gxi_val) * (1 + gzeta_val),
                -(1 + gxi_val) * (1 + gzeta_val),
                (1 + gxi_val) * (1 + gzeta_val),
                (1 - gxi_val) * (1 + gzeta_val),
            ], device=self.device, dtype=torch.float32) * 0.125

            N_zeta = torch.tensor([
                -(1 - gxi_val) * (1 - geta_val),
                -(1 + gxi_val) * (1 - geta_val),
                -(1 + gxi_val) * (1 + geta_val),
                -(1 - gxi_val) * (1 + geta_val),
                (1 - gxi_val) * (1 - geta_val),
                (1 + gxi_val) * (1 - geta_val),
                (1 + gxi_val) * (1 + geta_val),
                (1 - gxi_val) * (1 + geta_val),
            ], device=self.device, dtype=torch.float32) * 0.125

            # Jacobian matrix
            J = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
            J[0, :] = N_xi @ coords
            J[1, :] = N_eta @ coords
            J[2, :] = N_zeta @ coords

            detJ = torch.det(J)
            dN = torch.linalg.solve(J.T, torch.stack([N_xi, N_eta, N_zeta]))

            # B matrix (strain-displacement, 6×24)
            B = torch.zeros(6, 24, device=self.device, dtype=torch.float32)
            for i in range(8):
                B[0, 3*i] = dN[0, i]
                B[1, 3*i+1] = dN[1, i]
                B[2, 3*i+2] = dN[2, i]
                B[3, 3*i+1] = dN[2, i]
                B[3, 3*i+2] = dN[1, i]
                B[4, 3*i] = dN[2, i]
                B[4, 3*i+2] = dN[0, i]
                B[5, 3*i] = dN[1, i]
                B[5, 3*i+1] = dN[0, i]

            # Element stiffness
            Ke += B.T @ D @ B * detJ

        return Ke

    def _precompute_assembly_data(self):
        """Pre-compute reference element stiffness and assembly indices."""
        D = self._constitutive_matrix()

        # Reference element stiffness
        first_elem_nodes = self.elems_t[0]
        first_elem_coords = self.nodes_torch[first_elem_nodes]
        self.KE = self._hex8_stiffness(first_elem_coords, D)

        # Pre-compute assembly indices
        iK_list = []
        jK_list = []

        for elem_id, elem_nodes in enumerate(self.elems_t):
            edof = torch.zeros(24, dtype=torch.int64, device=self.device)
            for i, node in enumerate(elem_nodes):
                edof[3*i:3*i+3] = torch.tensor([3*node, 3*node+1, 3*node+2], 
                                               dtype=torch.int64, device=self.device)

            for i in range(24):
                for j in range(24):
                    iK_list.append(edof[i].item())
                    jK_list.append(edof[j].item())

        self.iK = torch.tensor(iK_list, dtype=torch.int64, device=self.device)
        self.jK = torch.tensor(jK_list, dtype=torch.int64, device=self.device)
        
        # Element DOF matrix
        edofMat = torch.zeros(self.n_elems, 24, dtype=torch.int64, device=self.device)
        for elem_id, elem_nodes in enumerate(self.elems_t):
            for i, node in enumerate(elem_nodes):
                edofMat[elem_id, 3*i:3*i+3] = torch.tensor([3*node, 3*node+1, 3*node+2], 
                                                             dtype=torch.int64, device=self.device)
        self.edofMat = edofMat

    def _assemble_K(self, density):
        """Assemble global stiffness matrix (differentiable with sparse COO)."""
        # SIMP scaling: E(ρ) = Emin + ρ^p * (E0 - Emin)
        self.density = density  # Save for compliance calculation
        E_eff = self.Emin + density**self.penalty * (self.E_mod - self.Emin)
        scale_factors = E_eff / self.E_mod

        # Scale element stiffness matrices CORRECTLY
        # KE is (24, 24), scale_factors is (n_elems,)
        # For each element, multiply its KE by its scale factor
        KE_scaled = self.KE.unsqueeze(0) * scale_factors.view(-1, 1, 1)  # (n_elems, 24, 24)
        sK = torch.flatten(KE_scaled)  # Flatten all scaled element matrices

        # Assemble using sparse COO tensor (differentiable)
        indices = torch.stack([self.iK, self.jK], dim=0)

        # Create sparse COO tensor and convert to dense
        K_sparse = torch.sparse_coo_tensor(indices, sK, (self.n_dofs, self.n_dofs), device=self.device)
        self.K_global = K_sparse.to_dense()

    def fix_face(self, axis=0, coord=0.0, tol=1e-6):
        """Fix all nodes on a face."""
        mask = torch.abs(self.nodes_torch[:, axis] - coord) < tol
        for node_id in torch.where(mask)[0]:
            for dof_offset in range(3):
                self.fixed_dofs.add((node_id * 3 + dof_offset).item())
        self.free_dofs = np.setdiff1d(np.arange(self.n_dofs),list(self.fixed_dofs))

    def add_point_load(self, location, direction=1, magnitude=1e4):
        """Apply point load at closest node to location."""
        location_torch = torch.tensor(location, device=self.device, dtype=torch.float32)
        distances = torch.norm(self.nodes_torch - location_torch, dim=1)
        closest_node = torch.argmin(distances).item()
        dof = closest_node * 3 + direction
        self.F_global[dof] = magnitude

    def add_distributed_load(self, axis=0, coord=1.0, direction=0, total=1e4, tol=1e-6):
        """Apply distributed load uniformly across a face.

        Distributes the total load evenly among all nodes on the specified face.
        This gives more accurate results for simple tension/compression tests
        compared to point loads.

        Parameters
        ----------
        axis : int
            Axis perpendicular to the face (0=X, 1=Y, 2=Z)
        coord : float
            Coordinate value of the face along the specified axis
        direction : int
            Load direction (0=X, 1=Y, 2=Z)
        total : float
            Total load magnitude to distribute
        tol : float
            Tolerance for finding nodes on the face
        """
        # Find all nodes on the face
        mask = torch.abs(self.nodes_torch[:, axis] - coord) < tol
        n_nodes = torch.sum(mask).item()

        if n_nodes > 0:
            # Distribute load evenly among nodes on the face
            load_per_node = total / n_nodes
            for node_id in torch.where(mask)[0]:
                dof = node_id * 3 + direction
                # Add to existing load (don't overwrite)
                self.F_global[dof] += load_per_node
        else:
            print(f"Warning: No nodes found on face at {axis}={coord}")

    def add_torque_load(self, center, axis, magnitude, radius=None, tol=1e-6):
        """
        Add torque (moment) load to the model.

        Applies torque as distributed equivalent nodal forces on nodes within
        a specified radius from the center point, perpendicular to the torque axis.

        For a displacement-based FEM without rotational DOFs, torque is converted
        to equivalent nodal forces that create the moment.

        Parameters
        ----------
        center : tuple or list
            (x, y, z) center point for torque application
        axis : int
            Torque axis: 0=x, 1=y, 2=z (right-hand rule direction)
        magnitude : float
            Torque magnitude (moment)
        radius : float, optional
            If specified, only nodes within this radius from center will receive forces.
            If None, uses the maximum distance from center to any node.
        tol : float
            Tolerance for floating point comparisons

        Returns
        -------
        None (modifies F_global in place)

        Notes
        -----
        The torque τ = r × F is converted to nodal forces such that:
        - Forces are perpendicular to both the torque axis and radius vector
        - The magnitude is distributed to create the specified moment
        - Forces on opposite sides of the center create opposing moments that sum

        For example, to apply torque about z-axis:
        - axis = 2
        - Nodes with +x displacements apply +y forces
        - Nodes with -x displacements apply -y forces
        """
        center_tensor = torch.tensor(center, dtype=torch.float32, device=self.device)
        axis_map = {0: 'x', 1: 'y', 2: 'z'}
        axis_name = axis_map.get(axis, 'unknown')

        # Find nodes to apply torque to
        distances = torch.norm(self.nodes_torch - center_tensor, dim=1)

        # Determine radius if not specified
        if radius is None:
            radius = torch.max(distances).item()

        # Get nodes within radius
        mask = distances <= (radius + tol)
        node_ids = torch.where(mask)[0]

        if len(node_ids) == 0:
            print(f"   Warning: No nodes found within radius {radius} of center {center}")
            return

        # Calculate position vectors relative to center
        pos_rel = self.nodes_torch[node_ids] - center_tensor

        # Apply torque as equivalent nodal forces
        # τ = r × F, so F is perpendicular to r and axis

        if axis == 0:  # Torque about x-axis (use y,z for forces)
            r_perp = torch.sqrt(pos_rel[:, 1]**2 + pos_rel[:, 2]**2)
            r_perp = torch.clamp(r_perp, min=tol)  # Avoid division by zero

            force_y = -magnitude / r_perp * pos_rel[:, 2]
            force_z = magnitude / r_perp * pos_rel[:, 1]

            for i, node_id in enumerate(node_ids):
                node_id_int = int(node_id.item())  # Convert tensor to Python int
                self.F_global[node_id_int * 3 + 1] = self.F_global[node_id_int * 3 + 1] + force_y[i].item()
                self.F_global[node_id_int * 3 + 2] = self.F_global[node_id_int * 3 + 2] + force_z[i].item()

        elif axis == 1:  # Torque about y-axis (use x,z for forces)
            r_perp = torch.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 2]**2)
            r_perp = torch.clamp(r_perp, min=tol)

            force_x = magnitude / r_perp * pos_rel[:, 2]
            force_z = -magnitude / r_perp * pos_rel[:, 0]

            for i, node_id in enumerate(node_ids):
                node_id_int = int(node_id.item())  # Convert tensor to Python int
                self.F_global[node_id_int * 3 + 0] = self.F_global[node_id_int * 3 + 0] + force_x[i].item()
                self.F_global[node_id_int * 3 + 2] = self.F_global[node_id_int * 3 + 2] + force_z[i].item()

        elif axis == 2:  # Torque about z-axis (use x,y for forces)
            r_perp = torch.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)
            r_perp = torch.clamp(r_perp, min=tol)

            force_x = -magnitude / r_perp * pos_rel[:, 1]
            force_y = magnitude / r_perp * pos_rel[:, 0]

            for i, node_id in enumerate(node_ids):
                node_id_int = int(node_id.item())  # Convert tensor to Python int
                self.F_global[node_id_int * 3 + 0] = self.F_global[node_id_int * 3 + 0] + force_x[i].item()
                self.F_global[node_id_int * 3 + 1] = self.F_global[node_id_int * 3 + 1] + force_y[i].item()

        else:
            raise ValueError(f"Invalid axis {axis}. Must be 0 (x), 1 (y), or 2 (z)")

        print(f"   Torque load applied: magnitude={magnitude:.3e}, axis={axis_name}, "
              f"center={center}, radius={radius:.4f}")
        print(f"   Applied to {len(node_ids)} nodes")

    def solve(self, density):
        """
        Solve FEM with differentiable operations.

        Parameters
        ----------
        density : torch.Tensor
            Element densities, shape (n_elements,)

        Returns
        -------
        dict
            "u": Displacement vector
            "compliance": Total compliance
            "compliance_e": Element compliance
        """
        # Assemble stiffness (fully differentiable)
        self._assemble_K(density)
        K = self.K_global
        F = self.F_global

        # Convert free_dofs to tensor for proper PyTorch indexing
        # CRITICAL: self.free_dofs is a NumPy array, must convert to tensor
        free_dofs_tensor = torch.tensor(self.free_dofs, dtype=torch.long, device=self.device)

        # Solve for free DOFs only
        try:
            K_ff = K[free_dofs_tensor, :][:, free_dofs_tensor]
            F_f = F[free_dofs_tensor]
            u_free = torch.linalg.solve(K_ff, F_f)
        except:
            # If singular, use least squares
            K_ff = K[free_dofs_tensor, :][:, free_dofs_tensor]
            F_f = F[free_dofs_tensor]
            u_free = torch.linalg.lstsq(K_ff, F_f).solution

        # Ensure u_free is 1D for easier indexing
        if u_free.dim() == 2:
            u_free = u_free.squeeze(1)

        # Build full displacement vector WITHOUT in-place operations
        # Use torch.cat to construct the full u vector (fully differentiable)
        u_list = []
        free_idx = 0

        for dof_id in range(self.n_dofs):
            if dof_id in self.fixed_dofs:
                # Fixed DOF: zero (no gradient contribution)
                u_list.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
            else:
                # Free DOF: solution value (gradient flows through!)
                u_list.append(u_free[free_idx])
                free_idx += 1

        # Stack all into a single column vector (fully differentiable)
        self.u = torch.stack(u_list, dim=0).reshape(-1, 1)  # Shape: (n_dofs, 1)

        # Compute element-wise compliance: c_e = u_e^T @ K_e @ u_e
        # Extract element displacements: (n_elems, 24)
        u_elem = self.u[self.edofMat].reshape(self.nx*self.ny*self.nz, 24)  # (n_elems, 24)

        # CRITICAL FIX #3: Scale element stiffness matrices using density
        # NumPy version scales before computing compliance:
        # KE_scaled = self.KE * scale_factors[:, np.newaxis, np.newaxis]
        # PyTorch must do the same!
        E_eff = self.Emin + self.density**self.penalty * (self.E_mod - self.Emin)
        scale_factors = E_eff / self.E_mod  # Shape: (n_elems,)
        KE_scaled = self.KE.unsqueeze(0) * scale_factors.view(-1, 1, 1)  # (n_elems, 24, 24)

        # Compute K_e @ u_e for each element using SCALED stiffness: (n_elems, 24)
        # FIX: Use einsum to properly compute K @ u (not u @ K^T)
        # Format: 'eij,ej->ei' means K_e[i,j] * u_e[j] summed over j
        Ku_elem = torch.einsum('eij,ej->ei', KE_scaled, u_elem)  # (n_elems, 24)

        # Element compliance: c_e = u_e^T @ K_e @ u_e = sum_i(u_e_i * (K_e @ u_e)_i)
        # FIX: Remove 0.5 factor to match NumPy (numpy doesn't include 0.5)
        compliance_e = torch.sum(u_elem * Ku_elem, dim=1)
        total_compliance = compliance_e.sum()


        return {
            "u": self.u,
            "compliance": total_compliance,
            "compliance_e": compliance_e,
        }

    def compute_compliance_sensitivities(self, u):
        """
        Compute element compliance sensitivities for topology optimization (VECTORIZED).

        Sensitivity: dc_e/dρ = -p * ρ^(p-1) * (E0 - Emin) / E0 * u_e^T K_e^base u_e

        Uses vectorized operations instead of loops for speed.

        Parameters
        ----------
        u : torch.Tensor
            Displacement vector (n_dofs,)

        Returns
        -------
        sensitivities : torch.Tensor
            dc/dρ for each element (n_elems,)
        """
        # Extract element displacements and reshape to (n_elems, 24)
        # u has shape (n_dofs, 1), indexing creates (n_elems, 24, 1), reshape to (n_elems, 24)
        u_e = u[self.edofMat].reshape(self.n_elems, 24)

        # Compute strain energy for all elements using base element stiffness
        # strain_energy = u_e^T @ KE @ u_e for each element
        Ku_e = torch.einsum('ij,ej->ei', self.KE, u_e)  # (n_elems, 24)
        strain_energy = torch.einsum('ei,ei->e', u_e, Ku_e)  # (n_elems,)

        # SIMP sensitivity: dc/dρ = -p * ρ^(p-1) * (E0 - Emin) / E0 * strain_energy
        # Vectorized for all elements
        rho = self.density
        sensitivity_coeff = -self.penalty * (rho**(self.penalty - 1)) * \
                           (self.E_mod - self.Emin) / self.E_mod

        # Avoid division by zero (rho > 0)
        valid = rho > 0
        sensitivities = torch.zeros(self.n_elems, dtype=torch.float32, device=self.device)
        sensitivities[valid] = sensitivity_coeff[valid] * strain_energy[valid]

        return sensitivities

    def reset(self):
        """Reset forces and boundary conditions for next solve."""
        if self.F_global is not None:
            self.F_global = torch.zeros(self.n_dofs, device=self.device, dtype=torch.float32)
            self.F_global.requires_grad = False
        self.fixed_dofs = set()


# Test function
if __name__ == "__main__":
    print("Testing PyTorch FEM Solver...")
    
    # Create solver
    fem = HexFEMSolver3D_Torch(device='cpu')
    fem.set_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=10, ny=2, nz=2)
    fem.fix_face(axis=0, coord=0.0)
    fem.add_point_load(location=(1.0, 0.05, 0.05), direction=1, magnitude=1e4)
    
    # Test with fixed density
    density = torch.ones(fem.n_elems, requires_grad=True)
    result = fem.solve(density)
    
    print(f"✅ Forward pass works!")
    print(f"   Compliance: {result['compliance'].item():.6e}")
    
    # Test backprop
    loss = result['compliance']
    loss.backward()
    
    print(f"✅ Backprop works!")
    print(f"   Density gradient shape: {density.grad.shape}")
    print(f"   Gradient norm: {torch.norm(density.grad):.6e}")
