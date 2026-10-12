"""
3D Finite Element Method — Hex8 (Pure NumPy, no PyTorch)
========================================================
A minimal implementation for topology optimization without PyTorch dependency.
"""

import numpy as np
from itertools import product as iproduct


class HexFEMSolver3D:
    """3D FEM Solver using Hex8 elements with pure NumPy/SciPy."""

    def __init__(self, E_mod=200e9, nu=0.3, Emin=1e-3, penalty=3.0):
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
        """
        self.E_mod = E_mod
        self.nu = nu
        self.Emin = Emin
        self.penalty = penalty

        self.nodes_np = None
        self.elems_t = None  # Keep as numpy array but with _t name for API compatibility
        self.n_dofs = None
        self.K_global = None
        self.F_global = None
        self.fixed_dofs = set()
        self.n_elems = 0
        self.density = None  # For topology optimization

        # Pre-computed for vectorized assembly
        self.KE = None  # Reference element stiffness (24, 24)
        self.iK = None  # Row indices for COO assembly
        self.jK = None  # Column indices for COO assembly
        self.edofMat = None  # Element DOF matrix (n_elems, 24)

    def set_mesh(self, Lx=1.0, Ly=0.1, Lz=0.1, nx=10, ny=3, nz=3):
        """Create structured hex mesh."""
        xs = np.linspace(0, Lx, nx + 1)
        ys = np.linspace(0, Ly, ny + 1)
        zs = np.linspace(0, Lz, nz + 1)

        def nid(i, j, k):
            return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

        nodes = np.array(
            [[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64
        )

        elems = []
        for i, j, k in iproduct(range(nx), range(ny), range(nz)):
            elems.append(
                [
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]
            )

        self.nodes_np = nodes
        self.elems_t = np.array(elems, dtype=np.int64)
        self.n_elems = len(elems)
        self.n_dofs = len(nodes) * 3
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Initialize global arrays (dense for small problems)
        self.K_global = np.zeros((self.n_dofs, self.n_dofs))
        self.F_global = np.zeros(self.n_dofs)
        self.fixed_dofs = set()
        self.density = np.ones(self.n_elems)  # Default: uniform density

        # Assemble stiffness matrix with default uniform density
        self._assemble_K()

    def _assemble_K(self, density=None):
        """
        Assemble global stiffness matrix with vectorized operations and sparse matrix format.

        Uses COO (coordinate) format for efficient assembly, then converts to dense.

        Parameters
        ----------
        density : ndarray, optional
            Element densities ∈ [0,1] for topology optimization (SIMP).
            If None, uses uniform density = 1.0.
            Shape: (n_elements,)
        """
        # Use provided density or default to uniform
        if density is not None:
            if len(density) != self.n_elems:
                raise ValueError(f"Density length {len(density)} != n_elems {self.n_elems}")
            self.density = density.copy()
        else:
            self.density = np.ones(self.n_elems)

        # Pre-compute reference element stiffness and assembly indices (only once)
        if self.KE is None:
            self._precompute_assembly_data()

        # VECTORIZED SIMP SCALING
        # Compute effective modulus for all elements: E(ρ) = Emin + ρ^p * (E0 - Emin)
        E_eff = self.Emin + self.density**self.penalty * (self.E_mod - self.Emin)

        # Scale factor for each element (compared to base modulus)
        scale_factors = E_eff / self.E_mod  # Shape: (n_elems,)

        # Flatten and broadcast: replicate KE for each element with its scale factor
        # KE is (24, 24), flattened to 576
        # We need to replicate this 576 times per element and multiply by scale factor
        sK = (self.KE.flatten()[np.newaxis, :] * scale_factors[:, np.newaxis]).flatten(order='C')

        # Assemble into dense matrix using NumPy accumulation (no scipy needed)
        # Convert (row, col) indices to raveled indices for 1D accumulation
        self.K_global = np.zeros((self.n_dofs, self.n_dofs))
        linear_indices = self.iK * self.n_dofs + self.jK  # Raveled indices
        np.add.at(self.K_global.ravel(), linear_indices, sK)

    def _precompute_assembly_data(self):
        """
        Pre-compute reference element stiffness, assembly indices, and edofMat.

        This is called once during mesh setup. For structured uniform meshes,
        all elements have identical stiffness (just different global DOF indices).
        """
        D = self._constitutive_matrix()

        # Compute reference element stiffness (using first element as template)
        first_elem_nodes = self.elems_t[0]
        first_elem_coords = self.nodes_np[first_elem_nodes]
        self.KE = self._hex8_stiffness(first_elem_coords, D)

        # Pre-compute edofMat: element DOF matrix (n_elems, 24)
        # Maps each element to its 24 global DOF indices
        edofMat_list = []
        iK_list = []
        jK_list = []

        for elem_id, elem_nodes in enumerate(self.elems_t):
            # Global DOF indices for this element (each node has 3 DOFs)
            dofs = []
            for node in elem_nodes:
                dofs.extend([node * 3, node * 3 + 1, node * 3 + 2])
            dofs = np.array(dofs)
            edofMat_list.append(dofs)

            # For (24, 24) element stiffness, add all pairs (i, j)
            for i in range(24):
                for j in range(24):
                    iK_list.append(dofs[i])
                    jK_list.append(dofs[j])

        self.edofMat = np.array(edofMat_list, dtype=np.int32)
        self.iK = np.array(iK_list, dtype=np.int32)
        self.jK = np.array(jK_list, dtype=np.int32)

    def _constitutive_matrix(self):
        """Build 6x6 constitutive matrix for isotropic elasticity."""
        E = self.E_mod
        nu = self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
        D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def _hex8_stiffness(self, coords, D):
        """Compute element stiffness matrix for Hex8 element."""
        # 2x2x2 Gauss quadrature
        gp = np.array([-1, 1]) / np.sqrt(3)
        Ke = np.zeros((24, 24))

        for gxi, geta, gzeta in iproduct(gp, repeat=3):
            # Shape function derivatives
            N_xi = np.array(
                [
                    -(1 - geta) * (1 - gzeta),
                    (1 - geta) * (1 - gzeta),
                    (1 + geta) * (1 - gzeta),
                    -(1 + geta) * (1 - gzeta),
                    -(1 - geta) * (1 + gzeta),
                    (1 - geta) * (1 + gzeta),
                    (1 + geta) * (1 + gzeta),
                    -(1 + geta) * (1 + gzeta),
                ]
            )
            N_eta = np.array(
                [
                    -(1 - gxi) * (1 - gzeta),
                    -(1 + gxi) * (1 - gzeta),
                    (1 + gxi) * (1 - gzeta),
                    (1 - gxi) * (1 - gzeta),
                    -(1 - gxi) * (1 + gzeta),
                    -(1 + gxi) * (1 + gzeta),
                    (1 + gxi) * (1 + gzeta),
                    (1 - gxi) * (1 + gzeta),
                ]
            )
            N_zeta = np.array(
                [
                    -(1 - gxi) * (1 - geta),
                    -(1 + gxi) * (1 - geta),
                    -(1 + gxi) * (1 + geta),
                    -(1 - gxi) * (1 + geta),
                    (1 - gxi) * (1 - geta),
                    (1 + gxi) * (1 - geta),
                    (1 + gxi) * (1 + geta),
                    (1 - gxi) * (1 + geta),
                ]
            )
            N_xi *= 0.125
            N_eta *= 0.125
            N_zeta *= 0.125

            # Jacobian matrix
            J = np.zeros((3, 3))
            J[0, :] = N_xi @ coords
            J[1, :] = N_eta @ coords
            J[2, :] = N_zeta @ coords
            detJ = np.linalg.det(J)

            if abs(detJ) < 1e-15:
                continue

            J_inv = np.linalg.inv(J)

            # Global shape function derivatives
            dNdx = J_inv[0, 0] * N_xi + J_inv[0, 1] * N_eta + J_inv[0, 2] * N_zeta
            dNdy = J_inv[1, 0] * N_xi + J_inv[1, 1] * N_eta + J_inv[1, 2] * N_zeta
            dNdz = J_inv[2, 0] * N_xi + J_inv[2, 1] * N_eta + J_inv[2, 2] * N_zeta

            # Strain-displacement matrix B (24x6)
            B = np.zeros((6, 24))
            for i in range(8):
                B[0, 3 * i] = dNdx[i]
                B[1, 3 * i + 1] = dNdy[i]
                B[2, 3 * i + 2] = dNdz[i]
                B[3, 3 * i + 1] = dNdz[i]
                B[3, 3 * i + 2] = dNdy[i]
                B[4, 3 * i] = dNdz[i]
                B[4, 3 * i + 2] = dNdx[i]
                B[5, 3 * i] = dNdy[i]
                B[5, 3 * i + 1] = dNdx[i]

            # Element stiffness
            Ke += B.T @ D @ B * abs(detJ)

        return Ke

    def fix_face(self, axis=0, coord=0.0, tol=1e-6):
        """Fix all nodes on a face."""
        mask = np.abs(self.nodes_np[:, axis] - coord) < tol
        for node_id in np.where(mask)[0]:
            for dof_offset in range(3):
                self.fixed_dofs.add(node_id * 3 + dof_offset)

    def add_distributed_load(
        self, axis=0, coord=1.0, direction=0, total=1e4, tol=1e-6
    ):
        """Add distributed load on a face."""
        mask = np.abs(self.nodes_np[:, axis] - coord) < tol
        n_nodes = np.sum(mask)

        if n_nodes > 0:
            load_per_node = total / n_nodes
            for node_id in np.where(mask)[0]:
                dof = node_id * 3 + direction
                self.F_global[dof] += load_per_node

    def add_point_load(self, location, direction, magnitude):
        """
        Add concentrated point load at a specific location.

        Parameters
        ----------
        location : tuple or list
            (x, y, z) coordinates of load application point
        direction : int
            0=x, 1=y, 2=z
        magnitude : float
            Force magnitude
        """
        # Find node closest to the specified location
        dists = np.sum((self.nodes_np - np.array(location))**2, axis=1)
        node_id = np.argmin(dists)

        # Apply force to that node
        dof = node_id * 3 + direction
        self.F_global[dof] += magnitude

        print(f"   Point load applied at node {node_id}, location ≈ {self.nodes_np[node_id]}")

    def solve(self, density=None):
        """
        Solve linear system with boundary conditions.

        Parameters
        ----------
        density : ndarray, optional
            Element densities for this solve. Shape (n_elements,), values in [0, 1].
            If None, uses the stored self.density (or uniform if not set).
            This parameter allows clean stateless solving without update_density().

        Returns
        -------
        dict
            "u": Displacement vector (n_dofs,)
            "compliance": Total compliance (scalar)
            "compliance_e": Element compliance (n_elems,)
            "K": Global stiffness matrix
            "F": Global force vector
        """
        # Assemble stiffness with provided or stored density
        self._assemble_K(density)

        # Only copy F (K_global was just regenerated and won't be reused)
        F = self.F_global.copy()

        # Apply Dirichlet BCs by modifying K_global directly and F
        for dof in self.fixed_dofs:
            self.K_global[dof, :] = 0
            self.K_global[:, dof] = 0
            self.K_global[dof, dof] = 1.0
            F[dof] = 0.0

        # Solve linear system K u = F
        try:
            u = np.linalg.solve(self.K_global, F)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix, using least squares solution")
            u = np.linalg.lstsq(self.K_global, F, rcond=None)[0]

        # Compute element compliance
        compliance_e = self._compute_element_compliance(u)
        total_compliance = np.sum(compliance_e)

        # Compute sensitivities
        sensitivities = self.compute_compliance_sensitivities(u)

        return {
            "u": u,
            "compliance": total_compliance,
            "compliance_e": compliance_e,
            "K": self.K_global,
            "F": self.F_global,
            "sensitivities": sensitivities,
            "sigma": None,
            "von_mises": None,
        }

    def _compute_element_compliance(self, u):
        """
        Compute element-wise compliance from displacement vector (VECTORIZED).

        Element compliance: c_e = u_e^T K_e u_e

        Uses vectorized operations with einsum for maximum speed.

        Parameters
        ----------
        u : ndarray
            Displacement vector (n_dofs,)

        Returns
        -------
        compliance_e : ndarray
            Element compliance (n_elems,)
        """
        # Extract element displacements: (n_elems, 24)
        u_e = u[self.edofMat]

        # Compute density scaling factors for all elements
        E_eff = self.Emin + self.density**self.penalty * (self.E_mod - self.Emin)
        scale_factors = E_eff / self.E_mod  # Shape: (n_elems,)

        # Scale reference element stiffness: (n_elems, 24, 24)
        # Broadcasting: KE (24, 24) * scale_factors (n_elems, 1, 1) -> (n_elems, 24, 24)
        KE_scaled = self.KE * scale_factors[:, np.newaxis, np.newaxis]

        # Vectorized compliance: c_e = sum_j(u_e[j] * (K_e @ u_e)[j])
        # Using einsum for efficiency: 'eij,ej,ej->e' means:
        # - e: element index
        # - i,j: matrix indices for K (i=row, j=col)
        # - Compute K @ u, then element-wise multiply with u, then sum
        Ku_e = np.einsum('eij,ej->ei', KE_scaled, u_e)  # (n_elems, 24)
        compliance_e = np.einsum('ei,ei->e', u_e, Ku_e)  # (n_elems,)

        return compliance_e

    def compute_compliance_sensitivities(self, u):
        """
        Compute element compliance sensitivities for topology optimization (VECTORIZED).

        Sensitivity: dc_e/dρ = -p * ρ^(p-1) * (E0 - Emin) / E0 * u_e^T K_e^base u_e

        Uses vectorized operations instead of loops for speed.

        Parameters
        ----------
        u : ndarray
            Displacement vector (n_dofs,)

        Returns
        -------
        sensitivities : ndarray
            dc/dρ for each element (n_elems,)
        """
        # Extract element displacements: (n_elems, 24)
        u_e = u[self.edofMat]

        # Compute strain energy for all elements using base element stiffness
        # strain_energy = u_e^T @ KE @ u_e for each element
        Ku_e = np.einsum('ij,ej->ei', self.KE, u_e)  # (n_elems, 24)
        strain_energy = np.einsum('ei,ei->e', u_e, Ku_e)  # (n_elems,)

        # SIMP sensitivity: dc/dρ = -p * ρ^(p-1) * (E0 - Emin) / E0 * strain_energy
        # Vectorized for all elements
        rho = self.density
        sensitivity_coeff = -self.penalty * (rho**(self.penalty - 1)) * \
                           (self.E_mod - self.Emin) / self.E_mod

        # Avoid division by zero (rho > 0)
        valid = rho > 0
        sensitivities = np.zeros(self.n_elems)
        sensitivities[valid] = sensitivity_coeff[valid] * strain_energy[valid]

        return sensitivities

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
        center = np.array(center)
        axis_map = {0: 'x', 1: 'y', 2: 'z'}
        axis_name = axis_map.get(axis, 'unknown')

        # Find nodes to apply torque to
        distances = np.linalg.norm(self.nodes_np - center, axis=1)

        # Determine radius if not specified
        if radius is None:
            radius = np.max(distances)

        # Get nodes within radius
        mask = distances <= (radius + tol)
        node_ids = np.where(mask)[0]

        if len(node_ids) == 0:
            print(f"   Warning: No nodes found within radius {radius} of center {center}")
            return

        # Calculate position vectors relative to center
        pos_rel = self.nodes_np[node_ids] - center

        # Apply torque as equivalent nodal forces
        # τ = r × F, so F = τ / |r| (magnitude per unit distance)

        if axis == 0:  # Torque about x-axis (use y,z for forces)
            # r = (0, y, z), F should be perpendicular and create moment about x
            # τ_x = r_y * F_z - r_z * F_y
            # Simple approach: F_z = magnitude / |r_yz| * r_y, F_y = -magnitude / |r_yz| * r_z
            r_perp = np.sqrt(pos_rel[:, 1]**2 + pos_rel[:, 2]**2)  # Distance in yz plane
            r_perp[r_perp < tol] = tol  # Avoid division by zero

            force_y = -magnitude / r_perp * pos_rel[:, 2]  # Perpendicular force in y
            force_z = magnitude / r_perp * pos_rel[:, 1]   # Perpendicular force in z

            for i, node_id in enumerate(node_ids):
                self.F_global[node_id * 3 + 1] += force_y[i]
                self.F_global[node_id * 3 + 2] += force_z[i]

        elif axis == 1:  # Torque about y-axis (use x,z for forces)
            # r = (x, 0, z), F should create moment about y
            # τ_y = r_z * F_x - r_x * F_z
            r_perp = np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 2]**2)  # Distance in xz plane
            r_perp[r_perp < tol] = tol

            force_x = magnitude / r_perp * pos_rel[:, 2]   # Perpendicular force in x
            force_z = -magnitude / r_perp * pos_rel[:, 0]  # Perpendicular force in z

            for i, node_id in enumerate(node_ids):
                self.F_global[node_id * 3 + 0] += force_x[i]
                self.F_global[node_id * 3 + 2] += force_z[i]

        elif axis == 2:  # Torque about z-axis (use x,y for forces)
            # r = (x, y, 0), F should create moment about z
            # τ_z = r_x * F_y - r_y * F_x
            r_perp = np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)  # Distance in xy plane
            r_perp[r_perp < tol] = tol

            force_x = -magnitude / r_perp * pos_rel[:, 1]  # Perpendicular force in x
            force_y = magnitude / r_perp * pos_rel[:, 0]   # Perpendicular force in y

            for i, node_id in enumerate(node_ids):
                self.F_global[node_id * 3 + 0] += force_x[i]
                self.F_global[node_id * 3 + 1] += force_y[i]

        else:
            raise ValueError(f"Invalid axis {axis}. Must be 0 (x), 1 (y), or 2 (z)")

        print(f"   Torque load applied: magnitude={magnitude:.3e}, axis={axis_name}, "
              f"center={center}, radius={radius:.4f}")
        print(f"   Applied to {len(node_ids)} nodes")

    def reset(self):
        """Reset forces for next solve."""
        self.F_global = np.zeros(self.n_dofs)
        self.fixed_dofs = set()
