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
        self.physical_elem_mask = None
        self.design_elem_mask = None
        self.passive_solid_mask = None
        self.physical_node_mask = None
        self.coupling_geometry = None

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
        self.physical_elem_mask = np.ones(self.n_elems, dtype=bool)
        self.design_elem_mask = np.ones(self.n_elems, dtype=bool)
        self.passive_solid_mask = np.zeros(self.n_elems, dtype=bool)
        self.physical_node_mask = np.ones(len(nodes), dtype=bool)
        self.coupling_geometry = None

        # Assemble stiffness matrix with default uniform density
        self._assemble_K()

    def set_cylindrical_coupling_domain(self, radius=None, interface_length=0.08, axis=0):
        """Restrict the structured box mesh to a cylindrical shaft-coupling domain.

        The mesh itself remains structured Hex8 for compatibility with the existing
        FEM/SIMP code. Elements whose centroids lie outside the cylinder are marked
        passive void. Short regions at both ends are marked passive solid so that
        the coupling keeps robust shaft interfaces for support and load transfer.

        Currently the website uses a shaft aligned with +X (axis=0).
        """
        if axis != 0:
            raise NotImplementedError("Shaft coupling domain currently supports the X axis only")
        if self.nodes_np is None or self.elems_t is None:
            raise RuntimeError("Call set_mesh() before set_cylindrical_coupling_domain()")

        mins = self.nodes_np.min(axis=0)
        maxs = self.nodes_np.max(axis=0)
        center_y = 0.5 * (mins[1] + maxs[1])
        center_z = 0.5 * (mins[2] + maxs[2])
        max_radius = 0.5 * min(maxs[1] - mins[1], maxs[2] - mins[2])
        if radius is None:
            radius = max_radius
        radius = float(radius)
        if radius <= 0 or radius > max_radius + 1e-12:
            raise ValueError(f"coupling radius must be in (0, {max_radius:.6g}] m")

        elem_centers = np.mean(self.nodes_np[self.elems_t], axis=1)
        er = np.sqrt((elem_centers[:, 1] - center_y) ** 2 + (elem_centers[:, 2] - center_z) ** 2)
        physical = er <= radius + 1e-12

        x_min, x_max = mins[0], maxs[0]
        interface_length = max(0.0, float(interface_length))
        interface_length = min(interface_length, 0.45 * (x_max - x_min))
        end_regions = ((elem_centers[:, 0] <= x_min + interface_length + 1e-12) |
                       (elem_centers[:, 0] >= x_max - interface_length - 1e-12))
        passive_solid = physical & end_regions
        design = physical & ~passive_solid

        nr = np.sqrt((self.nodes_np[:, 1] - center_y) ** 2 + (self.nodes_np[:, 2] - center_z) ** 2)
        physical_nodes = nr <= radius + 1e-12

        self.physical_elem_mask = physical
        self.passive_solid_mask = passive_solid
        self.design_elem_mask = design
        self.physical_node_mask = physical_nodes
        self.coupling_geometry = {
            "axis": "x",
            "radius": radius,
            "center_y": center_y,
            "center_z": center_z,
            "interface_length": interface_length,
            "physical_elements": int(np.sum(physical)),
            "design_elements": int(np.sum(design)),
            "passive_solid_elements": int(np.sum(passive_solid)),
        }
        if not np.any(design):
            raise ValueError("No design elements remain; reduce interface_length or refine/increase the mesh")
        return self.coupling_geometry

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
        if self.physical_node_mask is not None:
            mask &= self.physical_node_mask
        for node_id in np.where(mask)[0]:
            for dof_offset in range(3):
                self.fixed_dofs.add(node_id * 3 + dof_offset)

    def add_distributed_load(
        self, axis=0, coord=1.0, direction=0, total=1e4, tol=1e-6
    ):
        """Add distributed load on a face."""
        mask = np.abs(self.nodes_np[:, axis] - coord) < tol
        if self.physical_node_mask is not None:
            mask &= self.physical_node_mask
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
        if self.physical_node_mask is not None:
            dists = np.where(self.physical_node_mask, dists, np.inf)
        node_id = np.argmin(dists)

        # Apply force to that node
        dof = node_id * 3 + direction
        self.F_global[dof] += magnitude

        print(f"   Point load applied at node {node_id}, location ≈ {self.nodes_np[node_id]}")

    def add_torque_load(self, axis=0, coord=1.0, torque=100.0, tol=1e-6):
        """Apply a pure torque to a mesh face using a balanced tangential force field.

        Parameters
        ----------
        axis : int
            Torque axis and face-normal axis: 0=x, 1=y, 2=z.
        coord : float
            Coordinate of the loaded face along ``axis``.
        torque : float
            Requested resultant torque in N*m (mesh coordinates are metres).
        tol : float
            Face-selection tolerance.

        Notes
        -----
        The nodal forces are constructed as F_i = alpha * (a x r_i), where
        ``a`` is the unit torque axis and ``r_i`` is measured from the face
        centroid.  Alpha is chosen so that sum((r_i x F_i) . a) = torque.
        For a structured symmetric face the resultant translational force is
        numerically zero, while the requested moment is preserved.
        """
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")

        mask = np.abs(self.nodes_np[:, axis] - coord) < tol
        if self.physical_node_mask is not None:
            mask &= self.physical_node_mask
        node_ids = np.where(mask)[0]
        if len(node_ids) == 0:
            raise ValueError(f"No nodes found on torque face axis={axis}, coord={coord}")

        face_nodes = self.nodes_np[node_ids]
        centroid = np.mean(face_nodes, axis=0)
        axis_vec = np.zeros(3, dtype=np.float64)
        axis_vec[axis] = 1.0

        rel = face_nodes - centroid
        tangents = np.cross(axis_vec[None, :], rel)
        denom = float(np.sum(np.sum(tangents * tangents, axis=1)))
        if denom <= 1e-20:
            raise ValueError("Torque face has zero effective radius; increase transverse mesh size")

        alpha = float(torque) / denom
        nodal_forces = alpha * tangents

        for node_id, force_vec in zip(node_ids, nodal_forces):
            base = node_id * 3
            self.F_global[base:base + 3] += force_vec

        resultant_force = np.sum(nodal_forces, axis=0)
        resultant_moment = np.sum(np.cross(rel, nodal_forces), axis=0)
        print(
            f"   Torque load applied on axis {axis} face @ {coord}: "
            f"requested={torque:.6g} N*m, achieved={resultant_moment[axis]:.6g} N*m, "
            f"|sumF|={np.linalg.norm(resultant_force):.3e} N"
        )
        return {
            "node_ids": node_ids,
            "resultant_force": resultant_force,
            "resultant_moment": resultant_moment,
        }

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

    def reset(self):
        """Reset forces for next solve."""
        self.F_global = np.zeros(self.n_dofs)
        self.fixed_dofs = set()
