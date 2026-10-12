# Shaft Coupling Version — Changes

This version is built directly from the original 3D topology-optimization project.

## Implemented

- Combined **Force + Torque** loading in the same FEM global load vector.
- Pure torque is converted into balanced tangential nodal forces on the loaded face.
- Added a **cylindrical shaft-coupling design domain** while retaining the structured Hex8 mesh.
- Elements outside the coupling cylinder are **passive void** and are forced to density 0.
- Short regions at both coupling ends are **passive solid interfaces** and are forced to density 1.
- Only the middle cylindrical region is designable by SIMP.
- Boundary and distributed/torque loads automatically select physical cylinder-face nodes only.
- Point loads snap to the nearest physical shaft/coupling node.
- Web UI now includes:
  - Coupling Length
  - Mesh Box Y/Z
  - Coupling Radius
  - Solid End Interface Length
  - Force settings
  - Torque settings
  - SIMP settings
- Domain preview now shows a shaft/coupling cylinder with solid end interfaces.

## Verified numerically

A 100 N·m torque test produced a resultant moment of 100 N·m and near-zero resultant translational force (floating-point error only).

A combined transverse force + torque case also runs through the masked SIMP optimizer. Passive void remains density 0 and passive end interfaces remain density 1.

## Current modeling assumption

The shaft axis is +X. The cylinder is embedded in the existing structured Hex8 box mesh. This preserves compatibility with the original FEM/SIMP implementation and avoids rewriting the solver from scratch.
