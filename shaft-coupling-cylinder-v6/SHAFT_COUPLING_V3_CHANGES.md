# Shaft Coupling V3 changes

This version adds an interactive shaft-coupling load preview on top of the existing cylindrical FEM/SIMP model.

## New UI features

- Left shaft + central coupling + right shaft visualization
- Separate Shaft Radius control for the preview
- Coupling design domain shown in blue
- Solid end interfaces shown in green
- Fixed left end marker
- Force arrow shown at the loaded right shaft end
- Torque shown as a curved arrow around the selected torque axis
- Force direction/sign updates the arrow immediately
- Torque axis/sign updates the torque arrow immediately
- Drag the preview with the mouse to rotate it
- Use the mouse wheel to zoom
- Small XYZ axis gizmo

## Solver behavior

The FEM/SIMP implementation remains the cylindrical-coupling solver from V2. `shaft_radius` is a visualization parameter for the external shafts. The optimized design domain remains the central cylindrical coupling, with solid end interfaces and combined force + torque loading.
