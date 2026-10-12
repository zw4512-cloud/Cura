# Teacher Files Included in V4

This folder contains the teacher-provided code as a project reference and future integration target.

## Core baseline files

- `teacher_core/fem3d_numpy.py`  
  NumPy Hex8 FEM solver.

- `teacher_core/fem3d_torch.py`  
  PyTorch differentiable Hex8 FEM solver.

- `teacher_core/simp_numpy.py`  
  NumPy SIMP topology optimizer.

- `teacher_core/neural_network_topology_optimization.py`  
  Neural-network-based topology optimizer.

## Tests

- `teacher_tests/test_torque_load.py`  
  Torque-load tests, including torsion around X/Y/Z, combined bending + torsion, NumPy/Torch comparison, and topology optimization with torque.

- `teacher_tests/test_nn_optimizer.py`  
  Small neural-network optimizer integration test.

## Example script

- `examples/run_optimization_numpy.py`  
  Original NumPy SIMP running example.

## How this relates to the website

The top-level website still runs from `app.py`. The teacher files are included for preservation and future connection. The current website can be developed normally without deleting or changing these teacher files.
