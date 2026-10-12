# Shaft Coupling V5 changes

V5 keeps the V4 shaft-coupling website and all teacher files, and adds a real adapter layer so the website can select different optimization engines.

## Website selector

- Shaft SIMP — website cylindrical coupling implementation
- Teacher NumPy SIMP — calls `teacher_core/fem3d_numpy.py` + `teacher_core/simp_numpy.py`
- Teacher NN / PyTorch — calls `teacher_core/fem3d_torch.py` + `teacher_core/neural_network_topology_optimization.py`

Teacher files are not modified. `teacher_adapter.py` translates website inputs to their existing APIs.

## Geometry distinction

The Shaft SIMP mode uses the website cylindrical coupling design mask.
The teacher baseline modes deliberately keep the teacher code's original rectangular structured mesh, so their baseline behavior is preserved.

## PyTorch

Teacher NN mode requires PyTorch. If it is not installed in the active environment, the website returns a clear error rather than silently using another optimizer.
