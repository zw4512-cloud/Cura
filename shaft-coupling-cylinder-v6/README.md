# Shaft Coupling Cylinder V4

This version keeps the interactive shaft-coupling website from V3 and also includes the teacher's original optimization files.

## What to run for the website

Use the top-level website files:

- `app.py`
- `web/index.html`
- `web/app.js`
- `web/styles.css`
- `fem3d_numpy.py`
- `simp_numpy.py`

Windows / Anaconda Prompt:

```bat
cd C:\Users\34258\Desktop\shaft-coupling-cylinder-v4
conda activate shaft
pip install -r requirements_shaft.txt
python app.py
```

Then open:

```text
http://127.0.0.1:18080
```

## Teacher files included

The teacher's core code is preserved separately under:

```text
teacher_core/
  fem3d_numpy.py
  fem3d_torch.py
  simp_numpy.py
  neural_network_topology_optimization.py

teacher_tests/
  test_torque_load.py
  test_nn_optimizer.py

examples/
  run_optimization_numpy.py
```

The original uploaded filenames are also copied under:

```text
teacher_original_files/
```

## Important principle

The website code is kept separate from the teacher's original files. This avoids accidentally overwriting or heavily changing the teacher baseline. Future work can connect the website to teacher_core through adapter files, but the teacher files themselves should stay mostly unchanged.

## Suggested final direction

1. Keep the current website as the interactive front end.
2. Keep the teacher files as the algorithm baseline.
3. Add a small adapter layer if needed, instead of rewriting teacher code.
4. Expose only safe parameters on the website, such as mesh size, volume fraction, penalty, filter radius, iterations, force, and torque.

## V5 optimization engine selector

V5 adds a website selector for Shaft SIMP, Teacher NumPy SIMP, and Teacher NN/PyTorch. The files in `teacher_core/` are preserved; `teacher_adapter.py` is the integration layer.

## V6 run modes

V6 adds two run buttons:

- **Quick Run**: fast preview, capped at 8×6×6 elements and 10 iterations.
- **Normal Run**: uses the chosen settings, with website limits of 20×15×15 elements and 100 iterations.

The limits are website safety limits intended to prevent accidental very large CPU jobs; they are not theoretical FEM limits.
