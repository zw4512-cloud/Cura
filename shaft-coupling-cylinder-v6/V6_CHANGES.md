# V6 Changes — Quick Run / Normal Run

V6 keeps the V5 optimizer engines and teacher files, and adds safer run controls.

## Run modes

### Quick Run
- Uses the user's current settings but caps the actual run at:
  - Elements X: 8
  - Elements Y: 6
  - Elements Z: 6
  - Iterations: 10
- Intended for fast previews and parameter checks.
- The actual mesh and iteration count are included in the result summary.

### Normal Run
- Uses the user's selected mesh and iteration settings.
- Website safety limits:
  - Elements X: 20
  - Elements Y: 15
  - Elements Z: 15
  - Iterations: 100
- Requests above these limits are rejected with a clear message.

These are website safety limits, not theoretical FEM limits.

## UI additions
- Separate Quick Run and Normal Run buttons.
- Maximum values shown beside mesh and iteration fields.
- Live element count and node count before running.
- Result summary now records Run Mode and actual parameters.

## Backend protection
- Flask validates `run_mode`.
- Quick requests are capped again server-side.
- Normal requests are rejected if they exceed the website limits.

## Teacher files
The files under `teacher_core/`, `teacher_tests/`, `teacher_original_files/`, and `examples/` are retained from V5. V6 does not modify the teacher core algorithms.
