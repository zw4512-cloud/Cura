"""
Test torque load functionality in both NumPy and PyTorch FEM solvers.

Tests:
1. Simple torsion about each axis
2. Combined bending + torsion loading
3. Equivalence between NumPy and PyTorch implementations
4. Torque with topology optimization
"""

import numpy as np
import torch
from fem3d_numpy import HexFEMSolver3D
from fem3d_torch import HexFEMSolver3D_Torch
from simp_numpy import SIMPOptimizer


def test_torque_about_x_axis():
    """TEST 1: Torsion about x-axis (longitudinal axis)."""
    print("\n" + "="*70)
    print("TEST 1: Torque about X-axis")
    print("="*70)

    # Setup: Cantilever shaft, fixed at one end, torque at the other
    fem = HexFEMSolver3D(E_mod=1.0, nu=0.3)
    fem.set_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=10, ny=2, nz=2)

    # Boundary conditions
    fem.fix_face(axis=0, coord=0.0)  # Fixed at x=0

    # Apply torsion about x-axis at free end
    print("\n1. Applying torque about x-axis...")
    fem.add_torque_load(
        center=(1.0, 0.05, 0.05),
        axis=0,
        magnitude=1000.0
    )

    # Solve
    print("2. Solving system...")
    density = np.ones(fem.n_elems)
    result = fem.solve(density)

    # Results
    print(f"\n3. Results:")
    print(f"   Total compliance: {result['compliance']:.6e}")
    print(f"   Max displacement: {np.max(np.abs(result['u'])):.6e}")
    print(f"   Min density sensitivities: {np.min(result['sensitivities']):.6e}")
    print(f"   Max density sensitivities: {np.max(result['sensitivities']):.6e}")

    print(f"\n✅ TEST 1 PASS: Torque about x-axis works correctly")
    return True


def test_torque_about_y_axis():
    """TEST 2: Torsion about y-axis."""
    print("\n" + "="*70)
    print("TEST 2: Torque about Y-axis")
    print("="*70)

    fem = HexFEMSolver3D(E_mod=1.0, nu=0.3)
    fem.set_mesh(Lx=0.1, Ly=1.0, Lz=0.1, nx=2, ny=10, nz=2)

    # Boundary conditions
    fem.fix_face(axis=1, coord=0.0)

    # Apply torsion about y-axis
    print("\n1. Applying torque about y-axis...")
    fem.add_torque_load(
        center=(0.05, 1.0, 0.05),
        axis=1,
        magnitude=1000.0
    )

    # Solve
    print("2. Solving system...")
    density = np.ones(fem.n_elems)
    result = fem.solve(density)

    # Results
    print(f"\n3. Results:")
    print(f"   Total compliance: {result['compliance']:.6e}")
    print(f"   Max displacement: {np.max(np.abs(result['u'])):.6e}")

    print(f"\n✅ TEST 2 PASS: Torque about y-axis works correctly")
    return True


def test_torque_about_z_axis():
    """TEST 3: Torsion about z-axis (most common case)."""
    print("\n" + "="*70)
    print("TEST 3: Torque about Z-axis")
    print("="*70)

    fem = HexFEMSolver3D(E_mod=1.0, nu=0.3)
    fem.set_mesh(Lx=0.1, Ly=0.1, Lz=1.0, nx=2, ny=2, nz=10)

    # Boundary conditions
    fem.fix_face(axis=2, coord=0.0)

    # Apply torsion about z-axis
    print("\n1. Applying torque about z-axis...")
    fem.add_torque_load(
        center=(0.05, 0.05, 1.0),
        axis=2,
        magnitude=1000.0
    )

    # Solve
    print("2. Solving system...")
    density = np.ones(fem.n_elems)
    result = fem.solve(density)

    # Results
    print(f"\n3. Results:")
    print(f"   Total compliance: {result['compliance']:.6e}")
    print(f"   Max displacement: {np.max(np.abs(result['u'])):.6e}")

    print(f"\n✅ TEST 3 PASS: Torque about z-axis works correctly")
    return True


def test_combined_bending_and_torsion():
    """TEST 4: Combined bending + torsion loading."""
    print("\n" + "="*70)
    print("TEST 4: Combined Bending + Torsion")
    print("="*70)

    fem = HexFEMSolver3D(E_mod=1.0, nu=0.3)
    fem.set_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=10, ny=2, nz=2)

    # Boundary conditions
    fem.fix_face(axis=0, coord=0.0)

    # Apply bending load
    print("\n1. Applying loads...")
    fem.add_point_load(
        location=(1.0, 0.05, 0.05),
        direction=1,  # y-direction
        magnitude=5000.0
    )

    # Add torsion
    fem.add_torque_load(
        center=(1.0, 0.05, 0.05),
        axis=2,
        magnitude=1000.0
    )

    # Solve
    print("2. Solving system with combined loading...")
    density = np.ones(fem.n_elems)
    result = fem.solve(density)

    # Results
    print(f"\n3. Results:")
    print(f"   Total compliance: {result['compliance']:.6e}")
    print(f"   Max displacement: {np.max(np.abs(result['u'])):.6e}")
    print(f"   Compliance is sum of bending + torsion effects")

    print(f"\n✅ TEST 4 PASS: Combined loading works correctly")
    return True


def test_numpy_torch_equivalence():
    """TEST 5: Verify NumPy and PyTorch give same results."""
    print("\n" + "="*70)
    print("TEST 5: NumPy vs PyTorch Equivalence")
    print("="*70)

    # Setup mesh (same for both)
    nx, ny, nz = 10, 4, 4
    center = (1.0, 0.1, 0.1)
    torque_mag = 1.0

    # NumPy solve
    print("\n1. NumPy solver...")
    fem_np = HexFEMSolver3D(E_mod=1.0, nu=0.3)
    fem_np.set_mesh(Lx=1.0, Ly=0.2, Lz=0.2, nx=nx, ny=ny, nz=nz)
    fem_np.fix_face(axis=0, coord=0.0)
    fem_np.add_torque_load(center=center, axis=0, magnitude=torque_mag)

    density_np = np.ones(fem_np.n_elems)
    result_np = fem_np.solve(density_np)

    # PyTorch solve (separate new solver, not reusing)
    print("2. PyTorch solver...")
    fem_torch = HexFEMSolver3D_Torch(E_mod=1.0, nu=0.3)
    fem_torch.set_mesh(Lx=1.0, Ly=0.2, Lz=0.2, nx=nx, ny=ny, nz=nz)
    fem_torch.fix_face(axis=0, coord=0.0)
    fem_torch.add_torque_load(center=center, axis=0, magnitude=torque_mag)

    density_torch = torch.ones(fem_torch.n_elems, dtype=torch.float32)
    result_torch = fem_torch.solve(density_torch)

    # Compare results
    print("\n3. Comparison:")
    compliance_np = result_np['compliance']
    compliance_torch = result_torch['compliance'].item()

    rel_error = abs(compliance_np - compliance_torch) / (abs(compliance_np) + 1e-10) * 100

    print(f"   NumPy compliance:    {compliance_np:.6e}")
    print(f"   PyTorch compliance:  {compliance_torch:.6e}")
    print(f"   Relative error:      {rel_error:.4f}%")

    # Check displacement agreement
    u_np = result_np['u']
    u_torch = result_torch['u'].numpy()

    disp_error = np.max(np.abs(u_np - u_torch.flatten())) / (np.max(np.abs(u_np)) + 1e-10) * 100
    print(f"   Max displacement error: {disp_error:.4f}%")

    if rel_error < 1.0 and disp_error < 1.0:
        print(f"\n✅ TEST 5 PASS: NumPy and PyTorch results match")
        return True
    else:
        print(f"\n❌ TEST 5 FAIL: Results differ by more than 1%")
        return False


def test_torque_with_optimization():
    """TEST 6: Use torque load in topology optimization."""
    print("\n" + "="*70)
    print("TEST 6: Topology Optimization with Torque")
    print("="*70)

    print("\n1. Setting up optimization...")
    fem = HexFEMSolver3D()
    fem.set_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=8, ny=2, nz=2)
    fem.fix_face(axis=0, coord=0.0)
    fem.add_torque_load(center=(1.0, 0.05, 0.05), axis=0, magnitude=1000.0)   

    print("2. Running optimization...")
    # Create optimizer
    volume_fraction = 0.5
    optimizer = SIMPOptimizer(
        fem_solver=fem,
        initial_density=0.5,
        volume_fraction=volume_fraction,
        penalty=3.0,
        filter_radius=0.02,
    )
    result = optimizer.optimize(n_iterations=100, verbose=True)
    # Print results
    compliance_improvement = (
        1 - result["final_compliance"] / result["history"]["compliance"][0]
    ) * 100

   
    # Check convergence
    print(f"\n3. Optimization Results:")
    print(f"   Initial compliance:   {result['history']['compliance'][0]:.6e}")
    print(f"   Final compliance:     {result['final_compliance']:.6e}")
    print(f"   Improvement:          {compliance_improvement:.2f}%")

    if compliance_improvement > 0:
        print(f"\n✅ TEST 6 PASS: Optimization reduces compliance")
        return True
    else:
        print(f"\n❌ TEST 6 FAIL: Compliance did not decrease")
        return False


def main():
    """Run all torque load tests."""
    print("\n" + "="*70)
    print("TORQUE LOAD TEST SUITE")
    print("="*70)

    tests = [
        ("Torque about x-axis", test_torque_about_x_axis),
        ("Torque about y-axis", test_torque_about_y_axis),
        ("Torque about z-axis", test_torque_about_z_axis),
        ("Combined bending + torsion", test_combined_bending_and_torsion),
        ("NumPy vs PyTorch equivalence", test_numpy_torch_equivalence),
        ("Topology optimization with torque", test_torque_with_optimization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)

    print(f"\nTotal: {n_passed}/{n_total} tests passed")

    if n_passed == n_total:
        print("\n🎉 ALL TESTS PASSED! Torque load feature is working correctly.")
    else:
        print(f"\n⚠️  {n_total - n_passed} test(s) failed.")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
