import torch
import torch.nn as nn

from fem3d_torch import HexFEMSolver3D_Torch
from neural_network_topology_optimization import NNOptimizer


class SimpleDensityNetwork(nn.Module):
    def __init__(self, n_elems, device):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_elems, device=device))

    def forward(self):
        return torch.sigmoid(self.logits)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    fem = HexFEMSolver3D_Torch(E_mod=1.0, nu=0.3, device=device)
    fem.set_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=4, ny=2, nz=2)
    fem.fix_face(axis=0, coord=0.0)
    fem.add_torque_load(center=(1.0, 0.05, 0.05), axis=0, magnitude=1000.0)

    model = SimpleDensityNetwork(fem.n_elems, device=device)
    optimizer = NNOptimizer(fem_solver=fem, structure_model=model, volume_fraction=0.5, penalty=3.0)

    print("\nRunning NN-based topology optimization test...")
    result = optimizer.optimize(n_iterations=20, verbose=True)

    print("\nNN optimization test finished.")
    print("Initial compliance:", result["history"]["compliance"][0])
    print("Final compliance:", result["final_compliance"])
    print("Final volume:", result["final_volume"])
    print("Number of elements:", fem.n_elems)
    print("\nNNOptimizer integration test completed successfully.")


if __name__ == "__main__":
    main()
