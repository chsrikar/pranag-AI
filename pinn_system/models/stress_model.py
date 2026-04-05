import deepxde as dde
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework

class StressModel(BasePINNFramework):
    def __init__(self, L=1.0):
        super().__init__()
        self.L = L

    def setup_model(self):
        # Inputs: x (position), E (modulus), ε (strain) -> Output: σ (stress)
        geom = dde.geometry.Hypercube([0.0, 1e9, 0.0], [self.L, 2e11, 0.1])
        self.define_domain(geom)

        def loss_residual(inputs, outputs):
            E = inputs[:, 1:2]
            eps = inputs[:, 2:3]
            sigma = outputs[:, 0:1]
            return sigma - E * eps

        self.build_network(inputs=3, outputs=1)
        
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0))
        
        data = dde.data.PDE(
            geom,
            loss_residual,
            [bc],
            num_domain=250,
            num_boundary=32,
        )
        self.model = dde.Model(data, self.net)

if __name__ == "__main__":
    sm = StressModel()
    sm.setup_model()
    sm.compile()
    sm.train(10)
