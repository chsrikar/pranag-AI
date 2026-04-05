import deepxde as dde
import deepxde.backend as bkd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework

class ChemistryModel(BasePINNFramework):
    def __init__(self):
        super().__init__()
        self.R = 8.314

    def setup_model(self):
        # Inputs: T, Ea, A
        geom = dde.geometry.Hypercube([273.0, 30000.0, 1.0e10], [500.0, 80000.0, 1.0e15])
        self.define_domain(geom)

        def residual(inputs, outputs):
            T = inputs[:, 0:1]
            Ea = inputs[:, 1:2]
            A = inputs[:, 2:3]
            k = outputs[:, 0:1]
            
            expected_k = A * bkd.exp(-Ea / (self.R * T))
            return k - expected_k

        self.build_network(inputs=3, outputs=1)
        
        data = dde.data.PDE(
            geom,
            residual,
            [],
            num_domain=250,
        )
        self.model = dde.Model(data, self.net)

if __name__ == "__main__":
    cm = ChemistryModel()
    cm.setup_model()
    cm.compile()
    cm.train(10)
