import deepxde as dde
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework

class BiologyModel(BasePINNFramework):
    def __init__(self, T=10.0):
        super().__init__()
        self.T = T

    def setup_model(self):
        # Inputs: t, lam, N0
        geom = dde.geometry.Hypercube([0, 0.1, 10.0], [self.T, 1.0, 1000.0])
        self.define_domain(geom)

        def ode(inputs, outputs):
            lam = inputs[:, 1:2]
            N = outputs[:, 0:1]
            
            dN_dt = dde.grad.jacobian(outputs, inputs, i=0, j=0)
            return dN_dt + lam * N

        def boundary_l(inputs, on_boundary):
            return on_boundary and dde.utils.isclose(inputs[0], 0)

        def func_ic(inputs):
            return inputs[:, 2:3]

        ic = dde.icbc.DirichletBC(geom, func_ic, boundary_l)

        self.build_network(inputs=3, outputs=1)
        
        data = dde.data.PDE(
            geom,
            ode,
            [ic],
            num_domain=250,
            num_boundary=32,
        )
        self.model = dde.Model(data, self.net)

if __name__ == "__main__":
    bm = BiologyModel()
    bm.setup_model()
    bm.compile()
    bm.train(10)
