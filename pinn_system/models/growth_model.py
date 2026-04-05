import deepxde as dde
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework

class GrowthModel(BasePINNFramework):
    def __init__(self, T=10.0, P0=100.0):
        super().__init__()
        self.T = T
        self.P0 = P0

    def setup_model(self):
        # Inputs: t, r, K
        geom = dde.geometry.Hypercube([0, 0.1, 500], [self.T, 0.5, 2000])
        self.define_domain(geom)

        def ode(inputs, outputs):
            t = inputs[:, 0:1]
            r = inputs[:, 1:2]
            K = inputs[:, 2:3]
            P = outputs[:, 0:1]
            
            dP_dt = dde.grad.jacobian(outputs, inputs, i=0, j=0)
            return dP_dt - r * P * (1.0 - P / K)

        def boundary_l(inputs, on_boundary):
            return on_boundary and dde.utils.isclose(inputs[0], 0)

        ic = dde.icbc.IC(geom, lambda _: self.P0, boundary_l)

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
    gm = GrowthModel()
    gm.setup_model()
    gm.compile()
    gm.train(10)
