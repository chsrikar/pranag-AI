import deepxde as dde
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework

class HeatModel(BasePINNFramework):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

    def setup_model(self):
        self.setup_model_with_anchors()

    def setup_model_with_anchors(self, X_observed=None, y_observed=None):
        # Define Domain
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.define_domain(geom, timedomain)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        # PDE: du/dt - alpha * d^2u/dx^2 = 0
        def pde(x, u):
            du_dt = dde.grad.jacobian(u, x, i=0, j=1)
            d2u_dx2 = dde.grad.hessian(u, x, i=0, j=0)
            return du_dt - self.alpha * d2u_dx2

        # Boundary Conditions and Initial Condition
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

        bcs = [bc, ic]
        if X_observed is not None and y_observed is not None:
            observe_y = dde.icbc.PointSetBC(X_observed, y_observed, component=0)
            bcs.append(observe_y)

        # Build Network
        self.build_network(inputs=2, outputs=1)
        
        # Define DeepXDE Data and Model
        data = dde.data.TimePDE(
            geomtime,
            pde,
            bcs,
            num_domain=2000,
            num_boundary=200,
            num_initial=100,
            anchors=X_observed if X_observed is not None else None
        )
        self.model = dde.Model(data, self.net)

if __name__ == "__main__":
    hm = HeatModel()
    hm.setup_model()
    hm.compile(optimizer="adam", learning_rate=1e-3)
    hm.train(iterations=10)
