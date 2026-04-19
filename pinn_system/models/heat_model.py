"""
Module: heat_model.py
Purpose: PINN for heat diffusion surrogate.
         PDE  : du/dt - alpha*d²u/dx² = 0
         Data : universal_index.parquet
         Cols : inputs=[strength(norm), conductivity(norm)], target=temperature_max(norm)
         Domain is [0,1]² (normalised x-like and time-like axes).
"""
import deepxde as dde
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework


class HeatModel(BasePINNFramework):
    """
    Heat-diffusion PINN trained on temperature_max from universal_index.parquet.

    Input axes (normalised to [0,1]):
        x[:, 0] ← strength      (acts as spatial coordinate proxy)
        x[:, 1] ← conductivity  (acts as time-like coordinate proxy)
    Output:
        u        ← temperature_max  (normalised)
    """

    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    # ------------------------------------------------------------------
    def setup_model(self):
        self.setup_model_with_anchors()

    def setup_model_with_anchors(self, X_observed=None, y_observed=None):
        # ── Domain: both axes in [0, 1] due to Min-Max normalisation ──
        geom       = dde.geometry.Interval(0.0, 1.0)
        timedomain = dde.geometry.TimeDomain(0.0, 1.0)
        self.define_domain(geom, timedomain)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        # ── PDE: du/dt - alpha * d²u/dx² = 0 ─────────────────────────
        def pde(x, u):
            du_dt    = dde.grad.jacobian(u, x, i=0, j=1)
            d2u_dx2  = dde.grad.hessian(u, x, i=0, j=0)
            return du_dt - self.alpha * d2u_dx2

        # ── BCs / ICs ─────────────────────────────────────────────────
        bc = dde.icbc.DirichletBC(
            geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary
        )
        ic = dde.icbc.IC(
            geomtime,
            lambda x: np.sin(np.pi * x[:, 0:1]),
            lambda _, on_initial: on_initial,
        )

        bcs = [bc, ic]
        if X_observed is not None and y_observed is not None:
            observe_y = dde.icbc.PointSetBC(X_observed, y_observed, component=0)
            bcs.append(observe_y)

        # ── Network & model ───────────────────────────────────────────
        self.build_network(inputs=2, outputs=1)
        data = dde.data.TimePDE(
            geomtime,
            pde,
            bcs,
            num_domain=2000,
            num_boundary=200,
            num_initial=100,
            anchors=X_observed if X_observed is not None else None,
        )
        self.model = dde.Model(data, self.net)


if __name__ == "__main__":
    from pinn_system.data_loader import load_data

    X, y = load_data(model_name="heat")
    print(f"HeatModel smoke test — X={X.shape}, y={y.shape}")
    hm = HeatModel()
    hm.setup_model_with_anchors(X, y)
    hm.compile(optimizer="adam", learning_rate=1e-3)
    hm.train(iterations=10)
    print("HeatModel OK.")
