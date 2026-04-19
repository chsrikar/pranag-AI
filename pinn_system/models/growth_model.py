"""
Module: growth_model.py
Purpose: PINN for logistic-growth surrogate.
         ODE  : dP/dt - r*P*(1 - P/K) = 0
         Data : universal_index.parquet
         Cols : inputs=[temperature_max(norm), ph(norm)], target=salinity(norm)
         All variables normalised to [0, 1]; t-axis = temperature_max, P-axis = salinity.
"""
import deepxde as dde
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework


class GrowthModel(BasePINNFramework):
    """
    Logistic-growth PINN trained on salinity from universal_index.parquet.

    In the normalised [0,1] domain:
        x[:, 0] ← temperature_max (norm) — pseudo time axis (t)
        x[:, 1] ← ph              (norm) — growth-rate proxy (r)
    Output:
        P        ← salinity (norm)
    Physics: dP/dt = r * P * (1 - P)   [K=1 in normalised space]
    """

    def __init__(self, P0: float = 0.1):
        """
        Args:
            P0: Initial condition value in normalised space (default 0.1).
        """
        super().__init__()
        self.P0 = P0

    # ------------------------------------------------------------------
    def setup_model(self):
        self.setup_model_with_anchors()

    def setup_model_with_anchors(self, X_observed=None, y_observed=None):
        # ── Domain: [0,1]² ────────────────────────────────────────────
        geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
        self.define_domain(geom)

        # ── ODE residual: dP/dt - r*P*(1-P) = 0 ──────────────────────
        def ode(x, u):
            r    = x[:, 1:2]          # ph proxy → growth rate
            P    = u[:, 0:1]
            dP_dt = dde.grad.jacobian(u, x, i=0, j=0)
            return dP_dt - r * P * (1.0 - P)

        # ── IC: P(t=0) = P0 ───────────────────────────────────────────
        def on_t0(x, on_boundary):
            return on_boundary and dde.utils.isclose(x[0], 0.0)

        ic = dde.icbc.DirichletBC(geom, lambda x: self.P0, on_t0)

        bcs = [ic]
        if X_observed is not None and y_observed is not None:
            observe_y = dde.icbc.PointSetBC(X_observed, y_observed, component=0)
            bcs.append(observe_y)

        # ── Network & model ───────────────────────────────────────────
        self.build_network(inputs=2, outputs=1)
        data = dde.data.PDE(
            geom,
            ode,
            bcs,
            num_domain=1000,
            num_boundary=100,
            anchors=X_observed if X_observed is not None else None,
        )
        self.model = dde.Model(data, self.net)


if __name__ == "__main__":
    from pinn_system.data_loader import load_data

    X, y = load_data(model_name="growth")
    print(f"GrowthModel smoke test — X={X.shape}, y={y.shape}")
    gm = GrowthModel()
    gm.setup_model_with_anchors(X, y)
    gm.compile()
    gm.train(10)
    print("GrowthModel OK.")
