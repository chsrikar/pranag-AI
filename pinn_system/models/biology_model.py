"""
Module: biology_model.py
Purpose: PINN for first-order decay ODE surrogate.
         ODE  : dN/dt + λ*N = 0
         Data : universal_index.parquet
         Cols : inputs=[temperature_max(norm), salinity(norm)], target=ph(norm)
         All variables normalised to [0, 1]; t-axis = temperature_max, λ-proxy = salinity.
"""
import deepxde as dde
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework


class BiologyModel(BasePINNFramework):
    """
    Decay-ODE PINN trained on ph from universal_index.parquet.

    In the normalised [0,1] domain:
        x[:, 0] ← temperature_max (norm) — pseudo time axis (t)
        x[:, 1] ← salinity        (norm) — decay-rate proxy (λ)
    Output:
        N        ← ph (norm)
    Physics: dN/dt + λ*N = 0   →  residual = dN/dt + λ*N
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    def setup_model(self):
        self.setup_model_with_anchors()

    def setup_model_with_anchors(self, X_observed=None, y_observed=None):
        # ── Domain: [0,1]² ────────────────────────────────────────────
        geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
        self.define_domain(geom)

        # ── ODE residual: dN/dt + λ*N = 0 ────────────────────────────
        def ode(x, u):
            lam   = x[:, 1:2]       # salinity proxy → decay rate
            N     = u[:, 0:1]
            dN_dt = dde.grad.jacobian(u, x, i=0, j=0)
            return dN_dt + lam * N

        # ── IC: N(t=0) = 1.0 (fully undecayed in normalised space) ───
        def on_t0(x, on_boundary):
            return on_boundary and dde.utils.isclose(x[0], 0.0)

        ic = dde.icbc.DirichletBC(geom, lambda x: 1.0, on_t0)

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

    X, y = load_data(model_name="biology")
    print(f"BiologyModel smoke test — X={X.shape}, y={y.shape}")
    bm = BiologyModel()
    bm.setup_model_with_anchors(X, y)
    bm.compile()
    bm.train(10)
    print("BiologyModel OK.")
