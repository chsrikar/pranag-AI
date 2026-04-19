"""
Module: stress_model.py
Purpose: PINN for mechanical stress surrogate (Hooke's law).
         PDE residual : sigma - E * epsilon = 0
         Data : universal_index.parquet
         Cols : inputs=[temperature_max(norm), conductivity(norm)], target=strength(norm)
         Domain is [0,1]² (normalised).
"""
import deepxde as dde
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework


class StressModel(BasePINNFramework):
    """
    Mechanical-stress PINN trained on strength from universal_index.parquet.

    In the normalised [0,1] domain:
        x[:, 0] ← temperature_max (normalised) — proxy for elastic modulus E
        x[:, 1] ← conductivity    (normalised) — proxy for strain ε
    Output:
        σ        ← strength (normalised) — mechanical stress
    Physics: σ = E · ε  →  residual = σ_pred - x[:,0]*x[:,1]
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    def setup_model(self):
        self.setup_model_with_anchors()

    def setup_model_with_anchors(self, X_observed=None, y_observed=None):
        # ── Domain: [0,1]² after Min-Max normalisation ────────────────
        geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
        self.define_domain(geom)

        # ── PDE residual: σ - E·ε = 0 ────────────────────────────────
        def pde(x, u):
            E   = x[:, 0:1]   # temperature_max proxy → modulus
            eps = x[:, 1:2]   # conductivity proxy    → strain
            sigma = u[:, 0:1]
            return sigma - E * eps

        # ── Boundary condition: zero stress at origin ─────────────────
        bc = dde.icbc.DirichletBC(
            geom,
            lambda x: 0.0,
            lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0.0),
        )

        bcs = [bc]
        if X_observed is not None and y_observed is not None:
            observe_y = dde.icbc.PointSetBC(X_observed, y_observed, component=0)
            bcs.append(observe_y)

        # ── Network & model ───────────────────────────────────────────
        self.build_network(inputs=2, outputs=1)
        data = dde.data.PDE(
            geom,
            pde,
            bcs,
            num_domain=1000,
            num_boundary=100,
            anchors=X_observed if X_observed is not None else None,
        )
        self.model = dde.Model(data, self.net)


if __name__ == "__main__":
    from pinn_system.data_loader import load_data

    X, y = load_data(model_name="stress")
    print(f"StressModel smoke test — X={X.shape}, y={y.shape}")
    sm = StressModel()
    sm.setup_model_with_anchors(X, y)
    sm.compile()
    sm.train(10)
    print("StressModel OK.")
