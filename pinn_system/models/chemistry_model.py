"""
Module: chemistry_model.py
Purpose: PINN for Arrhenius kinetics surrogate.
         Residual: k - A*exp(-Ea/(R*T)) = 0
         Data : universal_index.parquet
         Cols : inputs=[temperature_max(norm), strength(norm)], target=conductivity(norm)
         All variables normalised to [0, 1]:
           T_norm  = temperature_max  (proxy for temperature T)
           Ea_norm = strength         (proxy for activation energy Ea)
           k_norm  = conductivity     (proxy for reaction rate k)
         In normalised space the residual becomes: k - (1-Ea)*(T+ε)
"""
import deepxde as dde
import deepxde.backend as bkd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pinn_system.base_pinn_framework import BasePINNFramework


class ChemistryModel(BasePINNFramework):
    """
    Arrhenius-kinetics PINN trained on conductivity from universal_index.parquet.

    In the normalised [0,1] domain:
        x[:, 0] ← temperature_max (norm) — T proxy (higher T → faster reaction)
        x[:, 1] ← strength        (norm) — Ea proxy (higher strength → higher barrier)
    Output:
        k        ← conductivity (norm) — reaction-rate / conductivity proxy

    Arrhenius law in normalised form:
        k ≈ (1 - Ea_norm) * T_norm
        Residual: k_pred - (1 - Ea_norm) * T_norm = 0
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

        # ── Physics residual (normalised Arrhenius) ───────────────────
        def residual(x, u):
            T_norm  = x[:, 0:1]   # temperature_max proxy
            Ea_norm = x[:, 1:2]   # strength proxy (activation energy)
            k       = u[:, 0:1]   # conductivity (predicted rate)
            # Higher T → faster; higher Ea → slower
            expected_k = (1.0 - Ea_norm) * (T_norm + 1e-6)
            return k - expected_k

        bcs = []
        if X_observed is not None and y_observed is not None:
            observe_y = dde.icbc.PointSetBC(X_observed, y_observed, component=0)
            bcs.append(observe_y)

        # ── Network & model ───────────────────────────────────────────
        self.build_network(inputs=2, outputs=1)
        data = dde.data.PDE(
            geom,
            residual,
            bcs,
            num_domain=1000,
            num_boundary=0 if not bcs else 100,
            anchors=X_observed if X_observed is not None else None,
        )
        self.model = dde.Model(data, self.net)


if __name__ == "__main__":
    from pinn_system.data_loader import load_data

    X, y = load_data(model_name="chemistry")
    print(f"ChemistryModel smoke test — X={X.shape}, y={y.shape}")
    cm = ChemistryModel()
    cm.setup_model_with_anchors(X, y)
    cm.compile()
    cm.train(10)
    print("ChemistryModel OK.")
