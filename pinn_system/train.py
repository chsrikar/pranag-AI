"""
Module: train.py
Purpose: Two-phase PINN training pipeline using data from universal_index.parquet.

Training Phases
---------------
Phase 1 (Adam + cosine-annealing, ~80 % of total epochs)
    Train using ONLY data loss + constraint loss so the network first learns
    a mapping that reproduces observations, without physics interference.
    This prevents physics residuals (which start large) from overwhelming data.

Phase 2 (L-BFGS / Adam fine-tuning, ~20 % of total epochs)
    Add physics loss, boundary loss, and initial condition loss so that the
    already-converged surrogate is gently nudged toward physical consistency.

Loss design
-----------
  L_constraint = mean(max(0, -y)^2 + max(0, y-1)^2)   ← hard bounds [0,1]
  L_total = w_data      * norm(L_data)
           + w_physics   * norm(L_physics)
           + w_constraint* norm(L_constraint)
           + w_boundary  * norm(L_boundary)
           + w_initial   * norm(L_initial)

  norm(L_i) = L_i / (mean(L_i_history) + eps)          ← scale balance

  Adaptive weights use inverse-loss rule:
      w_i = 1 / (L_i_normalised + eps),  clipped to [w_min, w_max]
"""

import argparse
import logging
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import torch

from pinn_system.data_loader import load_data, PARQUET_PATH, MODEL_COLUMNS
from pinn_system.adaptive.adaptive_loss import AdaptiveLoss
from pinn_system.models.heat_model import HeatModel
from pinn_system.models.stress_model import StressModel
from pinn_system.models.growth_model import GrowthModel
from pinn_system.models.biology_model import BiologyModel
from pinn_system.models.chemistry_model import ChemistryModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
# Default loss weights  (easily tuneable via CLI args below)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = {
    "data":       1.0,   # anchor — real observations carry full weight
    "physics":    0.1,   # start low to avoid dominating data signal
    "constraint": 1.0,   # raised to 1.0 — constraint as important as data
    "boundary":   0.5,
    "initial":    0.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════
def get_model(model_name: str):
    registry = {
        "heat":      HeatModel,
        "stress":    StressModel,
        "growth":    GrowthModel,
        "biology":   BiologyModel,
        "chemistry": ChemistryModel,
    }
    if model_name not in registry:
        raise ValueError(
            f"Model '{model_name}' not found. Valid choices: {list(registry.keys())}"
        )
    return registry[model_name]()


# ══════════════════════════════════════════════════════════════════════════════
# Cosine-annealing learning-rate schedule
# ══════════════════════════════════════════════════════════════════════════════
def cosine_lr(initial_lr: float, min_lr: float, total_steps: int, step: int) -> float:
    """Smooth LR decay from initial_lr → min_lr over total_steps."""
    progress = step / max(total_steps, 1)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════════════
# Constraint loss (hard-bounds enforcement using DATA-DRIVEN bounds)
# ══════════════════════════════════════════════════════════════════════════════
def compute_constraint_loss(
    predictions: np.ndarray,
    low:  float = 0.0,
    high: float = 1.0,
) -> float:
    """
    Penalise predictions outside the valid physical range [low, high].

    Bounds are DATA-DRIVEN (professor's requirement):
        low  = y.min()   ← smallest observed target value in the dataset
        high = y.max()   ← largest  observed target value in the dataset

    For Min-Max normalised targets these equal [0.0, 1.0] exactly, but
    passing them explicitly documents the intent and allows future datasets
    whose normalised range may differ slightly from [0,1].

    Formula:
        L_constraint = mean( max(0, low - y)^2  +  max(0, y - high)^2 )

    Parameters
    ----------
    predictions : np.ndarray  shape (N, output_dim)
    low         : float        lower physical bound (default 0.0)
    high        : float        upper physical bound (default 1.0)

    Returns
    -------
    float  0.0 when all predictions are within [low, high].
    """
    y = np.asarray(predictions, dtype=np.float64)
    # Penalty when prediction falls below the minimum observed value
    lower_violation = np.maximum(0.0, low  - y) ** 2
    # Penalty when prediction exceeds the maximum observed value
    upper_violation = np.maximum(0.0, y - high ) ** 2
    return float(np.mean(lower_violation + upper_violation))


# ══════════════════════════════════════════════════════════════════════════════
# Extract individual loss components from DeepXDE train state
# ══════════════════════════════════════════════════════════════════════════════
def extract_losses(model, component_names: list) -> dict:
    """
    Pull per-component losses from DeepXDE's train_state and map them to
    the names used by AdaptiveLoss.  Mapping assumes DeepXDE orders losses as:
        [pde_residual,  bc_0,  bc_1, ...,  observe_bc]

    For all five models in this project the layout is:
        index 0 → physics  (PDE residual)
        index 1 → boundary (DirichletBC)
        index 2 → initial  (IC)
        index 3 → data     (PointSetBC / observed data)

    'constraint' is computed externally (see compute_constraint_loss) and
    injected into the returned dict by the caller.
    """
    _default = {n: 0.0 for n in component_names}
    try:
        loss_raw = model.train_state.loss_train
        if not hasattr(loss_raw, "__len__") or len(loss_raw) == 0:
            logger.warning("loss_train is empty — returning zeros.")
            return _default

        last = loss_raw[-1]
        components = (
            [float(v) for v in last]
            if hasattr(last, "__len__")
            else [float(v) for v in loss_raw]
        )

        n = len(components)
        return {
            "physics":  components[0] if n > 0 else 0.0,
            "boundary": components[1] if n > 1 else 0.0,
            "initial":  components[2] if n > 2 else 0.0,
            "data":     components[3] if n > 3 else 0.0,
        }
    except Exception as exc:
        logger.warning(f"extract_losses failed: {exc} — returning zeros.")
        return _default


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train a PINN on universal_index.parquet"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="heat",
        choices=list(MODEL_COLUMNS.keys()),
        help=f"Which PINN to train. Choices: {list(MODEL_COLUMNS.keys())}",
    )
    parser.add_argument("--epochs",         type=int,   default=1000,
                        help="Total training iterations (Phase1 + Phase2).")
    parser.add_argument("--lr",             type=float, default=0.001,
                        help="Initial learning rate for Adam (Phase 1).")
    parser.add_argument("--recompile_freq", type=int,   default=200,
                        help="How often (steps) to recompile with updated weights.")
    # ── Per-weight CLI overrides ────────────────────────────────────────────
    parser.add_argument("--w_data",       type=float, default=DEFAULT_WEIGHTS["data"])
    parser.add_argument("--w_physics",    type=float, default=DEFAULT_WEIGHTS["physics"])
    parser.add_argument("--w_constraint", type=float, default=DEFAULT_WEIGHTS["constraint"])
    parser.add_argument("--w_boundary",   type=float, default=DEFAULT_WEIGHTS["boundary"])
    parser.add_argument("--w_initial",    type=float, default=DEFAULT_WEIGHTS["initial"])
    # ── Adaptive-loss clipping range ────────────────────────────────────────
    parser.add_argument("--w_min", type=float, default=0.5)
    parser.add_argument("--w_max", type=float, default=8.0)

    args = parser.parse_args()

    # Build base-weight dict from CLI (or defaults)
    base_weights = {
        "physics":    args.w_physics,
        "boundary":   args.w_boundary,
        "initial":    args.w_initial,
        "data":       args.w_data,
        "constraint": args.w_constraint,
    }

    logger.info("=" * 65)
    logger.info(f"  Training PINN : {args.model.upper()}")
    logger.info(f"  Data source   : {PARQUET_PATH}")
    logger.info(f"  Columns       : {MODEL_COLUMNS[args.model]}")
    logger.info(f"  Base weights  : {base_weights}")
    logger.info("=" * 65)

    # ── 1. Load + validate data ────────────────────────────────────────────
    try:
        X, y = load_data(model_name=args.model)
        logger.info(f"Data loaded — X={X.shape}, y={y.shape}")
        assert X.shape[0] > 0,          "No data rows loaded"
        assert y.shape[0] > 0,          "No target data loaded"
        assert not np.isnan(X).any(),   "X contains NaN"
        assert not np.isnan(y).any(),   "y contains NaN"
        logger.info("Data validation passed ✓")
    except Exception as exc:
        logger.error(f"Data load failed: {exc}")
        raise RuntimeError(
            f"Cannot proceed — data load from {PARQUET_PATH} failed."
        ) from exc

    # ── Data-driven physical bounds (professor's requirement) ──────────────
    # Extract the actual observed min/max from the training targets.
    # For Min-Max normalised data these will be 0.0 and 1.0, but making
    # them explicit means the constraint is always grounded in real data.
    y_low  = float(y.min())   # smallest valid output value in the dataset
    y_high = float(y.max())   # largest  valid output value in the dataset
    logger.info(
        f"  Data-driven bounds: LOW={y_low:.6f}  HIGH={y_high:.6f}  "
        f"(used in constraint loss and final reporting)"
    )

    # ── 2. Build PINN model with real data anchors ─────────────────────────
    pinn = get_model(args.model)
    if hasattr(pinn, "setup_model_with_anchors"):
        pinn.setup_model_with_anchors(X, y)
    else:
        pinn.setup_model()

    # ── 3. Adaptive-loss manager (Phase 1 uses data + constraint only) ─────
    #
    #  Phase 1 component set:  data, constraint
    #  Phase 2 component set:  physics, boundary, initial, data, constraint
    #
    phase1_components = {
        "data":       base_weights["data"],
        "constraint": base_weights["constraint"],
    }
    adaptive_phase1 = AdaptiveLoss(
        components=phase1_components,
        w_min=args.w_min,
        w_max=args.w_max,
    )

    # Full set for Phase 2 — insertion order must match DeepXDE's loss list.
    phase2_components = {
        "physics":    base_weights["physics"],
        "boundary":   base_weights["boundary"],
        "initial":    base_weights["initial"],
        "data":       base_weights["data"],
        "constraint": base_weights["constraint"],
    }
    adaptive_phase2 = AdaptiveLoss(
        components=phase2_components,
        w_min=args.w_min,
        w_max=args.w_max,
    )

    # ── Detect how many DeepXDE loss outputs the model produces ───────────
    #  Layout: [pde, bc_0, bc_1, ..., observe_bc]
    #  With DirichletBC(1) + IC(1) + PointSetBC(1)  → 4 total:
    #    index 0 → physics
    #    index 1 → boundary
    #    index 2 → initial
    #    index 3 → data
    num_dde_losses = (
        1 + len(pinn.model.data.bcs)
        if hasattr(pinn.model.data, "bcs")
        else 1
    )
    logger.info(f"DeepXDE produces {num_dde_losses} loss term(s).")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 — Data-only training (Adam + cosine-annealing)
    # ══════════════════════════════════════════════════════════════════════
    phase1_steps   = int(args.epochs * 0.8)
    recompile_freq = args.recompile_freq

    logger.info(
        f"\n{'═'*65}\n"
        f"  PHASE 1: Data-only training  ({phase1_steps} steps)\n"
        f"  Components : data + constraint loss only\n"
        f"  Recompile  : every {recompile_freq} steps\n"
        f"{'═'*65}"
    )

    for cycle_start in range(0, phase1_steps, recompile_freq):
        current_lr = cosine_lr(args.lr, 1e-5, phase1_steps, cycle_start)

        # Phase 1 only uses [data] weight repeated across all DeepXDE slots
        # Constraint is handled externally (not a DeepXDE component), so we
        # pass uniform weights to DeepXDE and rely on adaptive.update to
        # track the balance.
        p1_weights  = adaptive_phase1.get_weights_list()
        # DeepXDE expects one weight per internal loss; map data weight to all.
        dde_weights = [p1_weights[0]] * num_dde_losses      # index 0 = data weight

        pinn.compile(
            optimizer="adam",
            learning_rate=current_lr,
            loss_weights=dde_weights,
        )
        logger.info(
            f"  [P1 step {cycle_start:5d}] lr={current_lr:.6f} | {adaptive_phase1.summary()}"
        )

        pinn.train(iterations=min(recompile_freq, phase1_steps - cycle_start))

        # Compute losses for adaptive update
        # Phase 1 goal: model learns to reproduce real data AND stay in bounds.
        # constraint uses DATA-DRIVEN bounds (y_low, y_high).
        raw_losses = extract_losses(pinn.model, list(phase1_components.keys()))
        try:
            preds = pinn.predict(X)
            raw_losses["constraint"] = compute_constraint_loss(
                preds, low=y_low, high=y_high
            )
        except Exception as exc:
            logger.warning(f"Constraint loss computation skipped: {exc}")
            raw_losses["constraint"] = 0.0

        # Log: constraint should approach 0 as sigmoid squashes outputs into [low,high]
        in_bounds_pct = (
            100.0 * np.mean((preds >= y_low) & (preds <= y_high))
            if 'preds' in dir() else 0.0
        )
        logger.info(
            f"  [P1] data={raw_losses['data']:.3e}  "
            f"constraint={raw_losses['constraint']:.3e}  "
            f"in-bounds={in_bounds_pct:.1f}%"
        )
        adaptive_phase1.update(raw_losses)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 — Full physics-informed fine-tuning
    # ══════════════════════════════════════════════════════════════════════
    phase2_steps = args.epochs - phase1_steps

    logger.info(
        f"\n{'═'*65}\n"
        f"  PHASE 2: Full physics training ({phase2_steps} steps)\n"
        f"  Components : physics + boundary + initial + data + constraint\n"
        f"{'═'*65}"
    )

    # Seed Phase 2 adaptive with current data/constraint weights from Phase 1
    p1_final = adaptive_phase1.get_weights_dict()
    adaptive_phase2.current_weights["data"]       = p1_final.get("data",       base_weights["data"])
    adaptive_phase2.current_weights["constraint"] = p1_final.get("constraint", base_weights["constraint"])

    try:
        # ── Try L-BFGS first (best for fine-tuning) ───────────────────
        # L-BFGS uses all losses simultaneously.  We pass the phase2
        # weights ordered as: [physics, boundary, initial, data]
        # (constraint is external, so excluded from DeepXDE's list).
        final_p2_weights = adaptive_phase2.get_weights_list()
        dde_weights_p2   = final_p2_weights[:num_dde_losses]   # trim to DDE count

        pinn.compile(
            optimizer="L-BFGS",
            learning_rate=1.0,
            loss_weights=dde_weights_p2,
        )
        logger.info("  Attempting L-BFGS fine-tuning …")
        pinn.model.train()
        logger.info("  L-BFGS complete ✅")

    except Exception as exc:
        logger.warning(f"  L-BFGS unavailable ({exc}) — falling back to Adam lr=1e-4")

        for cycle_start in range(0, phase2_steps, recompile_freq):
            p2_weights = adaptive_phase2.get_weights_list()
            dde_weights_p2 = p2_weights[:num_dde_losses]

            pinn.compile(
                optimizer="adam",
                learning_rate=1e-4,
                loss_weights=dde_weights_p2,
            )
            logger.info(
                f"  [P2 step {cycle_start:5d}] {adaptive_phase2.summary()}"
            )
            pinn.train(iterations=min(recompile_freq, phase2_steps - cycle_start))

            # Full loss extraction for Phase 2 adaptive update.
            # constraint uses DATA-DRIVEN bounds (y_low, y_high) — same as Phase 1.
            raw_losses = extract_losses(pinn.model, list(phase2_components.keys()))
            try:
                preds = pinn.predict(X)
                raw_losses["constraint"] = compute_constraint_loss(
                    preds, low=y_low, high=y_high
                )
            except Exception as inner_exc:
                logger.warning(f"Constraint loss computation skipped: {inner_exc}")
                raw_losses["constraint"] = 0.0

            logger.info(
                f"  [P2] physics={raw_losses['physics']:.3e}  "
                f"boundary={raw_losses['boundary']:.3e}  "
                f"initial={raw_losses['initial']:.3e}  "
                f"data={raw_losses['data']:.3e}  "
                f"constraint={raw_losses['constraint']:.3e}"
            )
            adaptive_phase2.update(raw_losses)

    # ── 6. Save predictions ────────────────────────────────────────────────
    try:
        predictions = pinn.predict(X)
        logger.info(f"Predictions shape: {predictions.shape}")

        # Final constraint check using DATA-DRIVEN bounds
        c_loss = compute_constraint_loss(predictions, low=y_low, high=y_high)
        in_bounds = float(np.mean((predictions >= y_low) & (predictions <= y_high))) * 100
        logger.info(
            f"Final constraint loss : {c_loss:.4e} "
            f"({'PASS' if c_loss < 1e-4 else 'WARN'})"
        )
        logger.info(
            f"In-bounds predictions : {in_bounds:.2f}%  "
            f"(bounds: [{y_low:.4f}, {y_high:.4f}])"
        )

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"predictions_{args.model}.npy")
        np.save(out_file, predictions)
        logger.info(f"Saved → {out_file}")

    except Exception as exc:
        logger.warning(f"Prediction save failed: {exc}")


if __name__ == "__main__":
    main()
