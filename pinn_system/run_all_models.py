"""
run_all_models.py
-----------------
Trains all five PINN models (heat, stress, growth, biology, chemistry)
and reports per-model accuracy metrics:
  - R2      (1.0 = perfect)
  - MAE     (lower = better)
  - RMSE    (lower = better)
  - MaxErr  (worst-case deviation)
  - InBounds% (predictions within [y_low, y_high])

Usage:
    python -m pinn_system.run_all_models --epochs 500 --recompile_freq 100
"""

import argparse
import logging
import math
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from pinn_system.data_loader import load_data, MODEL_COLUMNS
from pinn_system.adaptive.adaptive_loss import AdaptiveLoss
from pinn_system.models.heat_model      import HeatModel
from pinn_system.models.stress_model    import StressModel
from pinn_system.models.growth_model    import GrowthModel
from pinn_system.models.biology_model   import BiologyModel
from pinn_system.models.chemistry_model import ChemistryModel

# Suppress noisy DeepXDE / INFO logs during benchmark
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
dde.config.set_default_float("float32")

MODEL_REGISTRY = {
    "heat":      HeatModel,
    "stress":    StressModel,
    "growth":    GrowthModel,
    "biology":   BiologyModel,
    "chemistry": ChemistryModel,
}

DEFAULT_WEIGHTS = {
    "data":       1.0,
    "physics":    0.1,
    "constraint": 1.0,
    "boundary":   0.5,
    "initial":    0.5,
}


# ---------------------------------------------------------------------------
def cosine_lr(ini, lo, total, step):
    p = step / max(total, 1)
    return lo + 0.5 * (ini - lo) * (1 + math.cos(math.pi * p))


def constraint_loss(preds, low=0.0, high=1.0):
    y = np.asarray(preds, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, low - y)**2 + np.maximum(0.0, y - high)**2))


def extract_losses(model):
    try:
        lr = model.train_state.loss_train
        if not hasattr(lr, "__len__") or len(lr) == 0:
            return {"physics": 0, "boundary": 0, "initial": 0, "data": 0}
        c = [float(v) for v in lr[-1]] if hasattr(lr[-1], "__len__") else [float(v) for v in lr]
        return {"physics": c[0] if len(c)>0 else 0,
                "boundary":c[1] if len(c)>1 else 0,
                "initial": c[2] if len(c)>2 else 0,
                "data":    c[3] if len(c)>3 else 0}
    except Exception:
        return {"physics": 0, "boundary": 0, "initial": 0, "data": 0}


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def progress(label, step, total, extra=""):
    pct = min(100, int(step / max(total, 1) * 100))
    bar = "#" * (pct // 5) + "." * (20 - pct // 5)
    print(f"    {label} [{bar}] {pct:3d}%  {extra}", end="\r", flush=True)


# ---------------------------------------------------------------------------
def train_model(model_name, epochs, lr, recompile_freq, w_min, w_max):
    X, y = load_data(model_name=model_name)
    y_low  = float(y.min())
    y_high = float(y.max())
    print(f"  Rows: {X.shape[0]}  bounds: [{y_low:.4f}, {y_high:.4f}]")

    pinn = MODEL_REGISTRY[model_name]()
    if hasattr(pinn, "setup_model_with_anchors"):
        pinn.setup_model_with_anchors(X, y)
    else:
        pinn.setup_model()

    n_dde = (1 + len(pinn.model.data.bcs)
             if hasattr(pinn.model.data, "bcs") else 1)

    # --- Phase 1: data + constraint only ---
    p1_steps = int(epochs * 0.8)
    p2_steps = epochs - p1_steps
    adapt1   = AdaptiveLoss({"data": DEFAULT_WEIGHTS["data"],
                              "constraint": DEFAULT_WEIGHTS["constraint"]},
                             w_min=w_min, w_max=w_max)
    adapt2   = AdaptiveLoss({k: DEFAULT_WEIGHTS[k] for k in
                              ["physics","boundary","initial","data","constraint"]},
                             w_min=w_min, w_max=w_max)

    print(f"  Phase 1: {p1_steps} steps (data+constraint) ...")
    for step in range(0, p1_steps, recompile_freq):
        cur_lr = cosine_lr(lr, 1e-5, p1_steps, step)
        dde_w  = [adapt1.get_weights_list()[0]] * n_dde
        pinn.compile(optimizer="adam", learning_rate=cur_lr, loss_weights=dde_w)
        pinn.train(iterations=min(recompile_freq, p1_steps - step))
        raw = extract_losses(pinn.model)
        try:
            preds = pinn.predict(X)
            raw["constraint"] = constraint_loss(preds, y_low, y_high)
        except Exception:
            raw["constraint"] = 0.0
        adapt1.update(raw)
        progress("P1", step + recompile_freq, p1_steps,
                 f"data={raw['data']:.2e}  c={raw['constraint']:.2e}")
    print()

    # --- Phase 2: all losses ---
    print(f"  Phase 2: {p2_steps} steps (all losses) ...")
    p1f = adapt1.get_weights_dict()
    adapt2.current_weights["data"]       = p1f.get("data",       1.0)
    adapt2.current_weights["constraint"] = p1f.get("constraint", 1.0)

    # Phase 2 uses Adam only (bounded) for fast, predictable benchmark timing.
    # Use train.py for full L-BFGS fine-tuning on production runs.
    lbfgs_ok = False
    for step in range(0, p2_steps, recompile_freq):
        w2 = adapt2.get_weights_list()[:n_dde]
        pinn.compile(optimizer="adam", learning_rate=1e-4, loss_weights=w2)
        pinn.train(iterations=min(recompile_freq, p2_steps - step))
        raw = extract_losses(pinn.model)
        try:
            preds = pinn.predict(X)
            raw["constraint"] = constraint_loss(preds, y_low, y_high)
        except Exception:
            raw["constraint"] = 0.0
        adapt2.update(raw)
        progress("P2", step + recompile_freq, p2_steps,
                 f"phys={raw['physics']:.2e}  data={raw['data']:.2e}")
    print()

    # --- Metrics ---
    predictions = pinn.predict(X)
    yt = y.flatten()
    yp = predictions.flatten()
    return {
        "R2":         r2_score(yt, yp),
        "MAE":        float(np.mean(np.abs(yt - yp))),
        "RMSE":       float(np.sqrt(np.mean((yt - yp)**2))),
        "MaxErr":     float(np.max(np.abs(yt - yp))),
        "Constraint": constraint_loss(predictions, y_low, y_high),
        "InBounds%":  float(np.mean((yp >= y_low) & (yp <= y_high))) * 100.0,
        "Samples":    X.shape[0],
        "LBFGSused":  lbfgs_ok,
    }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run all PINNs and report accuracy")
    parser.add_argument("--epochs",         type=int,   default=500)
    parser.add_argument("--lr",             type=float, default=0.001)
    parser.add_argument("--recompile_freq", type=int,   default=100)
    parser.add_argument("--w_min",          type=float, default=0.5)
    parser.add_argument("--w_max",          type=float, default=8.0)
    parser.add_argument("--models",         type=str,   default="all")
    args = parser.parse_args()

    models_to_run = (list(MODEL_REGISTRY.keys())
                     if args.models == "all"
                     else [m.strip() for m in args.models.split(",")])

    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  PINN ACCURACY BENCHMARK")
    print(f"  Epochs={args.epochs}  LR={args.lr}  Recompile={args.recompile_freq}")
    print(f"  Models: {models_to_run}")
    print(SEP)

    results     = {}
    total_start = time.time()

    for mn in models_to_run:
        print(f"\n{'-'*72}")
        print(f"  >> MODEL: {mn.upper()}")
        print(f"{'-'*72}")
        t0 = time.time()
        try:
            m = train_model(mn, args.epochs, args.lr,
                            args.recompile_freq, args.w_min, args.w_max)
            m["time_s"] = time.time() - t0
            results[mn] = m
            print(f"  DONE in {m['time_s']:.1f}s  |  R2={m['R2']:.4f}  MAE={m['MAE']:.4f}")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results[mn] = {"error": str(exc)}

    # --- Summary table ---
    total_elapsed = time.time() - total_start
    print(f"\n\n{SEP}")
    print(f"  ACCURACY RESULTS   (total wall-time: {total_elapsed:.1f}s)")
    print(SEP)
    print(f"  {'Model':<12} {'R2':>7} {'MAE':>9} {'RMSE':>9} {'MaxErr':>9}"
          f" {'Constraint':>12} {'InBounds%':>10} {'N':>6}  Grade")
    print(f"  {'-'*68}")

    for mn in models_to_run:
        m = results.get(mn, {})
        if "error" in m:
            print(f"  {mn:<12}   ERROR: {m['error'][:40]}")
            continue
        grade = ("[EXCELLENT]" if m["R2"] >= 0.90 else
                 "[GOOD]"      if m["R2"] >= 0.75 else
                 "[FAIR]"      if m["R2"] >= 0.50 else
                 "[POOR]")
        print(f"  {mn:<12} {m['R2']:>7.4f} {m['MAE']:>9.4f} {m['RMSE']:>9.4f}"
              f" {m['MaxErr']:>9.4f} {m['Constraint']:>12.2e}"
              f" {m['InBounds%']:>9.1f}% {m['Samples']:>6d}  {grade}")

    print(SEP)
    print("  GRADE: R2>=0.90=EXCELLENT  R2>=0.75=GOOD  R2>=0.50=FAIR  R2<0.50=POOR")
    print(SEP)

    # Save results
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "accuracy_results.txt")
    with open(out_file, "w") as f:
        f.write(f"PINN Accuracy Benchmark  epochs={args.epochs} lr={args.lr}\n\n")
        f.write(f"{'Model':<12} {'R2':>8} {'MAE':>9} {'RMSE':>9} {'MaxErr':>9}"
                f" {'InBounds%':>10}  Grade\n")
        f.write("-" * 65 + "\n")
        for mn in models_to_run:
            m = results.get(mn, {})
            if "error" not in m:
                grade = ("EXCELLENT" if m["R2"]>=0.90 else
                         "GOOD"      if m["R2"]>=0.75 else
                         "FAIR"      if m["R2"]>=0.50 else "POOR")
                f.write(f"{mn:<12} {m['R2']:>8.4f} {m['MAE']:>9.4f} {m['RMSE']:>9.4f}"
                        f" {m['MaxErr']:>9.4f} {m['InBounds%']:>9.1f}%  {grade}\n")
    print(f"\n  Results saved -> {out_file}\n")


if __name__ == "__main__":
    main()
