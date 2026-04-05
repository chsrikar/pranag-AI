import logging
import argparse
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from pinn_system.data_loader import load_data
from pinn_system.adaptive.adaptive_loss import AdaptiveLoss
from pinn_system.models.heat_model import HeatModel
from pinn_system.models.stress_model import StressModel
from pinn_system.models.growth_model import GrowthModel
from pinn_system.models.biology_model import BiologyModel
from pinn_system.models.chemistry_model import ChemistryModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(model_name):
    models = {
        "heat": HeatModel,
        "stress": StressModel,
        "growth": GrowthModel,
        "biology": BiologyModel,
        "chemistry": ChemistryModel
    }
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found.")
    return models[model_name]()

def cosine_lr(initial_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    progress = current_step / max(total_steps, 1)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

def extract_losses(model) -> dict:
    try:
        loss_raw = model.train_state.loss_train
        if hasattr(loss_raw, '__len__') and len(loss_raw) > 0:
            last = loss_raw[-1]
            components = [float(v) for v in last] if hasattr(last, '__len__') else [float(v) for v in loss_raw]
        else:
            logger.warning('loss_train is scalar — returning zeros')
            return {'physics': 0.0, 'boundary': 0.0, 'initial': 0.0, 'data': 0.0}
        n = len(components)
        # DeepXDE's order is PDE first, then BCs in the order they were defined
        # For heat_model: bcs = [DirichletBC, IC, PointSetBC] => order: physics, boundary, initial, data
        return {
            'physics':  components[0] if n > 0 else 0.0,
            'boundary': components[1] if n > 1 else 0.0,
            'initial':  components[2] if n > 2 else 0.0,
            'data':     components[3] if n > 3 else 0.0,
        }
    except Exception as e:
        logger.warning(f'extract_losses failed: {e} — returning zeros')
        return {'physics': 0.0, 'boundary': 0.0, 'initial': 0.0, 'data': 0.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          type=str,   default='heat')
    parser.add_argument('--epochs',         type=int,   default=1000)
    parser.add_argument('--data',           type=str,   default='data/materials.parquet')
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--recompile_freq', type=int,   default=200)
    args = parser.parse_args()

    # 1. Load data
    try:
        expected_cols = ["time", "x", "temperature"]
        if args.model == "stress": expected_cols = ["x", "E", "eps", "sigma"]
        X, y = load_data(args.data, expected_cols)
        logger.info(f'Data loaded: X={X.shape}, y={y.shape}')
    except Exception as e:
        logger.warning(f"Data load failed: {e}. Generating dummy data.")
        X, y = np.random.rand(10, 2), np.random.rand(10, 1)

    # 2. Build model
    pinn = get_model(args.model)
    if hasattr(pinn, 'setup_model_with_anchors'):
        pinn.setup_model_with_anchors(X, y)
    else:
        pinn.setup_model()

    # 3. Adaptive loss — ordered to match DeepXDE outputs exactly
    adaptive_components = {
        'physics':  {'threshold': 0.020, 'weight': 1.0},
        'boundary': {'threshold': 0.050, 'weight': 1.0},
        'initial':  {'threshold': 0.030, 'weight': 1.0},
        'data':     {'threshold': 0.010, 'weight': 1.0},
    }
    
    # Adjust components dynamically based on what model produces
    # If the model only has 3 bc/ic features, we should remove the 4th, or adjust dynamically.
    # HeatModel definitely outputs 4 when anchors are provided.
    
    adaptive = AdaptiveLoss(
        components=adaptive_components,
        increase_factor=1.1,
        decay_factor=0.98,
        w_min=0.5,
        w_max=8.0,
    )

    # 4. Phase 1 — Adam + cosine annealing
    phase1_steps   = int(args.epochs * 0.8)
    recompile_freq = args.recompile_freq
    logger.info(f'Phase 1: Adam cosine annealing for {phase1_steps} steps (recompile every {recompile_freq} steps)')

    for cycle_start in range(0, phase1_steps, recompile_freq):
        current_lr = cosine_lr(args.lr, 1e-5, phase1_steps, cycle_start)
        
        # Get dynamic weights list based on actual number of bcs
        # Just use what adaptive gives, but strip it if model has fewer losses
        weights    = adaptive.get_weights_list()
        # count number of bcs + 1
        num_losses = len(pinn.model.data.bcs) + 1 if hasattr(pinn.model.data, 'bcs') else 1
        weights = weights[:num_losses]

        pinn.compile(optimizer='adam', learning_rate=current_lr, loss_weights=weights)
        logger.info(f'Step {cycle_start}: lr={current_lr:.6f} | {adaptive.summary()}')

        pinn.train(iterations=recompile_freq)

        losses = extract_losses(pinn.model)
        logger.info(f'Losses: physics={losses["physics"]:.3e} boundary={losses["boundary"]:.3e} initial={losses["initial"]:.3e} data={losses["data"]:.3e}')
        adaptive.update(losses)

    # 5. Phase 2 — L-BFGS fine-tuning
    phase2_steps  = args.epochs - phase1_steps
    final_weights = adaptive.get_weights_list()
    final_weights = final_weights[:num_losses]
    
    logger.info(f'Phase 2: L-BFGS fine-tuning for approx {phase2_steps} steps')
    try:
        pinn.compile(optimizer='L-BFGS', learning_rate=1.0, loss_weights=final_weights)
        pinn.model.train()
        logger.info('L-BFGS complete')
    except Exception as e:
        logger.warning(f'L-BFGS unavailable ({e}) — using Adam lr=1e-4 as fallback')
        pinn.compile(optimizer='adam', learning_rate=1e-4, loss_weights=final_weights)
        pinn.train(iterations=phase2_steps)

    # 6. Save predictions
    try:
        predictions = pinn.predict(X)
        logger.info(f'Predictions shape: {predictions.shape}')
        os.makedirs('outputs', exist_ok=True)
        np.save('outputs/predictions.npy', predictions)
        logger.info('Saved to outputs/predictions.npy')
    except Exception as e:
        logger.warning(f"Prediction failed: {e}")

if __name__ == '__main__':
    main()
