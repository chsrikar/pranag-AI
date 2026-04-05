import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveLoss:
    def __init__(self, components: dict, increase_factor=1.1, decay_factor=0.98, w_min=0.5, w_max=8.0):
        # We rely on insertion order to match DeepXDE loss components order.
        self.components = {n: {'threshold': c['threshold'], 'weight': float(c.get('weight', 1.0))} for n, c in components.items()}
        self.increase_factor = increase_factor
        self.decay_factor    = decay_factor
        self.w_min           = w_min
        self.w_max           = w_max

    def update(self, losses: dict) -> list:
        for name, cfg in self.components.items():
            loss_val = float(losses.get(name, 0.0))
            if loss_val > cfg['threshold']:
                cfg['weight'] *= self.increase_factor
                logger.info(f'[AdaptiveLoss] {name:10s} {loss_val:.3e} > {cfg["threshold"]:.3e} => RAISED  to {cfg["weight"]:.4f}')
            else:
                cfg['weight'] *= self.decay_factor
                logger.info(f'[AdaptiveLoss] {name:10s} {loss_val:.3e} <= {cfg["threshold"]:.3e} => decayed to {cfg["weight"]:.4f}')
            cfg['weight'] = float(np.clip(cfg['weight'], self.w_min, self.w_max))
        return self.get_weights_list()

    def get_weights_list(self) -> list:
        return [cfg['weight'] for cfg in self.components.values()]

    def get_weights_dict(self) -> dict:
        return {name: cfg['weight'] for name, cfg in self.components.items()}

    def summary(self) -> str:
        return 'weights: ' + ' | '.join(f'{n}={cfg["weight"]:.3f}' for n, cfg in self.components.items())
