"""
Module: adaptive_loss.py
Purpose: Adaptive loss weight manager for PINN training.

Improvements over v1:
  1. Inverse-loss weighting  — w_i = 1 / (L_i + eps)  replaces brittle threshold logic.
  2. Per-component normalisation — L_i_norm = L_i / (mean(L_i_history) + eps)
     smooths out scale differences across physics, data, boundary, initial, and
     constraint loss terms.
  3. Clip weights to a stable range [w_min, w_max] to prevent gradient explosion.
  4. 'constraint' loss component is tracked alongside the standard four.
"""

import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveLoss:
    """
    Manages per-component loss weights using inverse-loss weighting.

    Parameters
    ----------
    components : dict
        Mapping of component name → initial weight, e.g.
        {'physics': 0.1, 'data': 1.0, 'boundary': 1.0,
         'initial': 1.0, 'constraint': 1.0}
    w_min : float
        Minimum allowed weight (prevents weights collapsing to zero).
    w_max : float
        Maximum allowed weight (prevents runaway amplification).
    eps : float
        Small constant added to loss values for numerical stability.
    history_len : int
        Number of recent loss values kept per component for running-mean
        normalisation.  Set to 1 to disable normalisation.
    """

    def __init__(
        self,
        components: dict,
        w_min: float = 0.5,
        w_max: float = 8.0,
        eps: float   = 1e-8,
        history_len: int = 50,
        # Legacy parameters kept for backwards-compatibility; no longer used.
        increase_factor: float = 1.1,
        decay_factor:    float = 0.98,
    ):
        # Store component names in insertion order (Python 3.7+).
        # Value is the *base* (initial) weight — used to initialise inverse-loss
        # weights before any training losses are observed.
        self.component_names  = list(components.keys())
        self.base_weights     = {
            n: float(components[n]) if not isinstance(components[n], dict)
               else float(components[n].get('weight', 1.0))
            for n in self.component_names
        }
        self.current_weights  = dict(self.base_weights)  # live weights

        self.w_min       = w_min
        self.w_max       = w_max
        self.eps         = eps
        self.history_len = history_len

        # Rolling history of raw loss values for normalisation
        self._history: dict[str, deque] = {
            n: deque(maxlen=history_len) for n in self.component_names
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, losses: dict) -> list:
        """
        Receive a dict of raw loss values, compute inverse-loss weights, and
        return them as an ordered list matching component insertion order.

        Algorithm
        ---------
        1. Append each loss to its rolling history.
        2. Compute running mean from history  →  used for normalisation.
        3. Normalise: L_i_norm = L_i / (running_mean_i + eps)
        4. Inverse weight: w_i_raw = 1 / (L_i_norm + eps)
        5. Scale so that the *data* component always has weight == base_weight
           (keeps the data term as the anchor and scales physics etc.
            relative to it).
        6. Clip to [w_min, w_max].
        7. Persist and log.
        """
        # Step 1 + 2: update history and compute normalised losses
        norm_losses = {}
        for name in self.component_names:
            raw = float(losses.get(name, 0.0))
            self._history[name].append(raw)
            running_mean = float(np.mean(self._history[name])) if self._history[name] else raw
            norm_losses[name] = raw / (running_mean + self.eps)

        # Step 3: inverse-loss weights (raw, unscaled)
        raw_weights = {
            n: 1.0 / (norm_losses[n] + self.eps)
            for n in self.component_names
        }

        # Step 4: anchor scale to the 'data' component's base weight so that
        # the data term is treated as ground-truth and physics is relative.
        anchor_name  = 'data' if 'data' in raw_weights else self.component_names[0]
        anchor_scale = (
            self.base_weights[anchor_name] / (raw_weights[anchor_name] + self.eps)
        )
        scaled_weights = {n: raw_weights[n] * anchor_scale for n in self.component_names}

        # Step 5: clip and store
        for name in self.component_names:
            w = float(np.clip(scaled_weights[name], self.w_min, self.w_max))
            self.current_weights[name] = w
            logger.info(
                f'[AdaptiveLoss] {name:12s} '
                f'loss={losses.get(name, 0.0):.3e}  '
                f'norm={norm_losses[name]:.3e}  '
                f'w={w:.4f}'
            )

        return self.get_weights_list()

    def get_weights_list(self) -> list:
        """Return weights as an ordered list (matches DeepXDE loss_weights order)."""
        return [self.current_weights[n] for n in self.component_names]

    def get_weights_dict(self) -> dict:
        """Return weights as a name → value dict."""
        return dict(self.current_weights)

    def summary(self) -> str:
        """Human-readable one-liner of current weights."""
        return 'weights: ' + ' | '.join(
            f'{n}={self.current_weights[n]:.3f}' for n in self.component_names
        )

    def reset(self):
        """Reset weights and history to base values (useful between phases)."""
        self.current_weights = dict(self.base_weights)
        for name in self.component_names:
            self._history[name].clear()
        logger.info('[AdaptiveLoss] weights and history reset to base values.')
