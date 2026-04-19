"""
Module: surrogate_trainer.py
Purpose: Lightweight PyTorch surrogate that replaces the PINN at inference time.
         Trained exclusively on real data from universal_index.parquet.
         NO DUMMY DATA ALLOWED.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pinn_system.data_loader import load_data, PARQUET_PATH, MODEL_COLUMNS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── Surrogate Architecture ─────────────────────────────────────────────────────
class SurrogateMLP(nn.Module):
    def __init__(self, inputs: int = 2, outputs: int = 1,
                 hidden_layers: int = 3, neurons_per_layer: int = 64):
        super().__init__()
        layers = []
        in_dim = inputs
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            in_dim = neurons_per_layer
        layers.append(nn.Linear(in_dim, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training ───────────────────────────────────────────────────────────────────
def train_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 0.001,
    save_path: str = None,
) -> SurrogateMLP:
    """
    Train a lightweight surrogate on real physics data.

    Args:
        X_train    : Feature matrix (float32, shape [N, F]).
        y_train    : Target vector  (float32, shape [N, 1]).
        epochs     : Training epochs.
        batch_size : Mini-batch size.
        lr         : Learning rate.
        save_path  : Where to save the .pth file (defaults to surrogate/ dir).

    Returns:
        Trained SurrogateMLP model.
    """
    # ── Validation firewall ────────────────────────────────────────────
    assert X_train.shape[0] > 0,                          "X_train is empty"
    assert y_train.shape[0] > 0,                          "y_train is empty"
    assert X_train.shape[0] == y_train.shape[0],          "X and y length mismatch"
    assert not np.isnan(X_train).any(),                   "X_train contains NaN"
    assert not np.isnan(y_train).any(),                   "y_train contains NaN"
    assert y_train.std() > 0.01, (
        f"y_train variance too low: std={y_train.std():.4e}. "
        "Check universal_index.parquet for degenerate columns."
    )

    logging.info("Input validation passed ✅")
    logging.info(f"Training data — X={X_train.shape}, y={y_train.shape}")
    logging.info(
        f"Target stats — mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
        f"min={y_train.min():.4f}, max={y_train.max():.4f}"
    )

    # ── Build model ────────────────────────────────────────────────────
    model     = SurrogateMLP(inputs=X_train.shape[1], outputs=y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor  = torch.tensor(X_train, dtype=torch.float32)
    y_tensor  = torch.tensor(y_train, dtype=torch.float32)
    dataset   = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # ── Training loop ──────────────────────────────────────────────────
    start_time = time.time()
    best_loss  = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss  = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 100 == 0:
            logging.info(
                f"Epoch [{epoch+1}/{epochs}]  loss={avg_loss:.6f}  best={best_loss:.6f}"
            )

    logging.info(
        f"Training complete in {time.time() - start_time:.2f}s  "
        f"best_loss={best_loss:.6f} ✅"
    )

    # ── Evaluation ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor).numpy()
        mse       = np.mean((all_preds - y_train) ** 2)
        mae       = np.mean(np.abs(all_preds - y_train))
        ss_res    = np.sum((y_train - all_preds) ** 2)
        ss_tot    = np.sum((y_train - y_train.mean()) ** 2)
        r2        = 1.0 - (ss_res / (ss_tot + 1e-12))

        logging.info(f"Performance — MSE={mse:.6f}  MAE={mae:.6f}  R²={r2:.4f}  ({r2*100:.2f}%)")
        if r2 * 100 < 90:
            logging.warning(f"R²={r2*100:.2f}% is below 90% target.")
        else:
            logging.info(f"R²={r2*100:.2f}% meets target ✅")

        # Inference speed
        t0  = time.time()
        _   = model(X_tensor[:1])
        logging.info(f"Inference per sample: {time.time()-t0:.6f}s")

    # ── Save ───────────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "surrogate_model.pth")
    torch.save(model, save_path)
    logging.info(f"Model saved → {save_path}")
    return model


# ── CLI entry-point ────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train surrogate on universal_index.parquet"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='heat',
        choices=list(MODEL_COLUMNS.keys()),
        help=f"Which physics model's column set to use: {list(MODEL_COLUMNS.keys())}",
    )
    parser.add_argument('--epochs',     type=int,   default=1000)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=0.001)
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("SURROGATE MODEL TRAINING — universal_index.parquet")
    logging.info(f"Model  : {args.model.upper()}")
    logging.info(f"Source : {PARQUET_PATH}")
    logging.info(f"Cols   : {MODEL_COLUMNS[args.model]}")
    logging.info("=" * 60)

    X, y = load_data(model_name=args.model)
    logging.info(f"Data loaded — X={X.shape}, y={y.shape}")

    save_dir  = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, f"surrogate_{args.model}.pth")

    train_surrogate(X, y, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    save_path=save_path)

    logging.info("=" * 60)
    logging.info("✅ SURROGATE TRAINING COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
