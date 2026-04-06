"""
Module: surrogate_trainer.py
Purpose: Replace the slow PINN at inference time with a fast, lightweight PyTorch model.
         Trains ONLY on real data - NO DUMMY DATA ALLOWED.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pinn_system.data_loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SurrogateMLP(nn.Module):
    def __init__(self, inputs=2, outputs=1, hidden_layers=3, neurons_per_layer=64):
        super(SurrogateMLP, self).__init__()
        layers = []
        in_dim = inputs
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            in_dim = neurons_per_layer
        layers.append(nn.Linear(in_dim, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_surrogate(X_train: np.ndarray, y_train: np.ndarray, epochs=1000, batch_size=256, lr=0.001):
    """
    Trains a lightweight PyTorch surrogate model on real physics data.
    
    Args:
        X_train: Real input features (temperature, time, etc.)
        y_train: Real target values (survival_rate from physics formula or PINN predictions)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Trained model
    """
    # Validate inputs - NO DUMMY DATA
    assert X_train.shape[0] > 0, "❌ X_train is empty"
    assert y_train.shape[0] > 0, "❌ y_train is empty"
    assert X_train.shape[0] == y_train.shape[0], "❌ X and y have different lengths"
    assert not np.isnan(X_train).any(), "❌ X_train contains NaN"
    assert not np.isnan(y_train).any(), "❌ y_train contains NaN"
    assert y_train.std() > 0.01, f"❌ y_train variance too low: {y_train.std()}"
    
    logging.info("✅ Input validation passed")
    logging.info(f"Training data: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}, min={y_train.min():.4f}, max={y_train.max():.4f}")
    
    logging.info("Initializing Surrogate Model...")
    inputs = X_train.shape[1]
    outputs = y_train.shape[1]
    model = SurrogateMLP(inputs=inputs, outputs=outputs)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            
        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

    train_time = time.time() - start_time
    logging.info(f"✅ Surrogate training completed in {train_time:.2f} seconds.")
    
    # Evaluate model accuracy
    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor).numpy()
        mse = np.mean((all_preds - y_train) ** 2)
        mae = np.mean(np.abs(all_preds - y_train))
        # Calculate R² score
        ss_res = np.sum((y_train - all_preds) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        accuracy = r2 * 100  # Convert to percentage
        
        logging.info(f"✅ Model Performance:")
        logging.info(f"   MSE: {mse:.6f}")
        logging.info(f"   MAE: {mae:.6f}")
        logging.info(f"   R² Score: {r2:.4f}")
        logging.info(f"   Accuracy: {accuracy:.2f}%")
        
        if accuracy < 90:
            logging.warning(f"⚠️  Accuracy {accuracy:.2f}% is below target 90%")
        else:
            logging.info(f"✅ Accuracy {accuracy:.2f}% meets target!")
        
        # Test inference speed
        test_sample = X_tensor[:1]
        t0 = time.time()
        _ = model(test_sample)
        inf_time = time.time() - t0
        logging.info(f"Surrogate inference time per sample: {inf_time:.6f} seconds")
        
    save_path = os.path.join(os.path.dirname(__file__), "surrogate_model.pth")
    torch.save(model, save_path)
    logging.info(f"✅ Model saved to {save_path}")
    return model

def main():
    """
    Main function to train surrogate model using REAL data from dataset.parquet
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train surrogate model on real physics data")
    parser.add_argument('--data', type=str, default='data/dataset.parquet', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    logging.info("="*60)
    logging.info("SURROGATE MODEL TRAINING - REAL DATA ONLY")
    logging.info("="*60)
    
    # Load real data
    try:
        # Get absolute path
        if not os.path.isabs(args.data):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_path = os.path.join(project_root, args.data)
        else:
            data_path = args.data
            
        logging.info(f"Loading data from: {data_path}")
        
        # Load features and target
        expected_cols = ["time", "x", "temperature", "survival_rate"]
        X, y = load_data(data_path, expected_cols)
        
        logging.info(f"✅ Loaded real data: X={X.shape}, y={y.shape}")
        
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        logging.error("❌ CANNOT PROCEED - NO DUMMY DATA ALLOWED")
        raise
    
    # Train surrogate model
    model = train_surrogate(X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    logging.info("="*60)
    logging.info("✅ SURROGATE TRAINING COMPLETE")
    logging.info("="*60)
    
    return model

if __name__ == "__main__":
    main()
