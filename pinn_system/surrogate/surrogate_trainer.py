"""
Module: surrogate_trainer.py
Purpose: Replace the slow PINN at inference time with a fast, lightweight PyTorch model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

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
    Trains a lightweight PyTorch surrogate model on PINN predictions.
    X_train: PINN inputs
    y_train: PINN outputs (physics prediction)
    """
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
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

    train_time = time.time() - start_time
    logging.info(f"Surrogate training completed in {train_time:.2f} seconds.")
    
    model.eval()
    with torch.no_grad():
        test_sample = X_tensor[:1]
        t0 = time.time()
        _ = model(test_sample)
        inf_time = time.time() - t0
        logging.info(f"Surrogate inference time per sample: {inf_time:.6f} seconds")
        
    import os
    save_path = os.path.join(os.path.dirname(__file__), "surrogate_model.pth")
    torch.save(model, save_path)
    logging.info(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    dummy_X = np.random.rand(1000, 2)
    dummy_y = np.random.rand(1000, 1)
    train_surrogate(dummy_X, dummy_y, epochs=100)
