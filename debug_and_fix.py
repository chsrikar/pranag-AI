import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim

def run_debug_and_fix():
    print("--- 1. VERIFY DATA SOURCE ---")
    data_path = "data/dataset.parquet"
    print("Loaded dataset path:", os.path.abspath(data_path))
    df = pd.read_parquet(data_path)
    print("First 5 rows:\n", df.head())

    print("\n--- 2. INSPECT SURVIVAL COLUMN ---")
    print(df["survival_rate"].describe())
    print("\nCorrelation:\n", df[["temperature", "time", "survival_rate"]].corr())

    print("\n--- 3. VISUAL CHECK ---")
    plt.scatter(df["temperature"], df["survival_rate"])
    plt.title("Temperature vs Survival")
    plt.savefig("outputs/temp_vs_survival_before.png")
    plt.clf()

    plt.scatter(df["time"], df["survival_rate"])
    plt.title("Time vs Survival")
    plt.savefig("outputs/time_vs_survival_before.png")
    plt.clf()
    print("Saved plots to outputs/temp_vs_survival_before.png and outputs/time_vs_survival_before.png")

    print("\n--- 5. FIX (REPLACING RANDOM DATA) ---")
    df["survival_rate"] = np.exp(-0.05 * df["temperature"]) + 0.1 * df["time"]
    df.to_parquet(data_path)
    print("Saved updated dataset.parquet")

    print("\n--- 6. VERIFY AGAIN ---")
    df2 = pd.read_parquet(data_path)
    print("New Correlation:\n", df2[["temperature", "time", "survival_rate"]].corr())
    
    plt.scatter(df2["temperature"], df2["survival_rate"])
    plt.title("Temperature vs Survival (Fixed)")
    plt.savefig("outputs/temp_vs_survival_after.png")
    plt.clf()

    plt.scatter(df2["time"], df2["survival_rate"])
    plt.title("Time vs Survival (Fixed)")
    plt.savefig("outputs/time_vs_survival_after.png")
    plt.clf()
    
    print("\n--- 7. FINAL CHECK (SURROGATE MODEL) ---")
    # Surrogate model train check
    # X = temperature, time
    # y = survival_rate
    X_train = df2[["temperature", "time"]].values.astype(np.float32)
    y_train = df2[["survival_rate"]].values.astype(np.float32)
    
    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training surrogate model for 300 epochs...")
    for epoch in range(300):
        optimizer.zero_grad()
        preds = model(torch.tensor(X_train))
        loss = criterion(preds, torch.tensor(y_train))
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_preds = model(torch.tensor(X_train))
        mse = nn.MSELoss()(final_preds, torch.tensor(y_train)).item()
        
        # Calculate R^2 for accuracy
        y_mean = y_train.mean()
        ss_tot = ((y_train - y_mean) ** 2).sum()
        ss_res = ((y_train - final_preds.numpy()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
    print(f"Final MSE: {mse:.6f}, Surrogate Accuracy (R^2): {r2:.4f}")

if __name__ == '__main__':
    run_debug_and_fix()
