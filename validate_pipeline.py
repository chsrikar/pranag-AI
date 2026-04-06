import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import deepxde as dde
from pinn_system.data_loader import load_data

def run_validation():
    # Verify the generated file
    data_path = "data/dataset.parquet"
    print("Dataset path:", os.path.abspath(data_path))
    print("Exists:", os.path.exists(data_path))
    
    df = pd.read_parquet(data_path)
    print("Preview:\n", df.head())
    print("Shape:", df.shape)

    # Statistical Validation
    print("\n--- Statistical Validation ---")
    print(df["survival_rate"].describe())
    corr = df.corr(numeric_only=True)
    print("\nCorrelation to survival_rate:")
    print(corr["survival_rate"].sort_values(ascending=False))

    # Visual Validation
    os.makedirs("outputs", exist_ok=True)
    plt.scatter(df["temperature"], df["survival_rate"], alpha=0.5)
    plt.title("Temperature vs Survival")
    plt.savefig("outputs/temperature_vs_survival_validated.png")
    plt.clf()

    plt.scatter(df["time"], df["survival_rate"], alpha=0.5)
    plt.title("Time vs Survival")
    plt.savefig("outputs/time_vs_survival_validated.png")
    plt.clf()
    
    # Save a versioned copy
    df.to_parquet("data/dataset_v2_physics.parquet", index=False)
    print("\nSaved bonus physics version to data/dataset_v2_physics.parquet")

if __name__ == '__main__':
    run_validation()
