import pandas as pd
import numpy as np
import os

def create_dummy_dataset(output_path="dataset.parquet"):
    """Generates a dummy dataset for PINN training."""
    np.random.seed(42)
    
    # Generate random points in space and time
    n_samples = 1000
    time = np.random.uniform(0, 1, n_samples)
    spatial_input = np.random.uniform(0, 1, n_samples)
    
    # Generic outputs for testing
    temperature = np.sin(np.pi * spatial_input) * np.exp(-time)  # Heat-like
    stress = 2.0E11 * spatial_input  # Stress-like
    population = 1000 / (1 + ((1000 - 100) / 100) * np.exp(-0.5 * time)) # Growth-like
    quantity = 100 * np.exp(-0.1 * time) # Biology radioactive decay-like
    reaction_rate = 1.0e13 * np.exp(-50000 / (8.314 * (300 + 100 * time))) # Chemistry-like
    survival_rate = np.random.uniform(0, 1, n_samples) # Dummy

    
    df = pd.DataFrame({
        "time": time,
        "x": spatial_input,
        "temperature": temperature,
        "stress": stress,
        "population": population,
        "quantity": quantity,
        "reaction_rate": reaction_rate,
        "survival_rate": survival_rate
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Dataset generated at {output_path} with {n_samples} rows.")

if __name__ == "__main__":
    create_dummy_dataset("d:/ai_ml_internship/data/dataset.parquet")
