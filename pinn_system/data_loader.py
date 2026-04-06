"""
Module: data_loader.py
Purpose: Ingest and preprocess physical dataset for PINN training using DuckDB.
"""
import duckdb
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(parquet_path: str, query_columns: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a Parquet file using DuckDB.
    
    Args:
        parquet_path: Path to the dataset.parquet file.
        query_columns: List of columns to extract (e.g., ['time', 'x', 'temperature']).
                       Assumes last column is target (y), rest are inputs (X).
                       
    Returns:
        Tuple (X, y) as numpy arrays.
    """
    try:
        logging.info(f"Connecting to DuckDB and querying {parquet_path}...")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

        con = duckdb.connect(database=':memory:')
        
        cols_str = ", ".join(query_columns)
        query = f"SELECT {cols_str} FROM read_parquet('{parquet_path}')"
        
        # Execute query and get result
        res = con.execute(query).df()
        
        logging.info(f"Loaded {len(res)} rows from {parquet_path}.")
        logging.info(f"Columns and types: \n{res.dtypes}")
        
        res.dropna(inplace=True)
        logging.info(f"Rows after dropping missing values: {len(res)}.")
        
        # --- DATA VALIDATION FIREWALL ---
        if "survival_rate" in res.columns:
            std = res["survival_rate"].std()
            # Find max correlation with *other* columns
            corr = res.corr(numeric_only=True)["survival_rate"].drop("survival_rate", errors="ignore").abs().max()

            print("Survival std:", std)
            print("Max correlation:", corr)

            if std < 0.01:
                raise ValueError("❌ survival_rate variance too low")

            if corr < 0.1:
                raise ValueError("❌ survival_rate not correlated with inputs")
        # --------------------------------

        # X is all but last column, y is last column
        X = res[query_columns[:-1]].to_numpy(dtype=np.float32)
        y = res[query_columns[-1:]].to_numpy(dtype=np.float32)
        
        return X, y
        
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

if __name__ == "__main__":
    try:
        dummy_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.parquet")
        X, y = load_data(dummy_path, ["time", "x", "temperature", "survival_rate"])
        print("X shape:", X.shape)
        print("y shape:", y.shape)
    except Exception as e:
        print("Smoke test failed (data might not exist):", e)
