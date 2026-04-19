"""
Module: data_loader.py
Purpose: Ingest and preprocess physical dataset for PINN training using DuckDB.
         Data source: universal_index.parquet (materials science dataset with 1882 samples,
         columns: temperature_max, strength, conductivity, ph, salinity)
"""
import duckdb
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Canonical path to the single data source ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.abspath(os.path.join(_HERE, "..", "data", "universal_index.parquet"))

# ── Physical columns present in universal_index.parquet ───────────────────────
NUMERIC_COLS = ["temperature_max", "strength", "conductivity", "ph", "salinity"]

# ── Per-model column sets (inputs + [target]) ──────────────────────────────────
# Last entry in each list is the regression target (y); the rest are inputs (X).
MODEL_COLUMNS = {
    "heat":      ["strength", "conductivity", "temperature_max"],   # heat diffusion  → T_max
    "stress":    ["temperature_max", "conductivity", "strength"],   # Hooke's law     → strength
    "growth":    ["temperature_max", "ph", "salinity"],             # logistic growth → salinity
    "biology":   ["temperature_max", "salinity", "ph"],             # decay ODE       → ph
    "chemistry": ["temperature_max", "strength", "conductivity"],   # Arrhenius       → conductivity
}


def load_data(
    parquet_path: str = PARQUET_PATH,
    query_columns: list = None,
    model_name: str = None,
) -> tuple:
    """
    Load and preprocess data from universal_index.parquet.

    Priority:
      1. If `model_name` is given, use MODEL_COLUMNS[model_name].
      2. Else if `query_columns` is given, use those columns directly.
      3. Else raise ValueError.

    Args:
        parquet_path  : Path to universal_index.parquet (defaults to PARQUET_PATH).
        query_columns : Explicit list of columns; last entry is the target.
        model_name    : One of 'heat','stress','growth','biology','chemistry'.

    Returns:
        (X, y) as float32 numpy arrays, both Min-Max normalised to [0, 1].
    """
    if model_name is not None:
        if model_name not in MODEL_COLUMNS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid choices: {list(MODEL_COLUMNS.keys())}"
            )
        query_columns = MODEL_COLUMNS[model_name]

    if query_columns is None:
        raise ValueError(
            "Provide either model_name or query_columns. "
            f"Known models: {list(MODEL_COLUMNS.keys())}"
        )

    try:
        logging.info(f"Querying {parquet_path} | columns={query_columns}")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        con = duckdb.connect(database=":memory:")
        cols_str = ", ".join(query_columns)
        # Escape backslashes for DuckDB on Windows
        safe_path = parquet_path.replace("\\", "/")
        query = f"SELECT {cols_str} FROM read_parquet('{safe_path}')"
        res = con.execute(query).df()
        logging.info(f"Fetched {len(res)} rows before cleaning.")

        res = res[query_columns].dropna()
        logging.info(f"Rows after dropna: {len(res)}")

        if len(res) == 0:
            raise ValueError("No valid rows remain after dropping NaNs.")

        # ── Min-Max normalisation ──────────────────────────────────────────────
        arr = res.to_numpy(dtype=np.float32)
        mins = arr.min(axis=0, keepdims=True)
        maxs = arr.max(axis=0, keepdims=True)
        denom = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
        arr_norm = (arr - mins) / denom

        X = arr_norm[:, :-1]   # all columns except last
        y = arr_norm[:, -1:]   # last column is the target

        logging.info(f"X shape: {X.shape} | y shape: {y.shape}")
        logging.info(f"y stats → mean={y.mean():.4f}, std={y.std():.4f}, "
                     f"min={y.min():.4f}, max={y.max():.4f}")

        # ── Data quality firewall ──────────────────────────────────────────────
        if y.std() < 0.01:
            raise ValueError(
                f"Target variance too low (std={y.std():.4f}). "
                "Data has insufficient signal for training."
            )

        return X, y

    except Exception as e:
        logging.error(f"load_data failed: {e}")
        raise


if __name__ == "__main__":
    print("=== Smoke test: universal_index.parquet ===")
    for mn in MODEL_COLUMNS:
        X, y = load_data(model_name=mn)
        print(f"  [{mn:10s}]  X={X.shape}  y={y.shape}  "
              f"y_std={y.std():.4f}")
    print("All models OK.")
