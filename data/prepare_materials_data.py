import os
import json
import pandas as pd
import glob

def process_materials(db_path, out_path):
    print("Processing materials from:", db_path)
    files = glob.glob(os.path.join(db_path, "**", "*.json"), recursive=True)
    
    records = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        if 'structure' in item and 'lattice' in item['structure']:
                            lattice = item['structure']['lattice']
                            records.append({
                                'x': lattice.get('a', 0.0),
                                'time': lattice.get('b', 0.0),
                                'temperature': lattice.get('volume', 0.0) 
                            })
            except Exception as e:
                pass
                
    df = pd.DataFrame(records).dropna()
    print(f"Extracted {len(df)} records.")
    
    if len(df) > 0:
        for col in ['time', 'x', 'temperature']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
                
        df.to_parquet(out_path, engine="pyarrow")
        print(f"Saved normalized data to {out_path}")
    else:
        print("No valid records found.")

if __name__ == "__main__":
    db_path = "d:/ai_ml_internship/Database-20260405T091335Z-3-001/Database/data/raw/materials_project/json"
    out_path = "d:/ai_ml_internship/data/materials.parquet"
    process_materials(db_path, out_path)
