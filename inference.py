# inference.py

import torch
import json
import requests
import os
from pathlib import Path

from models.physics_models import (
    BiologyPINN, StressPINN, HeatPINN, ChemistryPINN, GrowthPINN
)
from datasrc.data_loader import PINNDataLoader

# ============================================================
# Load ENV
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("No API key → LLM disabled (fallback will be used)")


# ============================================================
# 1. Load trained models
# ============================================================
def load_models():
    models = {
        "biology": BiologyPINN(),
        "stress": StressPINN(),
        "heat": HeatPINN(),
        "chemistry": ChemistryPINN(),
        "growth": GrowthPINN()
    }

    for name, model in models.items():
        path = f"outputs/models/{name}_pinn.pt"

        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model.load(path)
        model.eval()

    print("All models loaded successfully")
    return models


# ============================================================
# 2. JSON → SAME PIPELINE AS TRAINING
# ============================================================
def prepare_inputs(json_data):
    loader = PINNDataLoader(data_dir="datasrc/")
    loader.load()

    df = loader.build_feature_matrix(extra_features=json_data)

    bio_tensors  = loader.to_biology_tensors(df)
    heat_tensors = loader.to_heat_tensors(df)
    chem_tensors = loader.to_chemistry_tensors(df)

    inputs = {
        "biology":   bio_tensors["X_test"][:1],
        "stress":    bio_tensors["X_test"][:1][:, :3],
        "growth":    bio_tensors["X_test"][:1][:, :3],
        "heat":      heat_tensors["X_test"][:1],
        "chemistry": chem_tensors["X_test"][:1],
    }

    return inputs


# ============================================================
# 3. Run inference
# ============================================================
def run_inference(models, json_data):
    inputs = prepare_inputs(json_data)

    results = {}

    with torch.no_grad():
        for name, model in models.items():
            X = inputs[name]
            print(f"{name}: input shape {X.shape}")
            pred = model(X)
            results[name] = float(pred.mean().item())

    return results


# ============================================================
# 4. Interpretation
# ============================================================
_CONDITION_LABELS = {
    "biology": {
        "high": "High biological activity",
        "moderate": "Moderate biological activity",
        "low": "Low biological activity",
    },
    "stress": {
        "high": "High stress",
        "moderate": "Moderate stress",
        "low": "Low stress",
    },
    "heat": {
        "high": "High thermal load",
        "moderate": "Moderate thermal load",
        "low": "Low thermal load",
    },
    "chemistry": {
        "high": "High chemical activity",
        "moderate": "Moderate chemical activity",
        "low": "Low chemical activity",
    },
    "growth": {
        "high": "High growth potential",
        "moderate": "Moderate growth potential",
        "low": "Low growth potential",
    },
}

def _score_to_tier(v):
    if v >= 0.8:
        return "high"
    elif v >= 0.3:
        return "moderate"
    return "low"

def explain_results(results):
    out = {}
    for k, v in results.items():
        tier = _score_to_tier(v)
        out[k] = f"{v:.2f} → {_CONDITION_LABELS[k][tier]}"
    return out


# ============================================================
# 5. LLM + DEBUG + FALLBACK
# ============================================================
def generate_human_readable(json_input, predictions, explanations):

    # If no API key → skip LLM
    if not OPENROUTER_API_KEY:
        return fallback_report(predictions, explanations)

    url = "https://openrouter.ai/api/v1/chat/completions"

    prompt = f"""
You are an agricultural expert.

INPUT:
{json.dumps(json_input, indent=2)}

PREDICTIONS:
{json.dumps(predictions, indent=2)}

EXPLANATIONS:
{json.dumps(explanations, indent=2)}

Write a clear real-world analysis with:
- Overall condition
- Key strengths
- Risks
- Practical recommendation
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "PINN System"
    }

    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        print("LLM STATUS:", response.status_code)
        print("LLM RAW:", response.text[:300])  # partial for safety

        if response.status_code != 200:
            return fallback_report(predictions, explanations)

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM exception:", str(e))
        return fallback_report(predictions, explanations)


# ============================================================
# FALLBACK
# ============================================================
def fallback_report(predictions, explanations):
    report = "Crop Analysis Report:\n\n"

    for k, v in explanations.items():
        report += f"- {k.capitalize()}: {v}\n"

    report += "\nSummary:\n"

    if predictions["growth"] > 0.8:
        report += "Strong growth potential detected.\n"

    if predictions["chemistry"] < 0.3:
        report += "Soil chemistry imbalance detected.\n"

    if predictions["stress"] > 0.6:
        report += "Moderate environmental stress observed.\n"

    report += "\nRecommendation:\nAdjust soil nutrients and monitor conditions."

    return report


# ============================================================
# 6. Output
# ============================================================
def format_output(results, json_data):
    explanations = explain_results(results)

    report = generate_human_readable(
        json_data,
        results,
        explanations
    )

    return {
        "predictions": results,
        "explanations": explanations,
        "human_readable_report": report,
        "status": "success"
    }


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    print("\nStarting Inference Pipeline...\n")

    models = load_models()

    with open("datasrc/out1.json") as f:
        json_data = json.load(f)

    results = run_inference(models, json_data)

    final_output = format_output(results, json_data)

    print("\nFINAL OUTPUT:")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))