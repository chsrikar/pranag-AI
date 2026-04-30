import torch
import json
import requests
import os
import re
from pathlib import Path

from models.physics_models import (
    BiologyPINN, StressPINN, HeatPINN, ChemistryPINN, GrowthPINN
)
from datasrc.data_loader import PINNDataLoader

# ============================================================
# ENV
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("No API key. LLM disabled. Using fallback.")


# ============================================================
# 1. Load models
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
        model.load(path)
        model.eval()

    print("Models loaded successfully")
    return models


# ============================================================
# 2. Input pipeline
# ============================================================
def prepare_inputs(json_data):
    loader = PINNDataLoader(data_dir="datasrc/")
    loader.load()

    df = loader.build_feature_matrix(extra_features=json_data)

    bio = loader.to_biology_tensors(df)
    heat = loader.to_heat_tensors(df)
    chem = loader.to_chemistry_tensors(df)

    return {
        "biology": bio["X_test"][:1],
        "stress": bio["X_test"][:1][:, :3],
        "growth": bio["X_test"][:1][:, :3],
        "heat": heat["X_test"][:1],
        "chemistry": chem["X_test"][:1],
    }


# ============================================================
# 3. Inference
# ============================================================
def run_inference(models, json_data):
    inputs = prepare_inputs(json_data)
    results = {}

    with torch.no_grad():
        for name, model in models.items():
            pred = model(inputs[name])
            results[name] = float(pred.mean().item())

    return results


# ============================================================
# 4. Explanation
# ============================================================
LABELS = {
    "biology": ["Low biological activity", "Moderate biological activity", "High biological activity"],
    "stress": ["Low stress", "Moderate stress", "High stress"],
    "heat": ["Low thermal load", "Moderate thermal load", "High thermal load"],
    "chemistry": ["Low chemical activity", "Moderate chemical activity", "High chemical activity"],
    "growth": ["Low growth potential", "Moderate growth potential", "High growth potential"]
}

def tier(v):
    if v >= 0.8:
        return 2
    elif v >= 0.3:
        return 1
    return 0

def explain(results):
    return {
        k: f"{v:.2f} - {LABELS[k][tier(v)]}"
        for k, v in results.items()
    }


# ============================================================
# 5. Feasibility
# ============================================================
def detect_feasibility(json_data):
    temp = json_data.get("temperature", 0)
    water = json_data.get("water_availability", 1)

    if temp > 50:
        return "impossible"
    if temp > 40 or water < 0.1:
        return "risky"
    return "feasible"


# ============================================================
# 6. JSON extractor
# ============================================================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


# ============================================================
# 7. LLM
# ============================================================
def generate_structured(json_input, predictions, explanations):

    if not OPENROUTER_API_KEY:
        return fallback(explanations)

    prompt = f"""
Return valid JSON only.

Use these explanations:
{json.dumps(explanations)}

Fields:
overall_condition
biology
stress
heat
chemistry
growth
key_risk
recommendation
feasibility
"""

    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )

        if res.status_code != 200:
            return fallback(explanations)

        raw = res.json()["choices"][0]["message"]["content"]

        json_str = extract_json(raw)
        if not json_str:
            return fallback(explanations)

        data = json.loads(json_str)

        # enforce correct explanation values
        for k in explanations:
            data[k] = explanations[k]

        return data

    except:
        return fallback(explanations)


# ============================================================
# 8. Fallback
# ============================================================
def fallback(explanations):
    return {
        "overall_condition": "Moderate stress with strong growth potential",
        "biology": explanations["biology"],
        "stress": explanations["stress"],
        "heat": explanations["heat"],
        "chemistry": explanations["chemistry"],
        "growth": explanations["growth"],
        "key_risk": "Low chemical activity",
        "recommendation": "Improve irrigation and nutrients",
        "feasibility": "unknown"
    }


# ============================================================
# 9. Crisp Human Report
# ============================================================
def build_human_report(analysis):

    if analysis["feasibility"] == "impossible":
        return "Condition: Not feasible. Environmental limits exceeded."

    return (
        f"Condition: {analysis['overall_condition']}\n"
        f"Growth: {analysis['growth']}\n"
        f"Risk: {analysis['key_risk']}\n"
        f"Action: {analysis['recommendation']}"
    )


# ============================================================
# 10. Output
# ============================================================
def format_output(results, json_data):

    explanations = explain(results)

    structured = generate_structured(
        json_data,
        results,
        explanations
    )

    # system override
    structured["feasibility"] = detect_feasibility(json_data)

    # human report
    human_report = build_human_report(structured)

    return {
        "predictions": results,
        "explanations": explanations,
        "analysis": structured,
        "human_readable_report": human_report,
        "status": "success"
    }


# ============================================================
# 11. MAIN
# ============================================================
if __name__ == "__main__":
    print("\nRunning inference...\n")

    models = load_models()

    with open("datasrc/out1.json") as f:
        json_data = json.load(f)

    results = run_inference(models, json_data)

    output = format_output(results, json_data)

    print(json.dumps(output, indent=2))