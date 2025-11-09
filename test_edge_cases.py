"""
Run a set of edge-case planets through the saved models and print/save results.

Usage:
    (venv) python test_edge_cases.py
"""

import os
import joblib
import pandas as pd
import numpy as np

# Paths to saved artifacts
NB_MODEL_PATH = "data/naive_bayes_model.pkl"
LR_CALIB_PATH = "data/log_reg_calibrated.pkl"   # calibrated logistic regression
SCALER_PATH = "data/feature_scaler.pkl"
OUT_DIR = "results"
OUT_CSV = os.path.join(OUT_DIR, "edge_case_results.csv")

# Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load models and scaler
try:
    nb_model = joblib.load(NB_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load Naive Bayes model from {NB_MODEL_PATH}: {e}")

try:
    lr_model = joblib.load(LR_CALIB_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load calibrated Logistic Regression from {LR_CALIB_PATH}: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load scaler from {SCALER_PATH}: {e}")

# Column ordering expected by the scaler/model
feature_cols = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_insol', 'st_teff', 'st_rad']

# Define the edge cases (name + feature dict)
cases = [
    ("Earth-like (baseline)", {'pl_rade':1.0, 'pl_bmasse':1.0, 'pl_orbsmax':1.0, 'pl_insol':1.0, 'st_teff':5778, 'st_rad':1.0}),
    ("Hot-Jupiter (too hot, big)", {'pl_rade':12.0, 'pl_bmasse':1000.0, 'pl_orbsmax':0.05, 'pl_insol':2500.0, 'st_teff':6000.0, 'st_rad':1.2}),
    ("Frozen Giant (too cold)", {'pl_rade':5.0, 'pl_bmasse':100.0, 'pl_orbsmax':50.0, 'pl_insol':0.001, 'st_teff':5000.0, 'st_rad':0.8}),
    ("Close Rocky (too hot)", {'pl_rade':1.5, 'pl_bmasse':3.0, 'pl_orbsmax':0.1, 'pl_insol':15.0, 'st_teff':5800.0, 'st_rad':1.0}),
    ("M-dwarf HZ (cool-star)", {'pl_rade':1.2, 'pl_bmasse':2.0, 'pl_orbsmax':0.2, 'pl_insol':1.0, 'st_teff':3000.0, 'st_rad':0.3}),
    ("A-type star (white-hot)", {'pl_rade':2.0, 'pl_bmasse':5.0, 'pl_orbsmax':1.0, 'pl_insol':1000.0, 'st_teff':9000.0, 'st_rad':2.0}),
    ("Borderline (upper hab limit)", {'pl_rade':1.8, 'pl_bmasse':5.0, 'pl_orbsmax':1.1, 'pl_insol':0.35, 'st_teff':5600.0, 'st_rad':1.0}),
]

# Run each case
results = []
for name, feats in cases:
    # Build DataFrame with one row, keeping the proper column order
    df_input = pd.DataFrame([{k: feats[k] for k in feature_cols}])
    # Use DataFrame for scaler to preserve feature names (avoids warnings)
    input_scaled = scaler.transform(df_input)

    # Predict probabilities
    prob_nb = float(nb_model.predict_proba(input_scaled)[0, 1])
    prob_lr = float(lr_model.predict_proba(input_scaled)[0, 1])

    # Decide verdict: habitable if either model gives >= 0.5 (you can adjust threshold)
    verdict = "Habitable" if (prob_nb >= 0.5 or prob_lr >= 0.5) else "Non-habitable"

    results.append({
        'case': name,
        **feats,
        'prob_nb': prob_nb,
        'prob_lr': prob_lr,
        'verdict': verdict
    })

# Create results DataFrame and show
res_df = pd.DataFrame(results)
# Order columns nicely
cols = ['case'] + feature_cols + ['prob_nb', 'prob_lr', 'verdict']
res_df = res_df[cols]

# Pretty print probabilities with 4 decimals
pd.options.display.float_format = '{:0.4f}'.format
print("\nEdge case results:\n")
print(res_df.to_string(index=False))

# Save CSV
res_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results to: {OUT_CSV}")
