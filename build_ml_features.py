import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ============================
# LOAD SEQUENCES + LABELS
# ============================
print("Loading sequence dataset...")

X_seq = np.load("/Users/mithra/processed/sequences.npy")   # shape: (N, 100, 20)
y = np.load("/Users/mithra/processed/labels.npy")          # shape: (N,)

print("Sequences shape:", X_seq.shape)
print("Labels shape:", y.shape)


# ============================
# FEATURE ENGINEERING FOR ML
# ============================
print("Building ML feature matrix...")

# Compute features across the time dimension (axis=1)
X_mean = X_seq.mean(axis=1)        # shape: (N, 20)
X_std  = X_seq.std(axis=1)         # shape: (N, 20)
X_min  = X_seq.min(axis=1)         # shape: (N, 20)
X_max  = X_seq.max(axis=1)         # shape: (N, 20)

# Concatenate features → final (N, 80) matrix
X_ml = np.hstack([X_mean, X_std, X_min, X_max])

print("Final ML feature matrix shape:", X_ml.shape)

# ============================
# SAVE ML DATASET
# ============================
OUT_DIR = Path("processed_ml")
OUT_DIR.mkdir(exist_ok=True)

np.save(OUT_DIR / "X_ml.npy", X_ml)
np.save(OUT_DIR / "y_ml.npy", y)

print("\nML Feature Complete!")
print("Saved:")
print(" - processed_ml/X_ml.npy")
print(" - processed_ml/y_ml.npy")
