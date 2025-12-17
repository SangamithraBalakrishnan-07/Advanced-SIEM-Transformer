import pandas as pd
import numpy as np
from pandas import json_normalize
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
RAW_PATH = Path("/Users/mithra/advanced_siem_dataset.jsonl")   # <-- update path if needed

print("\n==============================")
print(" STEP 1: LOAD JSON FILE")
print("==============================\n")

# Try both JSON formats (line-delimited / standard)
try:
    df = pd.read_json(RAW_PATH, lines=True)
except ValueError:
    df = pd.read_json(RAW_PATH)

print("Loaded dataset shape:", df.shape)
print(df.head(), "\n")


# =========================================================
# STEP 2 — FLATTEN NESTED FIELDS
# =========================================================
print("\n==============================")
print(" STEP 2: FLATTEN NESTED JSON")
print("==============================\n")

df_flat = json_normalize(df.to_dict(orient="records"))

# Replace dot notation with underscores: event.src_ip → event_src_ip
df_flat.columns = [c.replace(".", "_") for c in df_flat.columns]

print("Flattened shape:", df_flat.shape)
print(df_flat.head(), "\n")


# =========================================================
# STEP 3 — CHECK COLUMNS + VERIFY LABELS
# =========================================================
print("\n==============================")
print(" STEP 3: COLUMN & LABEL CHECK")
print("==============================\n")

print("All columns:\n", df_flat.columns.tolist(), "\n")

# Find label column automatically
label_col = None
for col in df_flat.columns:
    if "attack" in col.lower() or "stage" in col.lower():
        label_col = col
        break

if label_col:
    print("Detected label column:", label_col)
    print("\nLabel distribution:\n", df_flat[label_col].value_counts(), "\n")
else:
    print("WARNING: No attack stage label found!")


# =========================================================
# STEP 4 — PREPARE FOR PREPROCESSING (CLEANING)
# =========================================================
print("\n==============================")
print(" STEP 4: BASIC CLEANING")
print("==============================\n")

# ---- Normalize timestamps ----
if "timestamp" in df_flat.columns:
    df_flat["timestamp"] = pd.to_datetime(
        df_flat["timestamp"], errors="coerce", utc=True
    )

# Drop missing timestamps
if "timestamp" in df_flat.columns:
    df_flat = df_flat.dropna(subset=["timestamp"]).reset_index(drop=True)

# ---- Add UNIX timestamp ----
if "timestamp" in df_flat.columns:
    df_flat["timestamp_unix"] = df_flat["timestamp"].astype("int64") // 10**9

# ---- Fill missing categorical values ----
cat_cols = df_flat.select_dtypes(include="object").columns
for col in cat_cols:
    df_flat[col] = df_flat[col].fillna("unknown")

# ---- Fill missing numeric values ----
num_cols = df_flat.select_dtypes(include=np.number).columns
for col in num_cols:
    df_flat[col] = df_flat[col].fillna(df_flat[col].median())

print("Cleaned dataset preview:")
print(df_flat.head(), "\n")

print("Final cleaned shape:", df_flat.shape)

df_flat.to_csv("advanced_siem_cleaned.csv", index=False)
print("\nSaved cleaned file as: advanced_siem_cleaned.csv")


print("\nINITIAL PROCESSING COMPLETE.")

