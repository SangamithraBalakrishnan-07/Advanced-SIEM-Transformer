import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

# ============================
# CONFIGURATION
# ============================
INPUT_FILE = "advanced_siem_labeled.csv"
SEQ_LEN = 100
OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

print("\n=== Loading labeled dataset ===")
df = pd.read_csv(INPUT_FILE)
print("Loaded:", df.shape)


# ============================
# SELECT FEATURES
# ============================

# Numeric features
numeric_features = [
    "risk_score",  # from advanced_metadata
    "confidence",  # from advanced_metadata
    "process_id",
    "firmware_version",
    "src_port",
    "dst_port",
    "bytes",
    "duration",
]

# Keep only existing numeric columns
numeric_features = [col for col in numeric_features if col in df.columns]

# Categorical features (safe + meaningful)
categorical_features = [
    "event_type",
    "source",
    "severity",
    "user",
    "action",
    "object",
    "device_type",
    "device_id",
    "src_ip",
    "dst_ip",
    "alert_type",
    "protocol",
    "method",
    "category",
]

# Keep only columns present in dataset
categorical_features = [col for col in categorical_features if col in df.columns]

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)


# ============================
# LABEL ENCODING
# ============================
print("\n=== Encoding categorical features ===")

label_encoders = {}
cat_arrays = []

for col in categorical_features:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    cat_arrays.append(df[col].values.reshape(-1, 1))

X_cat = np.hstack(cat_arrays)

# Save label encoders
joblib.dump(label_encoders, OUT_DIR / "label_encoders.pkl")


# ============================
# SCALE NUMERIC FEATURES
# ============================
print("\n=== Scaling numeric features ===")

df[numeric_features] = df[numeric_features].fillna(0)

scaler = StandardScaler()
X_num = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, OUT_DIR / "scaler.pkl")


# ============================
# FINAL FEATURE MATRIX
# ============================
print("\n=== Building final feature matrix ===")

X = np.hstack([X_num, X_cat])
y = df["label"].values

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)


# ============================
# SESSION GROUPING
# ============================
print("\n=== Grouping events by session (user + src_ip) ===")

session_cols = []
for col in ["user", "src_ip", "device_id"]:
    if col in df.columns:
        session_cols.append(col)

if len(session_cols) == 0:
    df["session_id"] = "global"
else:
    df["session_id"] = df[session_cols].astype(str).agg("|".join, axis=1)

session_map = defaultdict(list)
for idx, sid in enumerate(df["session_id"]):
    session_map[sid].append(idx)

print("Total sessions:", len(session_map))


# ============================
# BUILD FIXED-LENGTH SEQUENCES
# ============================
print("\n=== Building 100-event sequences for Transformer ===")

sequences = []
labels = []
session_ids = []
seq_indices = []

for sid, idxs in session_map.items():
    idxs = sorted(idxs)

    for start in range(0, len(idxs), SEQ_LEN):
        chunk = idxs[start : start + SEQ_LEN]

        # Extract features
        seq = X[chunk]

        # Pad sequence if shorter than SEQ_LEN
        if len(chunk) < SEQ_LEN:
            pad_len = SEQ_LEN - len(chunk)
            pad = np.zeros((pad_len, X.shape[1]))
            seq = np.vstack([seq, pad])

        sequences.append(seq)
        labels.append(y[chunk[-1]])  
        session_ids.append(sid)
        seq_indices.append(chunk)

sequences = np.array(sequences)
labels = np.array(labels)

print("Sequences shape:", sequences.shape)
print("Labels shape:", labels.shape)


# ============================
# SAVE OUTPUTS
# ============================
print("\n=== Saving processed outputs ===")

np.save(OUT_DIR / "sequences.npy", sequences)
np.save(OUT_DIR / "labels.npy", labels)
joblib.dump(session_ids, OUT_DIR / "session_ids.pkl")
joblib.dump(seq_indices, OUT_DIR / "seq_indices.pkl")

print("\nEncode Complete!")
print("Saved files in /processed")
