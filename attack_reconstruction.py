import json
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("networkx not installed; graph layout will be simplified.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_CSV = "advanced_siem_labeled.csv"
MODEL_DIR = "logbert-cls"
OUT_DIR = Path("reconstruction_outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_LABELS = 6  # 0=benign, 1=recon, 2=exploit, 3=priv_esc, 4=lateral, 5=exfil


# --------------------------------------------------
# Helper: build same log text as before
# --------------------------------------------------
def build_log_text(row):
    parts = []
    parts.append(f"[EVENT_TYPE] {row.get('event_type', '')}")
    parts.append(f"[SEVERITY] {row.get('severity', '')}")
    parts.append(f"[SOURCE] {row.get('source', '')}")
    if "user" in row and pd.notna(row["user"]):
        parts.append(f"[USER] {row['user']}")
    if "src_ip" in row and pd.notna(row["src_ip"]):
        parts.append(f"[SRC_IP] {row['src_ip']}")
    if "dst_ip" in row and pd.notna(row["dst_ip"]):
        parts.append(f"[DST_IP] {row['dst_ip']}")
    parts.append(f"[DESC] {row.get('description', '')}")
    if "additional_info" in row and pd.notna(row["additional_info"]):
        parts.append(f"[INFO] {row['additional_info']}")
    return " ".join(parts)


# --------------------------------------------------
# 1. Load events & model, compute anomaly scores
# --------------------------------------------------
print("Loading events...")
df = pd.read_csv(DATA_CSV)

if "event_id" not in df.columns:
    # If event_id missing, create one
    df["event_id"] = df.index.astype(str)

df["text"] = df.apply(build_log_text, axis=1)

# Parse timestamp into datetime
if "timestamp" in df.columns:
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    # if no timestamp, fake monotonic time
    df["timestamp_dt"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

labels = df["label"].values

print("Loading Log-BERT model from", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

texts = df["text"].tolist()
all_probs = []
all_preds = []

batch_size = 32
print("Scoring events with BERT...")
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    all_probs.append(probs)
    all_preds.append(preds)

all_probs = np.vstack(all_probs)
all_preds = np.concatenate(all_preds)

df["pred_label"] = all_preds
df["pred_prob_max"] = all_probs.max(axis=1)
df["benign_prob"] = all_probs[:, 0]
df["anomaly_score"] = 1.0 - df["benign_prob"]  # high anomaly = low benign prob


# --------------------------------------------------
# 2. Mark anomalous events
# --------------------------------------------------
print("Computing anomaly threshold...")
# Top 5% as anomalous
threshold = df["anomaly_score"].quantile(0.95)
df["is_anom"] = df["anomaly_score"] >= threshold
print(f"Anomaly threshold={threshold:.4f}, anomalous events={df['is_anom'].sum()}")

# Sort by timestamp for reconstruction
df = df.sort_values("timestamp_dt").reset_index(drop=True)


# Helper: define session / host grouping
def build_session_id(row):
    parts = []
    if "user" in row and pd.notna(row["user"]):
        parts.append(str(row["user"]))
    if "src_ip" in row and pd.notna(row["src_ip"]):
        parts.append(str(row["src_ip"]))
    if "dst_ip" in row and pd.notna(row["dst_ip"]):
        parts.append(str(row["dst_ip"]))
    if not parts:
        parts.append("global")
    return "|".join(parts)


df["session_id"] = df.apply(build_session_id, axis=1)


# --------------------------------------------------
# Method C — Greedy Decoding of Transitions
# --------------------------------------------------
def greedy_chains_for_session(session_df, lookahead_events=50, p_min=0.4):
    """
    For each anomalous seed in this session, greedily follow the most likely
    next anomalous event in time based on anomaly_score & temporal closeness.
    """
    chains = []

    # work with indices relative to session_df
    anom_idxs = session_df.index[session_df["is_anom"]].tolist()
    if not anom_idxs:
        return chains

    for seed_idx in anom_idxs:
        chain = [seed_idx]
        current_idx = seed_idx
        total_score = session_df.loc[seed_idx, "anomaly_score"]

        while True:
            current_time = session_df.loc[current_idx, "timestamp_dt"]
            # candidate future events
            future_mask = (session_df.index > current_idx)
            future_idxs = session_df.index[future_mask].tolist()

            if not future_idxs:
                break

            # restrict by lookahead
            future_idxs = future_idxs[:lookahead_events]

            best_idx = None
            best_score = 0.0

            for j in future_idxs:
                # prefer anomalous, but allow high anomaly score
                score = session_df.loc[j, "anomaly_score"]

                # simple time decay
                dt = (session_df.loc[j, "timestamp_dt"] - current_time).total_seconds()
                if dt < 0:
                    continue
                time_decay = math.exp(-dt / 3600.0)  # 1-hour scale

                score *= time_decay

                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx is None or best_score < p_min:
                break

            chain.append(best_idx)
            total_score += best_score
            current_idx = best_idx

        if len(chain) > 1:
            chains.append(
                {
                    "indices": chain,
                    "score": total_score,
                }
            )

    # Deduplicate chains (same starting point)
    # Keep only top chain by score per seed
    unique = {}
    for ch in chains:
        seed = ch["indices"][0]
        if seed not in unique or unique[seed]["score"] < ch["score"]:
            unique[seed] = ch

    return list(unique.values())


def method_c_reconstruction(df):
    print("\n=== Method C: Greedy Decoding Reconstruction ===")
    all_chains = []
    chain_id = 1

    for session_id, session_df in df.groupby("session_id"):
        session_df = session_df.sort_values("timestamp_dt")
        chains = greedy_chains_for_session(session_df)

        for ch in chains:
            idxs = ch["indices"]
            events = session_df.loc[idxs]

            chain_entry = {
                "chain_id": chain_id,
                "method": "greedy_transition",
                "session_id": session_id,
                "events": events["event_id"].tolist(),
                "start_time": events["timestamp_dt"].iloc[0].isoformat(),
                "end_time": events["timestamp_dt"].iloc[-1].isoformat(),
                "score": float(ch["score"]),
                "evidence": {
                    ev_id: {
                        "anomaly_score": float(events.loc[events["event_id"] == ev_id, "anomaly_score"].iloc[0]),
                        "pred_label": int(events.loc[events["event_id"] == ev_id, "pred_label"].iloc[0]),
                    }
                    for ev_id in events["event_id"].tolist()
                },
            }

            all_chains.append(chain_entry)
            chain_id += 1

    print(f"Method C produced {len(all_chains)} chains.")
    return all_chains


# --------------------------------------------------
# Method D — Graph Reconstruction Using Correlations
# --------------------------------------------------
def method_d_reconstruction(df, max_events=500, time_window_minutes=60.0):
    print("\n=== Method D: Graph Reconstruction (Correlation Graph) ===")

    # Focus on anomalous events only to keep graph manageable
    anom_df = df[df["is_anom"]].copy()
    if len(anom_df) > max_events:
        # take top max_events by anomaly_score
        anom_df = anom_df.nlargest(max_events, "anomaly_score")

    anom_df = anom_df.sort_values("timestamp_dt")
    nodes = anom_df["event_id"].tolist()

    # Precompute
    times = anom_df["timestamp_dt"].tolist()
    scores = anom_df["anomaly_score"].values
    users = anom_df["user"].astype(str).tolist() if "user" in anom_df.columns else [""] * len(anom_df)
    src_ips = anom_df["src_ip"].astype(str).tolist() if "src_ip" in anom_df.columns else [""] * len(anom_df)
    dst_ips = anom_df["dst_ip"].astype(str).tolist() if "dst_ip" in anom_df.columns else [""] * len(anom_df)

    # Build graph
    if HAS_NX:
        G = nx.DiGraph()
    else:
        G = {"nodes": set(), "edges": []}

    n = len(nodes)
    tau = time_window_minutes * 60.0  # seconds

    for i in range(n):
        for j in range(i + 1, n):
            dt = (times[j] - times[i]).total_seconds()
            if dt < 0 or dt > tau:
                continue

            # correlation score components
            score_i = scores[i]
            score_j = scores[j]
            anomaly_component = 0.5 * (score_i + score_j)

            # time decay
            time_component = math.exp(-dt / tau)

            # entity bonus
            same_entity = 0
            if users[i] == users[j] and users[i] != "":
                same_entity += 0.5
            if src_ips[i] == src_ips[j] and src_ips[i] != "":
                same_entity += 0.3
            if dst_ips[i] == dst_ips[j] and dst_ips[i] != "":
                same_entity += 0.2

            corr_score = anomaly_component * 0.6 + time_component * 0.2 + same_entity * 0.2

            if corr_score < 0.2:
                continue

            u = nodes[i]
            v = nodes[j]

            if HAS_NX:
                G.add_node(u)
                G.add_node(v)
                G.add_edge(u, v, weight=corr_score)
            else:
                G["nodes"].add(u)
                G["nodes"].add(v)
                G["edges"].append((u, v, corr_score))

    # Find best path (max sum of weights on DAG by time order)
    # We'll use simple DP over j > i ordering
    idx_map = {ev_id: k for k, ev_id in enumerate(nodes)}
    best_score = {}
    predecessor = {}

    for i, u in enumerate(nodes):
        best_score[u] = scores[i]  # base on its own anomaly
        predecessor[u] = None

    if HAS_NX:
        for u, v, data in G.edges(data=True):
            w = data["weight"]
            if best_score[u] + w > best_score[v]:
                best_score[v] = best_score[u] + w
                predecessor[v] = u
    else:
        for (u, v, w) in G["edges"]:
            if best_score[u] + w > best_score[v]:
                best_score[v] = best_score[u] + w
                predecessor[v] = u

    # pick endpoint with max score
    if not best_score:
        print("No edges created in Method D.")
        return []

    end_node = max(best_score, key=lambda k: best_score[k])

    # backtrack
    path = []
    cur = end_node
    while cur is not None:
        path.append(cur)
        cur = predecessor[cur]
    path = list(reversed(path))

    events_path = anom_df.set_index("event_id").loc[path]

    chain = {
        "chain_id": 1,
        "method": "graph_correlation",
        "session_id": "mixed" if len(events_path["session_id"].unique()) > 1 else events_path["session_id"].iloc[0],
        "events": events_path.index.tolist(),
        "start_time": events_path["timestamp_dt"].iloc[0].isoformat(),
        "end_time": events_path["timestamp_dt"].iloc[-1].isoformat(),
        "score": float(best_score[end_node]),
        "evidence": {
            ev_id: {
                "anomaly_score": float(events_path.loc[ev_id, "anomaly_score"]),
                "pred_label": int(events_path.loc[ev_id, "pred_label"]),
            }
            for ev_id in events_path.index.tolist()
        },
    }

    print(f"Method D produced 1 main chain with {len(path)} events.")
    return [chain], events_path


# --------------------------------------------------
# Visualization: timeline & graph
# --------------------------------------------------
def plot_timeline(events_path: pd.DataFrame, filename: str):
    plt.figure(figsize=(10, 2.5))
    times = events_path["timestamp_dt"]
    y = [1] * len(times)
    labels = events_path["pred_label"].tolist()

    plt.scatter(times, y, c=labels, cmap="viridis", s=40)
    for t, lbl, ev_id in zip(times, labels, events_path.index.tolist()):
        plt.text(t, 1.02, str(lbl), rotation=90, fontsize=6)

    plt.yticks([])
    plt.xlabel("Time")
    plt.title("Reconstructed Attack Chain Timeline (Method D)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def plot_graph(events_path: pd.DataFrame, filename: str):
    events = events_path.index.tolist()
    if not HAS_NX:
        print("networkx not available; skipping graph plot.")
        return

    G = nx.DiGraph()
    for i in range(len(events) - 1):
        G.add_edge(events[i], events[i + 1])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=6, arrows=True)
    plt.title("Reconstructed Attack Chain Graph (Method D)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    # Method C: greedy chains per session
    chains_c = method_c_reconstruction(df)

    # Method D: graph-based chain
    chains_d, events_path_d = method_d_reconstruction(df)

    # Combine chains into one JSON file
    all_chains = []
    chain_id_offset = 1
    # renumber C chains
    for i, ch in enumerate(chains_c, start=1):
        ch["chain_id"] = chain_id_offset + i
        all_chains.append(ch)
    # keep D's chain_id as 1
    all_chains = chains_d + all_chains

    json_path = OUT_DIR / "reconstructed_chains.json"
    with open(json_path, "w") as f:
        json.dump(all_chains, f, indent=2)

    print(f"\nSaved reconstructed chains to {json_path}")

    # Visualizations for the Method D main chain
    plot_timeline(events_path_d, "chain_1_timeline.png")
    plot_graph(events_path_d, "chain_1_graph.png")

    print("Saved:")
    print(f" - {OUT_DIR / 'reconstructed_chains.json'}")
    print(f" - {OUT_DIR / 'chain_1_timeline.png'}")
    print(f" - {OUT_DIR / 'chain_1_graph.png'}")
