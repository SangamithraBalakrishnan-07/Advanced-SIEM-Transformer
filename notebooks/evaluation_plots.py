import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

NUM_CLASSES = 6  # benign, recon, exploit, priv_esc, lateral, exfil

OUT_DIR = Path("evaluation_outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# DATASETS
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL 1: Transformer Encoder Classifier (same as before)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, num_heads=4, num_layers=4, num_classes=6, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


# ============================================================
# HELPER: Metrics + ROC + Confusion Heatmap
# ============================================================

def compute_basic_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec, f1


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, np.arange(NUM_CLASSES))
    plt.yticks(tick_marks, np.arange(NUM_CLASSES))
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add labels
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def plot_roc_curve(y_true, y_proba, title, filename):
    # y_true: (N,)
    # y_proba: (N, num_classes)
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def compute_correlation_accuracy(y_true, y_pred):
    """
    Approximate "correlation accuracy" as how well the model preserves
    the direction of stage transitions between consecutive items.
    1 if model predicts increase/no-change when true also increases/no-change; 0 otherwise.
    """
    if len(y_true) < 2:
        return np.nan
    true_trend = (np.diff(y_true) >= 0).astype(int)
    pred_trend = (np.diff(y_pred) >= 0).astype(int)
    return np.mean(true_trend == pred_trend)


# ============================================================
# EVALUATE MODEL 1 (SEQUENCE TRANSFORMER)
# ============================================================

def evaluate_model1():
    print("\n=== Evaluating Model 1 (Transformer Encoder Classifier) ===")
    seq_path = Path("processed/sequences.npy")
    lab_path = Path("processed/labels.npy")

    X = np.load(seq_path)
    y = np.load(lab_path)

    # Same split as transformer_classifier.py (first 80% train, last 20% val)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    val_dataset = SequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = TransformerClassifier(
        input_dim=20,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=NUM_CLASSES,
        dropout=0.1,
    ).to(DEVICE)

    state_dict = torch.load("transformer_classifier.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_probs = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(y_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    acc, prec, rec, f1 = compute_basic_metrics(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Sequence reconstruction accuracy (here = classification accuracy per sequence)
    seq_recon_acc = acc

    # Correlation accuracy: trend consistency
    corr_acc = compute_correlation_accuracy(all_true, all_preds)

    print(f"Sequence Reconstruction Accuracy (Model 1): {seq_recon_acc:.4f}")
    print(f"Correlation Accuracy (Model 1):           {corr_acc:.4f}")

    # Save confusion heatmap & ROC curve
    plot_confusion_matrix(cm, "Model 1 - Transformer Confusion Matrix", "confusion_model1.png")
    plot_roc_curve(all_true, all_probs, "Model 1 - Transformer ROC Curve", "roc_model1.png")

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "seq_recon_acc": seq_recon_acc,
        "corr_acc": corr_acc,
    }


# ============================================================
# EVALUATE MODEL 2 (Log-BERT / logbert-cls)
# ============================================================

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


def evaluate_model2():
    print("\n=== Evaluating Model 2 (Log-BERT style) ===")

    df = pd.read_csv("advanced_siem_labeled.csv")
    df["text"] = df.apply(build_log_text, axis=1)
    labels = df["label"].values

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=labels
    )

    test_texts = test_df["text"].tolist()
    y_true = test_df["label"].values

    tokenizer = AutoTokenizer.from_pretrained("logbert-cls")
    model = AutoModelForSequenceClassification.from_pretrained("logbert-cls").to(DEVICE)
    model.eval()

    all_probs = []
    all_preds = []

    batch_size = 16
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i : i + batch_size]
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

    acc, prec, rec, f1 = compute_basic_metrics(y_true, all_preds)
    cm = confusion_matrix(y_true, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Sequence reconstruction accuracy = event-level classification accuracy
    seq_recon_acc = acc

    # Correlation accuracy: trend consistency across test set
    corr_acc = compute_correlation_accuracy(y_true, all_preds)

    print(f"Sequence Reconstruction Accuracy (Model 2): {seq_recon_acc:.4f}")
    print(f"Correlation Accuracy (Model 2):           {corr_acc:.4f}")

    # Save confusion heatmap & ROC curve
    plot_confusion_matrix(cm, "Model 2 - LogBERT Confusion Matrix", "confusion_model2.png")
    plot_roc_curve(y_true, all_probs, "Model 2 - LogBERT ROC Curve", "roc_model2.png")

    # Attention heatmap (for a single example)
    print("Generating attention heatmap for Model 2...")
    sample_text = test_texts[0]
    enc = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    attentions = outputs.attentions  # tuple (num_layers, batch, num_heads, seq_len, seq_len)
    att_stack = torch.stack(attentions)  # (L, B, H, S, S)
    att_agg = att_stack.mean(dim=2).mean(dim=0)  # average over heads, then layers -> (B, S, S)
    att_matrix = att_agg[0].cpu().numpy()

    # Crop to first 30 tokens for readability
    max_tokens = 30
    att_crop = att_matrix[:max_tokens, :max_tokens]

    plt.figure(figsize=(6, 5))
    plt.imshow(att_crop, interpolation="nearest")
    plt.title("Model 2 - Attention Heatmap (first 30 tokens)")
    plt.colorbar()
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "attention_heatmap_model2.png")
    plt.close()

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "seq_recon_acc": seq_recon_acc,
        "corr_acc": corr_acc,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting evaluation...")

    res1 = evaluate_model1()
    res2 = evaluate_model2()

    # Save summary
    summary = pd.DataFrame(
        [
            ["Model 1 - Transformer", res1["acc"], res1["prec"], res1["rec"], res1["f1"], res1["seq_recon_acc"], res1["corr_acc"]],
            ["Model 2 - LogBERT", res2["acc"], res2["prec"], res2["rec"], res2["f1"], res2["seq_recon_acc"], res2["corr_acc"]],
        ],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Seq_Recon_Acc", "Corr_Acc"],
    )

    summary.to_csv(OUT_DIR / "model_comparison_metrics.csv", index=False)
    print("\nSaved model comparison metrics to evaluation_outputs/model_comparison_metrics.csv")
    print("Saved plots in evaluation_outputs/:")
    print(" - confusion_model1.png")
    print(" - confusion_model2.png")
    print(" - roc_model1.png")
    print(" - roc_model2.png")
    print(" - attention_heatmap_model2.png")
