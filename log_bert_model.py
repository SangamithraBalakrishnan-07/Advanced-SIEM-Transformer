import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

CSV_PATH = "advanced_siem_labeled.csv"
NUM_LABELS = 6          # benign, recon, exploit, priv_esc, lateral, exfil
MAX_LEN = 128           # BERT max sequence length
MLM_EPOCHS = 2          # you can increase to 3–5 if you have time
CLS_EPOCHS = 3

# --------------------------------------------------
# 1. LOAD LABELED DATASET
# --------------------------------------------------
print("Loading labeled dataset...")
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)

# Build a log-text column (you can tweak what goes in here)
def build_log_text(row):
    parts = []
    parts.append(f"[EVENT_TYPE] {row.get('event_type', '')}")
    parts.append(f"[SEVERITY] {row.get('severity', '')}")
    parts.append(f"[SOURCE] {row.get('source', '')}")
    if 'user' in row and pd.notna(row['user']):
        parts.append(f"[USER] {row['user']}")
    if 'src_ip' in row and pd.notna(row['src_ip']):
        parts.append(f"[SRC_IP] {row['src_ip']}")
    if 'dst_ip' in row and pd.notna(row['dst_ip']):
        parts.append(f"[DST_IP] {row['dst_ip']}")
    parts.append(f"[DESC] {row.get('description', '')}")
    if 'additional_info' in row and pd.notna(row['additional_info']):
        parts.append(f"[INFO] {row['additional_info']}")
    return " ".join(parts)

df["text"] = df.apply(build_log_text, axis=1)

# labels already created in previous steps: 0–5
labels = df["label"].values

# Train/test split at EVENT level for BERT model
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=labels
)

print("Train size:", train_df.shape, "Test size:", test_df.shape)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# --------------------------------------------------
# 2. TOKENIZER
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

# For MLM we only need input_ids & attention_mask
mlm_dataset = train_dataset.map(tokenize_function, batched=True)
mlm_dataset = mlm_dataset.remove_columns(
    [c for c in mlm_dataset.column_names if c not in ["input_ids", "attention_mask"]]
)

# For classification we keep labels
def add_labels(examples):
    return {"labels": examples["label"]}

cls_train_dataset = train_dataset.map(tokenize_function, batched=True)
cls_train_dataset = cls_train_dataset.map(add_labels, batched=True)

cls_test_dataset = test_dataset.map(tokenize_function, batched=True)
cls_test_dataset = cls_test_dataset.map(add_labels, batched=True)

cls_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
cls_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --------------------------------------------------
# 3. STAGE 1: MLM PRETRAINING ON LOG TEXT
# --------------------------------------------------
print("\n=== Stage 1: Masked Language Modeling pretraining ===")

data_collator_mlm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)

model_mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model_mlm.to(DEVICE)

training_args_mlm = TrainingArguments(
    output_dir="logbert-mlm",
    overwrite_output_dir=True,
    num_train_epochs=MLM_EPOCHS,
    per_device_train_batch_size=16,
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
)

trainer_mlm = Trainer(
    model=model_mlm,
    args=training_args_mlm,
    train_dataset=mlm_dataset,
    data_collator=data_collator_mlm,
)

trainer_mlm.train()
trainer_mlm.save_model("logbert-mlm")
tokenizer.save_pretrained("logbert-mlm")

print("MLM pretraining complete. Saved model in logbert-mlm/")


# --------------------------------------------------
# 4. STAGE 2: CLASSIFICATION FINE-TUNING
# --------------------------------------------------
print("\n=== Stage 2: Fine-tuning for attack-stage classification ===")

model_cls = AutoModelForSequenceClassification.from_pretrained(
    "logbert-mlm",
    num_labels=NUM_LABELS,
)
model_cls.to(DEVICE)


training_args_cls = TrainingArguments(
    output_dir="logbert-cls",
    overwrite_output_dir=True,
    num_train_epochs=CLS_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    evaluation_strategy="epoch",   # evaluate every epoch
    save_strategy="epoch",         # SAVE every epoch (matches eval)
    
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    
    load_best_model_at_end=True,   # works now
    metric_for_best_model="f1",
    
    report_to="none",
)


# Metrics function
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

trainer_cls = Trainer(
    model=model_cls,
    args=training_args_cls,
    train_dataset=cls_train_dataset,
    eval_dataset=cls_test_dataset,
    compute_metrics=compute_metrics,
)

trainer_cls.train()
eval_results = trainer_cls.evaluate()
print("\n=== Classification Evaluation (BERT-style Log Model) ===")
print(eval_results)

# --------------------------------------------------
# 5. CONFUSION MATRIX (DETAILED)
# --------------------------------------------------
print("\nComputing confusion matrix...")
predictions = trainer_cls.predict(cls_test_dataset)
logits = predictions.predictions
true_labels = predictions.label_ids
pred_labels = np.argmax(logits, axis=-1)

cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:\n", cm)

# Save model for later (e.g., reconstruction module)
trainer_cls.save_model("logbert-cls")
print("\nModel 2 (Log-BERT style) saved in logbert-cls/")
