import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
import math
import argparse

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "processed/"
SEQ_FILE = DATA_PATH + "sequences.npy"
LABEL_FILE = DATA_PATH + "labels.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET
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
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# ============================================================
# TRANSFORMER CLASSIFIER MODEL
# ============================================================

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, num_heads=4, num_layers=4, num_classes=6, dropout=0.1):
        super().__init__()

        # Project 20-d input → hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.attention_weights = None

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Save attention weights
        x = self.transformer(x)

        # Mean pooling over 100 tokens
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, val_loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y_batch.numpy())

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average="weighted", zero_division=0)
    rec = recall_score(trues, preds, average="weighted", zero_division=0)
    f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(trues, preds)

    print("\n=== Evaluation Results ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", cm)

    return acc, prec, rec, f1, cm


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # Load data
    X = np.load(SEQ_FILE)
    y = np.load(LABEL_FILE)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model hyperparameters (tweakable)
    model = TransformerClassifier(
        input_dim=20,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        num_classes=6
    ).to(DEVICE)

    print("Training Transformer model...")
    model = train_model(model, train_loader, val_loader, epochs=5, lr=1e-4)

    print("Evaluating model...")
    evaluate(model, val_loader)

    # Save model
    torch.save(model.state_dict(), "transformer_classifier.pth")
    print("Model saved as transformer_classifier.pth")


if __name__ == "__main__":
    main()
