"""
BiLSTM Post-Processing Model for Probability Correction
--------------------------------------------------------
Takes raw classifier probabilities [T] and learns to push them
toward clean 0/1 binary labels using a Bidirectional LSTM.

Input:  probabilities array of shape [T]  (one value per frame)
Output: corrected probabilities saved to .npy file, close to 0 or 1
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32    # number of chunks per batch
EARLY_STOPPING_PATIENCE = 10
CHUNK_SIZE = 30   # split the sequence into chunks of this size for training

PROBS_FILE  = "probs_all_train_smoothed.npy"
LABELS_FILE = "alllabs.npy"

SAVING_FILE_NAME = "bilstm_prob_corrector_all_train_smoothed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChunkedSequenceDataset(Dataset):
    """
    Splits a single long sequence into fixed-size chunks.
    Each chunk: probs [1, chunk_size], labels [chunk_size]
    """
    def __init__(self, probs, labels, chunk_size):
        assert len(probs) == len(labels)
        self.chunks_p = []
        self.chunks_l = []

        for start in range(0, len(probs) - chunk_size + 1, chunk_size):
            p = probs[start:start + chunk_size]
            l = labels[start:start + chunk_size]
            self.chunks_p.append(torch.tensor(p, dtype=torch.float32).unsqueeze(0))  # [1, chunk_size]
            self.chunks_l.append(torch.tensor(l, dtype=torch.float32))               # [chunk_size]

    def __len__(self):
        return len(self.chunks_p)

    def __getitem__(self, idx):
        return self.chunks_p[idx], self.chunks_l[idx]

class LSTMpostproc(nn.Module):
    """
    Bidirectional LSTM: raw prob sequence → corrected logits.
    Input:  [B, 1, T]
    Output: [B, T]
    """
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        x = x.permute(0, 2, 1)          # [B, 1, T] → [B, T, 1]
        out, _ = self.lstm(x)            # [B, T, hidden*2]
        out = self.dropout(out)
        logits = self.head(out)          # [B, T, 1]
        return logits.squeeze(-1)        # [B, T]


def compute_metrics(all_labels, all_probs):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs > 0.5).astype(int)
    return {
        "f1":          float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "roc_auc":     float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.0,
        "avg_precision": float(average_precision_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.0,
        "precision":  float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall":     float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
    }

def run_epoch(model, loader, optimizer, criterion, training=True):
    model.train() if training else model.eval()

    total_loss = 0
    all_labels, all_probs = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for probs, labels in loader:
            probs  = probs.to(device)   # [B, 1, chunk_size]
            labels = labels.to(device)  # [B, chunk_size]

            if training:
                optimizer.zero_grad()

            logits = model(probs)            # [B, chunk_size]
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            corrected = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(corrected.detach().cpu().numpy().flatten().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader)
    return metrics

if __name__ == "__main__":
    print("Loading data")
    all_probs  = np.load(PROBS_FILE).astype(np.float32)   # [T]
    all_labels = np.load(LABELS_FILE).astype(np.float32)  # [T]
    print(f"Loaded {len(all_probs)} frames | class 0: {int((all_labels==0).sum())} | class 1: {int((all_labels==1).sum())}")


    train_probs, val_probs, train_labels, val_labels = train_test_split(
        all_probs, all_labels,
        test_size=0.2,
        random_state=42,
        shuffle=False 
    )

    train_dataset = ChunkedSequenceDataset(train_probs, train_labels, CHUNK_SIZE)
    val_dataset   = ChunkedSequenceDataset(val_probs, val_labels, CHUNK_SIZE)

    print(f"Train chunks: {len(train_dataset)} | Val chunks: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMpostproc(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(model)

    counts = np.bincount(train_labels.astype(int))  # [count_0, count_1]
    weights = 1.0 / counts                           # [w0, w1]
    pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32).to(device)
    print(f"Class counts: {counts} | pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_m = run_epoch(model, train_loader, optimizer, criterion, training=True)
        val_m   = run_epoch(model, val_loader,   optimizer, criterion, training=False)

        improved = val_m["f1"] > best_f1
        if improved:
            best_f1          = val_m["f1"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "✓ improved"
        else:
            patience_counter += 1
            marker = f"patience {patience_counter}/{EARLY_STOPPING_PATIENCE}"

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS}  {marker:30s} | "
            f"loss train {train_m['loss']:.4f}  val {val_m['loss']:.4f} | "
            f"f1 train {train_m['f1']:.3f}  val {val_m['f1']:.3f} | "
            f"auc train {train_m['roc_auc']:.3f}  val {val_m['roc_auc']:.3f}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest val F1: {best_f1:.3f}")

    # ── Save model ────────────────────────────
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"{SAVING_FILE_NAME}.pth")
    print(f"Model saved to {SAVING_FILE_NAME}.pth")

    # ── Inference on full sequence ─────────────
    print("\nRunning inference on full sequence...")
    model.eval()
    os.makedirs("corrected_outputs", exist_ok=True)

    with torch.no_grad():
        # Pass the full sequence at once — no chunking needed at inference
        probs_tensor = torch.tensor(all_probs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
        logits       = model(probs_tensor)                                    # [1, T]
        corrected    = torch.sigmoid(logits).squeeze().cpu().numpy()          # [T]
        preds        = (corrected > 0.5).astype(int)

    out_path = f"corrected_outputs/{SAVING_FILE_NAME}.npy"
    np.save(out_path, corrected)

    print(f"  shape:     {corrected.shape}")
    print(f"  mean prob: {corrected.mean():.3f}")
    print(f"  pred 1s:   {preds.sum()}/{len(preds)}")
    print(f"  true 1s:   {int(all_labels.sum())}/{len(all_labels)}")
    print(f"  saved →    {out_path}")
    print("\nDone.")