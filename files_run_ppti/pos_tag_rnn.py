from preprocessing_class import Preprocessing, load_pres
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import (
    f1_score, average_precision_score,
    roc_auc_score, precision_score, recall_score
)
import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import json
from datetime import datetime

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_TYPE = "gru" # "lstm"

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────

alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)
alllabs = np.where(alllabs == 1, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    alltxt, alllabs,
    test_size=0.20,
    stratify=alllabs,
    random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)

# ─────────────────────────────────────────
# POS TAGGING
# ─────────────────────────────────────────

punc = set(string.punctuation + '\n\r\t')
punc.remove("'")
custom_punctuation = "".join(punc)

prep_pos = Preprocessing(
    low_case=False,
    rm_punctuation=False,
    rm_number=False,
    word_norm=None,
    pos_tagging=True,
    all_capital=True,
    cap_name=True,
    rm_accent=False,
    lang="french",
    punct=custom_punctuation,
    urls=False
)

print("POS tagging train...")
X_train_pos = [[p for (t, p) in prep_pos.process(t)[1]] for t in X_train_sub]
print("POS tagging val...")
X_val_pos   = [[p for (t, p) in prep_pos.process(t)[1]] for t in X_val]
print("POS tagging test...")
X_test_pos  = [[p for (t, p) in prep_pos.process(t)[1]] for t in X_test]

# ─────────────────────────────────────────
# VOCAB — built from train only
# ─────────────────────────────────────────

def build_vocab(all_tag_sequences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tags in all_tag_sequences:
        for tag in tags:
            if tag not in vocab:
                vocab[tag] = len(vocab)
    return vocab

vocab = build_vocab(X_train_pos)
print(f"Vocab size: {len(vocab)}")

# ─────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────

class POSDataset(Dataset):
    def __init__(self, tag_sequences, labels, vocab):
        self.labels = labels
        self.vocab  = vocab
        self.encoded = [
            torch.tensor([vocab.get(t, vocab["<UNK>"]) for t in seq], dtype=torch.long)
            for seq in tag_sequences
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded  = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


dataset_train    = POSDataset(X_train_pos, y_train_sub, vocab)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True,  collate_fn=collate_fn)

dataset_val      = POSDataset(X_val_pos, y_val, vocab)
dataloader_val   = DataLoader(dataset_val,   batch_size=32, shuffle=False, collate_fn=collate_fn)

dataset_test     = POSDataset(X_test_pos, y_test, vocab)                   # fix: was using X_val_pos/y_val
dataloader_test  = DataLoader(dataset_test,  batch_size=32, shuffle=False, collate_fn=collate_fn)

print("Datasets ready")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

class POSClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if MODEL_TYPE == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        
        else:
    
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb    = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if MODEL_TYPE == "gru":
            _, hn = self.rnn(packed)
        else:
            _, (hn, _) = self.rnn(packed)
        hn     = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(self.dropout(hn))


model = POSClassifier(
    vocab_size=len(vocab),
    embed_dim=32,
    hidden_dim=64,
    num_classes=2,
    num_layers=2,
    dropout=0.3,
).to(device)                          # move model to device

print(model)

# ─────────────────────────────────────────
# LOSS & OPTIMIZER
# ─────────────────────────────────────────

def compute_class_weights(labels):
    counts  = torch.bincount(torch.tensor(labels)).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights

class_weights = compute_class_weights(y_train_sub).to(device)  # fix: use train_sub, move to device
criterion     = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    return {
        "f1":            float(f1_score(all_labels, all_preds, average="binary")),
        "avg_precision": float(average_precision_score(all_labels, all_probs)),
        "roc_auc":       float(roc_auc_score(all_labels, all_probs)),
        "precision":     float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":        float(recall_score(all_labels, all_preds)),
    }

# ─────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, training=True):
    model.train() if training else model.eval()

    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for padded, lengths, labels in loader:
            padded = padded.to(device)       # move batch to device
            labels = labels.to(device)       # lengths stays on CPU for pack_padded_sequence

            if training:
                optimizer.zero_grad()

            logits = model(padded, lengths)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            all_labels .extend(labels.cpu().numpy())
            all_preds  .extend(preds.cpu().numpy())
            all_probs  .extend(probs[:, 1].detach().cpu().numpy())

    metrics         = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = float(total_loss / len(loader))
    return metrics

# ─────────────────────────────────────────
# TRAINING + SAVING RESULTS
# ─────────────────────────────────────────

history        = []          # one entry per epoch
best_f1        = 0.0
best_model_state = None

for epoch in range(10):
    train_m = run_epoch(model, dataloader_train, optimizer, criterion, training=True)
    val_m   = run_epoch(model, dataloader_val,   optimizer, criterion, training=False)

    if val_m["f1"] > best_f1:
        best_f1          = val_m["f1"]
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # store epoch results
    history.append({
        "epoch": epoch + 1,
        "train": train_m,
        "val":   val_m,
    })

    print(
        f"Epoch {epoch+1:2d} | "
        f"loss  train {train_m['loss']:.4f}  val {val_m['loss']:.4f} | "
        f"f1    train {train_m['f1']:.3f}  val {val_m['f1']:.3f} | "
        f"auc   train {train_m['roc_auc']:.3f}  val {val_m['roc_auc']:.3f} | "
        f"ap    train {train_m['avg_precision']:.3f}  val {val_m['avg_precision']:.3f} | "
        f"prec  train {train_m['precision']:.3f}  val {val_m['precision']:.3f} | "
        f"rec   train {train_m['recall']:.3f}  val {val_m['recall']:.3f}"
    )

print(f"\nBest val F1: {best_f1:.3f}")

# ─────────────────────────────────────────
# TEST SET EVALUATION
# ─────────────────────────────────────────

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "model_rnn_pos_tag.pth")

test_m = run_epoch(model, dataloader_test, optimizer, criterion, training=False)

print(
    f"\nTest results | "
    f"loss {test_m['loss']:.4f} | "
    f"f1 {test_m['f1']:.3f} | "
    f"auc {test_m['roc_auc']:.3f} | "
    f"ap {test_m['avg_precision']:.3f} | "
    f"prec {test_m['precision']:.3f} | "
    f"rec {test_m['recall']:.3f}"
)

# ─────────────────────────────────────────
# SAVE ALL RESULTS TO JSON
# ─────────────────────────────────────────

results = {
    "run_date":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device":    str(device),
    "best_val_f1": best_f1,
    "history":   history,           # all epochs, train + val metrics
    "test":      test_m,            # final test metrics
}

with open("results_rnn_pos.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results_rnn_pos.json")