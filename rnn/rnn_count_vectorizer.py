"""
RNN sentiment classification using CountVectorizer
Tests a single configuration with clean evaluation structure
"""

# 2 is corrected function

import os
import gc
import json
import codecs
import re
import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score, 
    precision_score, recall_score, accuracy_score
)
from datetime import datetime

from preprocessing_class import Preprocessing, load_movies, load_pres
from nltk.corpus import stopwords

final_stopwords_list = stopwords.words('french')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
VECTORIZER_TYPE = "count"  # 'count' or 'tfidf'
RNN_TYPE = 'gru'  # 'lstm' or 'gru'
DATASET = "pres" # "movie"
SAVING_FILE_NAME =  DATASET + "_rnn_" + RNN_TYPE + " " + VECTORIZER_TYPE

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 5
DROPOUT = 0.4

# Vectorizer config
MAX_FEATURES = 10000 # changed to 10000!
NGRAM_RANGE = (1, 1)  # (1, 1) unigram, (1, 2) bigram

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU memory available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

print("Loading data...")
if DATASET == "pres":
    alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
    alltxt = np.array(alltxt)
    # Convert labels: 1 -> 0, -1 -> 1
    alllabs = np.where(np.array(alllabs) == 1, 0, 1)
    print(f"  Total samples: {len(alltxt)}")
    print(f"  Label distribution: {np.bincount(alllabs)}")
else:
    path = "./Dataset/movies1000/"
    alltxt,alllabs = load_movies(path)
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)
    print(f"  Total samples: {len(alltxt)}")
    print(f"  Label distribution: {np.bincount(alllabs)}")


# Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    alltxt, alllabs,
    test_size=TEST_SIZE,
    stratify=alllabs,
    random_state=RANDOM_STATE
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train,
    test_size=VAL_SIZE,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print(f"  Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

print("\nSetting up preprocessing...")

punc = set(string.punctuation + '\n\r\t')
punc.discard("'")
custom_punctuation = "".join(punc)

lang = "french" if DATASET == "pres" else "english"

prep = Preprocessing(
    low_case=True,
    rm_punctuation=True,
    rm_number=False,
    word_norm=None,
    pos_tagging=False,
    all_capital=True,
    cap_name=True,
    rm_accent=False,
    lang=lang,
    punct=custom_punctuation,
    urls=False
)

print("Preprocessing train...")
X_train_prep = [prep.process(t) for t in X_train_sub]
print("Preprocessing val...")
X_val_prep = [prep.process(t) for t in X_val]
print("Preprocessing test...")
X_test_prep = [prep.process(t) for t in X_test]

# ─────────────────────────────────────────
# VECTORIZATION
# ─────────────────────────────────────────

print(f"\nVectorizing with {VECTORIZER_TYPE}...")

if VECTORIZER_TYPE == "tfidf":
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        max_df=0.95,
        min_df=2,
        ngram_range=NGRAM_RANGE,
        stop_words=final_stopwords_list
    )
else:
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        max_df=0.95,
        min_df=2,
        ngram_range=NGRAM_RANGE,
        stop_words=final_stopwords_list
    )

X_train_vec = vectorizer.fit_transform(X_train_prep)
X_val_vec = vectorizer.transform(X_val_prep)
X_test_vec = vectorizer.transform(X_test_prep)

print(f"  Vocabulary size: {X_train_vec.shape[1]}")

# ─────────────────────────────────────────
# CONVERT TO SEQUENCES
# ─────────────────────────────────────────

def sparse_to_sequences_preserve_order(X_sparse, shift=2):
    """
    Convert sparse matrix to sequences of feature indices while preserving word order.
    Each non-zero feature index appears in its original position in the document.
    """
    sequences = []
    for i in range(X_sparse.shape[0]):
        row = X_sparse[i].toarray().flatten()
        # Get all non-zero feature indices in their original order
        feature_indices = np.where(row > 0)[0]
        shifted_indices = (feature_indices + shift).tolist() if len(feature_indices) > 0 else [0]
        sequences.append(shifted_indices)
    return sequences

print("Converting to sequences...")
X_train_seqs = sparse_to_sequences_preserve_order(X_train_vec)
X_val_seqs = sparse_to_sequences_preserve_order(X_val_vec)
X_test_seqs = sparse_to_sequences_preserve_order(X_test_vec)

# ─────────────────────────────────────────
# VOCAB
# ─────────────────────────────────────────

vocab = {i: i for i in range(X_train_vec.shape[1] + 2)}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

print(f"Vocab size: {len(vocab)}")

# ─────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────

def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, vocab):
        self.labels = labels
        self.vocab = vocab
        self.encoded = [
            torch.tensor([vocab.get(t, vocab["<UNK>"]) for t in seq], dtype=torch.long)
            for seq in sequences
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


dataset_train = SequenceDataset(X_train_seqs, y_train_sub, vocab)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

dataset_val = SequenceDataset(X_val_seqs, y_val, vocab)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

dataset_test = SequenceDataset(X_test_seqs, y_test, vocab)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("Datasets ready")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, 
                 dropout=0.3, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        
        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'lstm':
            _, (hn, _) = self.rnn(packed)
        else:  # GRU
            _, hn = self.rnn(packed)
        
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # concat forward + backward
        return self.classifier(self.dropout(hn))


print(f"\nLoading model ({RNN_TYPE})...")
model = RNNClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rnn_type=RNN_TYPE,
).to(device)

print(model)

# ─────────────────────────────────────────
# LOSS & OPTIMIZER
# ─────────────────────────────────────────

def compute_class_weights(labels):
    counts = torch.bincount(torch.tensor(labels)).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights

class_weights = compute_class_weights(y_train_sub).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# ─────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute evaluation metrics."""
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    if DATASET == "pres":
        return {
            "f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
            "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(all_labels, all_probs)),
            "avg_precision": float(average_precision_score(all_labels, all_probs)),
        }
    
    return {
        "f1": float(f1_score(all_labels, all_preds, average="binary", zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
    }

# ─────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, training=True):
    """Run one epoch of training or evaluation."""
    model.train() if training else model.eval()

    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for padded, lengths, labels in loader:
            padded = padded.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(padded, lengths)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = float(total_loss / len(loader))
    return metrics

# ─────────────────────────────────────────
# TRAINING WITH EARLY STOPPING
# ─────────────────────────────────────────

print("\n" + "="*80)
print("TRAINING")
print("="*80)

history = []
best_f1 = 0.0
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}", end=" ")

    train_m = run_epoch(model, dataloader_train, optimizer, criterion, training=True)
    val_m = run_epoch(model, dataloader_val, optimizer, criterion, training=False)

    # Early stopping
    if val_m["f1"] > best_f1:
        best_f1 = val_m["f1"]
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        improvement_marker = "↑ (BEST)"
    else:
        patience_counter += 1
        improvement_marker = f"(patience {patience_counter}/{EARLY_STOPPING_PATIENCE})"

    history.append({
        "epoch": epoch + 1,
        "train": train_m,
        "val": val_m,
    })

    print(
        f"{improvement_marker:25s} | "
        f"loss  train {train_m['loss']:.4f}  val {val_m['loss']:.4f} | "
        f"f1    train {train_m['f1']:.3f}  val {val_m['f1']:.3f} | "
        f"auc   train {train_m['roc_auc']:.3f}  val {val_m['roc_auc']:.3f} | "
        f"prec  train {train_m['precision']:.3f}  val {val_m['precision']:.3f} | "
        f"rec   train {train_m['recall']:.3f}  val {val_m['recall']:.3f}"
    )

    # Early stopping check
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
        break

print(f"\nBest val F1: {best_f1:.3f}")

# ─────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────

print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), f"{SAVING_FILE_NAME}.pth")

test_m = run_epoch(model, dataloader_test, optimizer, criterion, training=False)

print(
    f"\nTest | loss {test_m['loss']:.4f} | f1 {test_m['f1']:.4f} | "
    f"prec {test_m['precision']:.4f} | rec {test_m['recall']:.4f} | "
    f"auc {test_m['roc_auc']:.4f} | ap {test_m['avg_precision']:.4f}"
)

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────

results = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device),
    "vectorizer": {
        "type": VECTORIZER_TYPE,
        "max_features": MAX_FEATURES,
        "ngram_range": NGRAM_RANGE,
    },
    "model_config": {
        "rnn_type": RNN_TYPE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 1e-5,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    },
    "best_val_f1": best_f1,
    "epochs_trained": len(history),
    "history": history,
    "test": test_m,
}

with open(f"results_{SAVING_FILE_NAME}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results_{SAVING_FILE_NAME}.json")
print(f"Model saved to {SAVING_FILE_NAME}.pth")

print("DATASET ", DATASET)
print(f"Transformer embedding dimension: {EMBED_DIM}")
print(f"\nLoaded model ({RNN_TYPE})...")
