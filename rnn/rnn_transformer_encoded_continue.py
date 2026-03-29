"""
RNN classification using Transformer-encoded word embeddings - CONTINUE TRAINING.
Loads a saved model and continues training for additional epochs.
Encodes each word using a transformer encoder (intfloat/multilingual-e5-base),
then passes word embeddings to an RNN.
TOKENIZE ENCODE WORDS, PASS THEM TO RNN, EMBEDDINGS SIZE 768
"""

import os
import gc
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import string

from preprocessing_class import Preprocessing, load_pres, load_movies

# CONFIGURATION !!!!

# choose dataset
DATASET = "movies" # "movie"
MAX_LENGHT_TOKEN = 256

# Continue training
CONTINUE_TRAINING = True  # Set to True to load existing model, False to start fresh
ADDITIONAL_EPOCHS = 50

# Same parameters as original training
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10

RNN_TYPE = 'gru'  # 'lstm' or 'gru'
HIDDEN_DIM = 64 # 32
NUM_LAYERS = 2
DROPOUT = 0.4

SAVING_FILE_NAME =  f"{DATASET}_rnn_encoder_{RNN_TYPE}" 

# SAVING_FILE_NAME =  f"{DATASET}_rnn_encoder_{RNN_TYPE}_{MAX_LENGHT_TOKEN}_{LEARNING_RATE}_{BATCH_SIZE}_{HIDDEN_DIM}_{NUM_LAYERS}" 

# Transformer encoder config
TRANSFORMER_MODEL = "intfloat/multilingual-e5-base"  # 768 embeddings
FREEZE_TRANSFORMER = True  # Don't train transformer, only RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU memory available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")


print(f"\nLoading transformer encoder ({TRANSFORMER_MODEL})...")
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
encoder = AutoModel.from_pretrained(TRANSFORMER_MODEL)  # Keep on CPU for encoding phase

if FREEZE_TRANSFORMER:
    for param in encoder.parameters():
        param.requires_grad = False
    print("Transformer encoder frozen")

EMBED_DIM = encoder.config.hidden_size
print(f"Embedding dimension: {EMBED_DIM}")

if DATASET == "pres":
    alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
    alltxt = np.array(alltxt)
    alllabs = np.where(np.array(alllabs) == 1, 0, 1)
else:
    alltxt, alllabs = load_movies("./Dataset/movies1000/")
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)

print(f"Size train: {len(alltxt)}")

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

print(f"Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")

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

print("Preprocessing texts")
X_train_prep = [prep.process(t) for t in X_train_sub]
X_val_prep = [prep.process(t) for t in X_val]
X_test_prep = [prep.process(t) for t in X_test]

def encode_texts_to_word_embeddings(texts, tokenizer, encoder, device, max_length=MAX_LENGHT_TOKEN):
    """Encode texts to word embeddings using transformer, skipping special tokens."""
    embeddings_list = []
    valid_indices = []  # Track which original indices are kept
    
    print(f"Encoding {len(texts)} texts...")
    for idx, text in enumerate(texts):
        if (idx + 1) % 500 == 0:
            print(f"  Encoded {idx + 1}/{len(texts)}")
        
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding=False
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state[0]
        
        word_embeddings = token_embeddings[1:-1]  # Remove [CLS] and [SEP]
        if len(word_embeddings) > 0:
            embeddings_list.append(word_embeddings.detach().cpu())
            valid_indices.append(idx)
    
    return embeddings_list, valid_indices

print("Encoding train ")
X_train_embeddings, train_valid_idx = encode_texts_to_word_embeddings(X_train_prep, tokenizer, encoder, device)
y_train_sub = y_train_sub[train_valid_idx]

print("Encoding val ")
X_val_embeddings, val_valid_idx = encode_texts_to_word_embeddings(X_val_prep, tokenizer, encoder, device)
y_val = y_val[val_valid_idx]

print("Encoding test")
X_test_embeddings, test_valid_idx = encode_texts_to_word_embeddings(X_test_prep, tokenizer, encoder, device)
y_test = y_test[test_valid_idx]

def collate_fn(batch):
    """Pad embedding sequences in a batch."""
    embeddings, labels = zip(*batch)
    lengths = torch.tensor([len(emb) for emb in embeddings], dtype=torch.long)
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


dataset_train = EmbeddingDataset(X_train_embeddings, y_train_sub)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

dataset_val = EmbeddingDataset(X_val_embeddings, y_val)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

dataset_test = EmbeddingDataset(X_test_embeddings, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


class RNNClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.3, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        
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
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'lstm':
            _, (hn, _) = self.rnn(packed)
        else:
            _, hn = self.rnn(packed)
        
        hn = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(self.dropout(hn))

model = RNNClassifier(
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rnn_type=RNN_TYPE,).to(device)

print(model)

# Load previous model if continuing training
previous_results = None
start_epoch = 1
if CONTINUE_TRAINING and os.path.exists(f"{SAVING_FILE_NAME}.pth"):
    print(f"\nLoading model from {SAVING_FILE_NAME}.pth...")
    model.load_state_dict(torch.load(f"{SAVING_FILE_NAME}.pth", map_location=device))
    print("Model loaded successfully!")
    
    # Load previous training history if available
    if os.path.exists(f"{SAVING_FILE_NAME}.json"):
        with open(f"{SAVING_FILE_NAME}.json", "r") as f:
            previous_results = json.load(f)
        print(f"Loaded previous training results")
        start_epoch = previous_results.get("epochs_trained", 1) + 1
        print(f"Resuming from epoch {start_epoch}")
else:
    if CONTINUE_TRAINING:
        print(f"\nModel file {SAVING_FILE_NAME}.pth not found. Starting fresh training.")
    else:
        print(f"\nStarting fresh training (CONTINUE_TRAINING=False)")

def compute_class_weights(labels):
    counts = torch.bincount(torch.tensor(labels)).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights

class_weights = compute_class_weights(y_train_sub).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute evaluation metrics."""
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    if DATASET == "pres":
        return {
            "f1": float(f1_score(all_labels, all_preds, average="binary", zero_division=0)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
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


def run_epoch(model, loader, optimizer, criterion, training=True):
    """Run one epoch of training or evaluation."""
    model.train() if training else model.eval()

    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for embeddings, lengths, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(embeddings, lengths)
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

# Initialize history from previous training or start fresh
history = previous_results.get("history", []) if previous_results else []
best_f1 = previous_results.get("best_val_f1", 0.0) if previous_results else 0.0
best_model_state = None
patience_counter = 0

print(f"\nStarting training for {ADDITIONAL_EPOCHS} epochs (resuming from epoch {start_epoch})...")

for epoch in range(start_epoch, start_epoch + ADDITIONAL_EPOCHS):
    print(f"Epoch {epoch}/{start_epoch + ADDITIONAL_EPOCHS - 1}")

    train_m = run_epoch(model, dataloader_train, optimizer, criterion, training=True)
    val_m = run_epoch(model, dataloader_val, optimizer, criterion, training=False)

    if val_m["f1"] > best_f1:
        best_f1 = val_m["f1"]
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        improvement_marker = "improved"
    else:
        patience_counter += 1
        improvement_marker = f"patience {patience_counter}/{EARLY_STOPPING_PATIENCE}"

    history.append({
        "epoch": epoch,
        "train": train_m,
        "val": val_m,})

    print(
        f"{improvement_marker:25s} | "
        f"loss train {train_m['loss']:.4f}  val {val_m['loss']:.4f} | "
        f"f1 train {train_m['f1']:.3f}  val {val_m['f1']:.3f} | "
        f"auc train {train_m['roc_auc']:.3f}  val {val_m['roc_auc']:.3f} | "
        f"prec train {train_m['precision']:.3f}  val {val_m['precision']:.3f} | "
        f"rec train {train_m['recall']:.3f}  val {val_m['recall']:.3f}"
    )

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nBest val F1: {best_f1:.3f}")

if best_model_state is not None:
    model.load_state_dict(best_model_state)

torch.save(model.state_dict(), f"{SAVING_FILE_NAME}.pth")

test_m = run_epoch(model, dataloader_test, optimizer, criterion, training=False)

print(
    f"\nTest | loss {test_m['loss']:.4f} | f1 {test_m['f1']:.4f} | "
    f"prec {test_m['precision']:.4f} | rec {test_m['recall']:.4f} | "
    f"auc {test_m['roc_auc']:.4f}"
)

results = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "continued_training": CONTINUE_TRAINING,
    "additional_epochs": ADDITIONAL_EPOCHS,
    "device": str(device),
    "transformer_encoder": {
        "model": TRANSFORMER_MODEL,
        "embed_dim": EMBED_DIM,
        "frozen": FREEZE_TRANSFORMER,
    },
    "model_config": {
        "rnn_type": RNN_TYPE,
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

with open(f"{SAVING_FILE_NAME}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {SAVING_FILE_NAME}.json")
print(f"Model saved to {SAVING_FILE_NAME}.pth")

print("DATASET ", DATASET)
print(f"\nLoaded transformer encoder ({TRANSFORMER_MODEL})...")
print(f"Transformer embedding dimension: {EMBED_DIM}")
print(f"\nLoaded model ({RNN_TYPE})...")
