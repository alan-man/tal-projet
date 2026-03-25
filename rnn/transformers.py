"""
Transformer-based sentiment classification using CamemBERT
Trains a transformer model for binary sentiment classification with early stopping and results tracking
"""

import os
import gc
import json
import codecs
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score, accuracy_score
)
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing_class import Preprocessing, load_pres, load_movies

# Fix CUDA memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DATASET = "pres"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
MAX_LENGTH = 128            # Reduced from 256 for memory savings
BATCH_SIZE = 8              # Reduced from 8 to save memory
GRADIENT_ACCUMULATION_STEPS = 2   # Accumulate 2 batches
EPOCHS = 10
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 5
USE_HALF_PRECISION = False   # Use fp16 to save memory (~50% reduction)


# Model selection
MODEL_NAME = "almanach/camemberta-base" # "camembert-base" # "almanach/camemberta-base" 'camembert-base' 
SAVING_FILE_NAME =  DATASET + "_transformer_"

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
alltxt, alllabs = None, None

if DATASET == "pres": 
    print("DATASET", DATASET)

    print("Loading data...")
    alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)
    # Convert labels: 1 -> 0, -1 -> 1
    alllabs = np.where(alllabs == 1, 0, 1)

    print(f"  Total samples: {len(alltxt)}")
    print(f"  Label distribution: {np.unique(alllabs, return_counts=True)}")

else:
    print("Loading data...")
    alltxt, alllabs = load_movies("./Dataset/movies1000/")
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)
    # Convert labels: 1 -> 0, -1 -> 1
    alllabs = np.where(alllabs == 1, 0, 1)

    print(f"  Total samples: {len(alltxt)}")
    print(f"  Label distribution: {np.unique(alllabs, return_counts=True)}")


val, counts = np.unique(alllabs, return_counts=True)
class_weights = 1 / counts

print(f"  Class weights: {class_weights}")

# ─────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────

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

# Convert to tensors
y_train_tens = torch.tensor(y_train_sub, dtype=torch.long)
y_val_tens = torch.tensor(y_val, dtype=torch.long)
y_test_tens = torch.tensor(y_test, dtype=torch.long)

# ─────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────

print(f"\nLoading tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_dataset(texts, tokenizer, max_length=MAX_LENGTH):
    """Tokenize a list of texts."""
    input_ids_list = []
    attention_masks_list = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 2500 == 0:
            print(f"  Tokenized {i + 1}/{len(texts)}")
        
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt",
            return_attention_mask=True
        )
        
        input_ids_list.append(encoded['input_ids'])
        attention_masks_list.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    
    return input_ids, attention_masks


print("Tokenizing train set...")
input_ids_train, attention_masks_train = tokenize_dataset(X_train_sub, tokenizer)

print("Tokenizing validation set...")
input_ids_val, attention_masks_val = tokenize_dataset(X_val, tokenizer)

print("Tokenizing test set...")
input_ids_test, attention_masks_test = tokenize_dataset(X_test, tokenizer)

# ─────────────────────────────────────────
# DATASETS AND DATALOADERS
# ─────────────────────────────────────────

print("\nCreating datasets and dataloaders...")

dataset_train = TensorDataset(input_ids_train, attention_masks_train, y_train_tens)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, y_val_tens)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, y_test_tens)

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

print(f"\nLoading model ({MODEL_NAME})...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Note: Using torch.cuda.amp.autocast() for mixed precision
# This is safer than model.half() as it avoids numerical instability in softmax/attention

model = model.to(device)
if USE_HALF_PRECISION:
    scaler = GradScaler()
    print("Using automatic mixed precision (autocast) with GradScaler")
else:
    scaler = None

# ─────────────────────────────────────────
# LOSS & OPTIMIZER
# ─────────────────────────────────────────

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # weight_decay=0.01

# ─────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute evaluation metrics."""
    if DATASET == "pres":
        return {
            "f1":            float(f1_score(all_labels, all_preds, average="macro")),
            "precision":     float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall":        float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            "roc_auc":       float(roc_auc_score(all_labels, all_probs)),
            "avg_precision": float(average_precision_score(all_labels, all_probs)),
        }
        
    return {
        "f1":            float(f1_score(all_labels, all_preds, average="binary")),
        "precision":     float(precision_score(all_labels, all_preds, average="binary", zero_division=0)),
        "recall":        float(recall_score(all_labels, all_preds, average="binary", zero_division=0)),
        "roc_auc":       float(roc_auc_score(all_labels, all_probs)),
        "avg_precision": float(average_precision_score(all_labels, all_probs)),
        "accuracy":      float(accuracy_score(all_labels, all_preds)),
    }


# ─────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────

def run_epoch(model, loader, optimizer, loss_fn, training=True, scaler=None):
    """Run one epoch of training or evaluation."""
    model.train() if training else model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for i, batch in enumerate(loader):
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

            if training:
                model.zero_grad()

            # Use autocast for mixed precision - safer than model.half()
            if USE_HALF_PRECISION and training:
                with autocast(dtype=torch.float16):
                    outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                    logits = outputs.logits.float()  # Ensure float32 for loss
            else:
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits.float()  # Always ensure float32
            
            loss = loss_fn(logits, b_labels)

            # Extract loss value immediately before any operations
            loss_value = loss.item()
            total_loss += loss_value

            if training:
                # Scale loss by accumulation steps
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update weights after accumulation steps
                if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    model.zero_grad()
                
                # Clean up loss-related tensors after backward
                del scaled_loss, loss
            else:
                del loss

            preds = torch.argmax(logits, dim=1)
            
            # Compute softmax using float32 to avoid NaN
            probs = torch.softmax(logits.float(), dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            # Aggressive memory cleanup - delete all GPU tensors
            del b_input_ids, b_attention_mask, b_labels, outputs, logits, preds, probs
            torch.cuda.empty_cache()
            gc.collect()

    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = float(total_loss / len(loader))

    del all_probs, all_preds, all_labels
    torch.cuda.empty_cache()
    gc.collect()

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
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_m = run_epoch(model, train_dataloader, optimizer, loss_fn, training=True, scaler=scaler)
    val_m = run_epoch(model, val_dataloader, optimizer, loss_fn, training=False, scaler=scaler)

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
        f"  {improvement_marker:25s} | "
        f"train loss {train_m['loss']:.4f} val loss {val_m['loss']:.4f} | "
        f"train f1 {train_m['f1']:.4f} val f1 {val_m['f1']:.4f} | "
        f"train auc {train_m['roc_auc']:.4f} val auc {val_m['roc_auc']:.4f}"
    )

    # Early stopping check
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
        break

print(f"\nBest val F1: {best_f1:.4f}")

# ─────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────

print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "{SAVING_FILE_NAME}.pth")

test_m = run_epoch(model, test_dataloader, optimizer, loss_fn, training=False, scaler=scaler)

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
    "model": MODEL_NAME,
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    },
    "best_val_f1": best_f1,
    "epochs_trained": len(history),
    "history": history,
    "test": test_m,
}

with open("{SAVING_FILE_NAME}_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {SAVING_FILE_NAME}_results.json")
print(f"✓ Model saved to {SAVING_FILE_NAME}.pth")
