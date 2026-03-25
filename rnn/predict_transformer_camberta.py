"""
Prediction script for transformer-based sentiment classification
Loads trained model and makes predictions on test data
"""

import codecs
import re
import numpy as np
import torch
import csv
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────
# CONFIGURATION (must match training script)
# ─────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
MAX_LENGTH = 128
BATCH_SIZE = 8
MODEL_NAME = "almanach/camemberta-base"
MODEL_PATH = "model_transformer_camberta.pth"
OUTPUT_FILE = "predictions_camberta.csv"

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# DATA LOADING (same as training)
# ─────────────────────────────────────────

def load_pres_test(fname):
    """
    Load test data without labels
    """
    alltxts = []
    with codecs.open(fname, 'r', 'utf-8') as s:
        while True:
            txt = s.readline()
            if len(txt) < 5:
                break
            
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", r"\1", txt)
            alltxts.append(txt)

    return alltxts

print("Loading data...")
alltxt = load_pres_test("Test_set/corpus.tache1.test.utf8")
alltxt = np.array(alltxt)
print(f"  Total samples: {len(alltxt)}")

# ─────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────

print(f"Loading tokenizer ({MODEL_NAME})...")
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


print("Tokenizing test set...")
input_ids_test, attention_masks_test = tokenize_dataset(alltxt, tokenizer)

# ─────────────────────────────────────────
# DATASET AND DATALOADER
# ─────────────────────────────────────────

print("Creating test dataloader...")
dataset_test = TensorDataset(input_ids_test, attention_masks_test)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ─────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────

print("Making predictions...")
all_probs = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        b_input_ids, b_attention_mask = [t.to(device) for t in batch]
        
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits.float()
        
        # Get probabilities for class 1 (positive class)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * BATCH_SIZE}/{len(alltxt)} samples")

all_probs = np.array(all_probs)

# ─────────────────────────────────────────
# SAVE PREDICTIONS TO CSV
# ─────────────────────────────────────────

print(f"\nSaving predictions to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prob'])
    for prob in all_probs:
        writer.writerow([prob])

print(f"✓ Predictions saved to {OUTPUT_FILE}")
print(f"  Total predictions: {len(all_probs)}")
print(f"  Min probability: {all_probs.min():.6f}")
print(f"  Max probability: {all_probs.max():.6f}")
print(f"  Mean probability: {all_probs.mean():.6f}")
print("Distribution ", np.unique(np.where(all_probs > 0.5, 1, 0), return_counts=True))
