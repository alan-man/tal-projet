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

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
MAX_LENGTH = 512 
BATCH_SIZE = 8
# MODEL_NAME = "almanach/camemberta-base"
# MODEL_PATH = "model_transformer_camberta.pth"
# OUTPUT_FILE = "predictions_camberta.csv"

# prediction movies with roberta
MODEL_NAME = "FacebookAI/roberta-base"
MODEL_PATH = "movies_transformer_512_2e-05_FacebookAI_roberta-base.pth"
OUTPUT_FILE = "predictions_" + f"movies_{str(MODEL_NAME).replace('/', '_')}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_movies_test(path2data):
    alltxts = []
    with open(path2data, 'r') as file:
        for line in file:        
            alltxts.append(line)
    return alltxts

print("Loading data...")
alltxt = load_movies_test("Test_set/testSentiment.txt")
alltxt = np.array(alltxt)
print(f"  Total samples: {len(alltxt)}")

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

print("Creating test dataloader...")
dataset_test = TensorDataset(input_ids_test, attention_masks_test)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Making predictions...")
all_probs = []
all_preds = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        b_input_ids, b_attention_mask = [t.to(device) for t in batch]
        
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits.float()
        
        # Get probabilities for class 1 (positive class)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * BATCH_SIZE}/{len(alltxt)} samples")

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
print(f"  Total predictions: {len(all_probs)}") 
print(np.unique(all_preds, return_counts=True))
final_preds = np.where(all_preds == 0, 'N', 'P')

print(f"\nSaving predictions to {OUTPUT_FILE}...")
with open(OUTPUT_FILE+".csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for pred in final_preds:
        writer.writerow([pred])

print(f"✓ Predictions saved to {OUTPUT_FILE}")
print(f"  Total predictions: {len(final_preds)}")