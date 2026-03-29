"""
Makes predictions on the President dataset with trained model
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
MODEL_NAME = "camembert-base" #"almanach/camemberta-base"
MODEL_PATH = "{SAVING_FILE_NAME}.pth" #"model_transformer_camberta.pth"
OUTPUT_FILE = "predictions_cambert_all_train.csv"

# prediction movies with roberta
# MODEL_NAME = "FacebookAI/roberta-base"
# MODEL_PATH = "movies_transformer_512_2e-05_FacebookAI_roberta-base.pth"
# OUTPUT_FILE = "predictions_" + f"movies_{str(MODEL_NAME).replace('/', '_')}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_pres_test(fname):
    alltxts = []
    with codecs.open(fname, 'r', 'utf-8') as s:
        while True:
            txt = s.readline()
            if len(txt) < 5:
                break
            
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", r"\1", txt)
            alltxts.append(txt)

    return alltxts

print("Loading data")
alltxt = load_pres_test("Dataset/corpus.tache1.learn.utf8") # "Test_set/corpus.tache1.test.utf8"
alltxt = np.array(alltxt)
print(f"Total samples: {len(alltxt)}")

print(f"Loading tokenizer ({MODEL_NAME})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_dataset(texts, tokenizer, max_length=MAX_LENGTH):
    input_ids_list = []
    attention_masks_list = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 2500 == 0:
            print(f"Tokenized {i + 1}/{len(texts)}")
        
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt",
            return_attention_mask=True)
        
        input_ids_list.append(encoded['input_ids'])
        attention_masks_list.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    
    return input_ids, attention_masks

print("Tokenizing test set ")
input_ids_test, attention_masks_test = tokenize_dataset(alltxt, tokenizer)
 
print("Creating test dataloader ")
dataset_test = TensorDataset(input_ids_test, attention_masks_test)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
 

print(f"Loading model from {MODEL_PATH}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


print("Predictions")
all_probs = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        b_input_ids, b_attention_mask = [t.to(device) for t in batch]
        
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits.float()
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        
        if (i + 1) % 10 == 0:
            print(f" Processed {(i + 1) * BATCH_SIZE}/{len(alltxt)} samples")

all_probs = np.array(all_probs) 

print(f"\nSaving predictions to {OUTPUT_FILE} ")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    for prob in all_probs:
        writer.writerow([prob])

print(f"Predictions saved to {OUTPUT_FILE}")
print(f" Total predictions: {len(all_probs)}")
print("Distribution ", np.unique(np.where(all_probs > 0.5, 1, 0), return_counts=True))
