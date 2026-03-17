from sklearn.model_selection import train_test_split
import numpy as np
import re, codecs
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification
import gc, torch

import gc
import json

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from torch import device


def load_pres(fname):
    """
    Charge les données
    """
    # M = -1, C = 1
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else:
            alllabs.append(1)
        alltxts.append(txt)

    return alltxts,alllabs

# load dataset
alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)

# train test split here for test in final prediction

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

# loading
data = torch.load("train_tokens_dist.pt")
input_ids = data["input_ids"]
attention_masks = data["attention_masks"]

data = torch.load("test_tokens_dist.pt")
input_ids_test = data["input_ids"]
attention_masks_test = data["attention_masks"]


# labels to array
y_train_tens = torch.tensor(y_train)
y_test_tens = torch.tensor(y_test) 

# Convert labels from -1/1 → 0/1 (BCE/CE convention)
y_train_tens = ((y_train_tens + 1) // 2).long() 
y_test_tens = ((y_test_tens + 1) // 2).long()


dataset_train = TensorDataset(input_ids,  attention_masks, y_train_tens)
dataset_test = TensorDataset(input_ids_test,  attention_masks_test, y_test_tens)

train_dataloader = DataLoader(
    dataset_train,
    batch_size=16,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset_test,
    batch_size=16,
    shuffle=False # f
)


# loading model
model = DistilBertForSequenceClassification.from_pretrained("bert_dist_cache/models--distilbert-base-cased/snapshots/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev)
print('Device:', dev) 

optimizer = optim.Adam(model.parameters(), lr=2e-4)
epochs = 1

epochs_evolution = {}

for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []
    all_train_logits = []  # collect logits for AUC later

    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_labels = [t.to(dev) for t in batch]
        model.zero_grad()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels, 
            show_progress_bar=True,
        )

        loss = outputs.loss
        logits = outputs.logits  # ← was missing in original

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Move to CPU immediately to free GPU memory
        preds = torch.argmax(logits, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(b_labels.cpu().numpy())
        all_train_logits.append(logits.detach().cpu())  # detach before storing


        # ── Free batch tensors ──────────────────────────────────────────
        del b_input_ids, b_attention_mask, b_labels
        del outputs, loss, logits, preds
        torch.cuda.empty_cache()
        gc.collect()
        # ────────────────────────────────────────────────────────────────
    
    avg_train_loss = total_loss / len(train_dataloader)

    # Training metrics
    train_f1        = f1_score(all_train_labels, all_train_preds, average="macro")
    train_precision = precision_score(all_train_labels, all_train_preds, average="macro")
    train_recall    = recall_score(all_train_labels, all_train_preds, average="macro")

    # Stack all logits at once (cleaner than the original list-of-logits approach)
    all_logits_tensor = torch.cat(all_train_logits, dim=0)
    train_probs = torch.softmax(all_logits_tensor, dim=1)[:, 1].numpy()

    train_auc = roc_auc_score(all_train_labels, train_probs)
    train_ap  = average_precision_score(all_train_labels, train_probs)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} | "
          f"F1: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | "
          f"AUC: {train_auc:.4f} | AP: {train_ap:.4f}")
    
    epochs_evolution["epoch"] = [avg_train_loss, train_f1, train_precision, train_recall, train_auc, train_ap]

    # ── Free epoch-level accumulators ───────────────────────────────────
    del all_train_logits, all_logits_tensor, train_probs
    del all_train_preds, all_train_labels
    torch.cuda.empty_cache()
    gc.collect()
    # ────────────────────────────────────────────────────────────────────


# Serialize data into file:
json.dump( epochs_evolution, open( "train_res.json", 'w' ) )

# ── Evaluation ──────────────────────────────────────────────────────────────
model.eval()
all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_attention_mask, b_labels = [t.to(dev) for t in batch]

        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1)[:, 1]
        preds   = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # ── Free batch tensors ──────────────────────────────────────────
        del b_input_ids, b_attention_mask, b_labels
        del outputs, logits, probs, preds
        torch.cuda.empty_cache()
        gc.collect()
        # ────────────────────────────────────────────────────────────────

f1                = f1_score(all_labels, all_preds, average="macro")
precision         = precision_score(all_labels, all_preds, average="macro")
recall            = recall_score(all_labels, all_preds, average="macro")
auc               = roc_auc_score(all_labels, all_probs)
average_precision = average_precision_score(all_labels, all_preds)
 
print("\nTest set metrics:")
print(f"F1 Score:          {f1:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"AUC:               {auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

np.save("test_preds.npy", np.array([f1, precision, recall, auc, average_precision]))

# ── Final cleanup ────────────────────────────────────────────────────────────
del all_preds, all_labels, all_probs
torch.cuda.empty_cache()
gc.collect()
