# code to run in ssh
import codecs, re, torch, gc, json
from torch import device
from transformers import DistilBertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_score, recall_score


# loading functions
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

def load_pres_test(fname):
    """
    Charge les données test, sans labels
    """
    # >0.5 = Mitterand, <0.5 = Chirac
    
    alltxts = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        alltxts.append(txt)

    return alltxts

# load dataset train
alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)

# load dataset test
alltxt_test = load_pres_test("Test_set/corpus.tache1.test.utf8")
alltxt_test = np.array(alltxt_test)

# loading the tokenized items

# loading tokenized train and validation
data = torch.load("train_tokens_dist.pt")
input_ids = data["input_ids"]
attention_masks = data["attention_masks"]

data = torch.load("test_tokens_dist.pt")
input_ids_test = data["input_ids"]
attention_masks_test = data["attention_masks"]

# train version

input_ids_train = torch.cat([input_ids, input_ids_test], dim=0)
attention_masks_train = torch.cat([attention_masks, attention_masks_test], dim=0)

# test version
data = torch.load("pred_test_tokens_dist.pt")
input_ids_test = data["input_ids"]
attention_masks_test = data["attention_masks"]

# ------- weights

alllabs = np.where(alllabs == -1, 0, alllabs) # 0-1 labels
val, counts = np.unique(alllabs, return_counts = True)

weights = 1/counts # weights for loss
y_train_tens = torch.tensor(alllabs)

# Convert labels from -1/1 → 0/1 (BCE/CE convention)
y_train_tens = ((y_train_tens + 1) // 2).long() 

# dataset loading

sample_weights = [weights[label] for label in alllabs]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(alllabs), replacement=True)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, y_train_tens)
dataset_test = TensorDataset(input_ids_test, attention_masks_test)

train_dataloader = DataLoader(dataset_train, batch_size=16, sampler=sampler)

test_dataloader = DataLoader(
    dataset_test,
    batch_size=16,
    shuffle=False # f
)

# loading not trained model
from transformers import DistilBertConfig, DistilBertForSequenceClassification

config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
    dropout=0.2,
    attention_dropout=0.2,
    seq_classif_dropout=0.4
)

model = DistilBertForSequenceClassification.from_pretrained("bert_dist_cache/models--distilbert-base-cased/snapshots/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0", config = config)


# Freeze all BERT layers, train only the classifier head
for param in model.bert.parameters():
    param.requires_grad = False

# Or unfreeze only the last N encoder layers
for layer in model.bert.encoder.layer[-4:]: # last 4 layers
    for param in layer.parameters():
        param.requires_grad = True

# pre emtpying cache
gc.collect()
torch.cuda.empty_cache()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev)
print("dev", dev)

class_weights = torch.tensor(weights, dtype=torch.float).to(dev)
loss_fn = CrossEntropyLoss(weight=class_weights)

# ------ running loop

optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 5 # to test

model.to(dev)
epochs_evolution = {}

# ── Training ─────────────────────────────────────────────────────────────────
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []
    all_train_logits = []

    for i, batch in enumerate(train_dataloader):
        print("batch", i)
        b_input_ids, b_attention_mask, b_labels = [t.to(dev) for t in batch]
        model.zero_grad()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            # labels=b_labels
        )
        # loss = outputs.loss
        logits = outputs.logits
        loss = loss_fn(logits, b_labels)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(b_labels.cpu().numpy())
        all_train_logits.append(logits.detach().cpu())

        del b_input_ids, b_attention_mask, b_labels
        del outputs, loss, logits, preds
        torch.cuda.empty_cache()
        gc.collect()

    avg_train_loss = total_loss / len(train_dataloader)

    # Training metrics
    train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
    train_precision = precision_score(all_train_labels, all_train_preds, average="macro")
    train_recall = recall_score(all_train_labels, all_train_preds, average="macro")

    all_logits_tensor = torch.cat(all_train_logits, dim=0)
    train_probs = torch.softmax(all_logits_tensor, dim=1)[:, 1].numpy()
    train_auc = roc_auc_score(all_train_labels, train_probs)
    train_ap = average_precision_score(all_train_labels, train_probs)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} | "
          f"F1: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | "
          f"AUC: {train_auc:.4f} | AP: {train_ap:.4f}")

    epochs_evolution[f"epoch_{epoch+1}"] = { # fixed key to track each epoch
        "loss": avg_train_loss,
        "f1": train_f1,
        "precision": train_precision,
        "recall": train_recall,
        "auc": train_auc,
        "ap": train_ap
    }

    del all_train_logits, all_logits_tensor, train_probs
    del all_train_preds, all_train_labels
    torch.cuda.empty_cache()
    gc.collect()

json.dump(epochs_evolution, open("train_res.json", "w"))
print("Training metrics saved to train_res.json")

# ── Prediction on unlabeled test set ─────────────────────────────────────────
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_attention_mask = [t.to(dev) for t in batch] # no labels

        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        del b_input_ids, b_attention_mask
        del outputs, logits, probs, preds
        torch.cuda.empty_cache()
        gc.collect()

# Save predictions and probabilities
np.save("PRED_test_preds.npy", np.array(all_preds))
np.save("PRED_test_probs.npy", np.array(all_probs))
print(f"Predictions saved — {len(all_preds)} samples")
print(f"Class distribution: { {v: int((np.array(all_preds)==v).sum()) for v in np.unique(all_preds)} }")

print("predictions unique", np.unique(all_preds, return_counts=True))

# ── Final cleanup ─────────────────────────────────────────────────────────────
del all_preds, all_probs
torch.cuda.empty_cache()
gc.collect()