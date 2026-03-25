# code to run cross-validation in ssh
import codecs, re, torch, gc, json
from torch import device
from transformers import DistilBertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
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

# load dataset train
alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)

# loading the tokenized items
data = torch.load("train_tokens_dist.pt")
input_ids = data["input_ids"]
attention_masks = data["attention_masks"]

data = torch.load("test_tokens_dist.pt")
input_ids_test = data["input_ids"]
attention_masks_test = data["attention_masks"]

# train version

input_ids_train = torch.cat([input_ids, input_ids_test], dim=0)
attention_masks_train = torch.cat([attention_masks, attention_masks_test], dim=0)

# Convert labels from -1/1 → 0/1 (BCE/CE convention)
alllabs = np.where(alllabs == -1, 0, alllabs)  # 0-1 labels
y_train_tens = torch.tensor(alllabs, dtype=torch.long)

# pre emptying cache
gc.collect()
torch.cuda.empty_cache()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

# ── Cross-Validation ──────────────────────────────────────────────────────────
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_results = {}

fold_idx = 0
for train_idx, val_idx in skf.split(input_ids_train, alllabs):
    fold_idx += 1
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}/{n_splits}")
    print(f"{'='*80}")
    
    # Split data for this fold
    X_train_fold = input_ids_train[train_idx]
    mask_train_fold = attention_masks_train[train_idx]
    y_train_fold = y_train_tens[train_idx]
    
    X_val_fold = input_ids_train[val_idx]
    mask_val_fold = attention_masks_train[val_idx]
    y_val_fold = y_train_tens[val_idx]
    
    # Compute weights for weighted sampler
    _, counts = np.unique(alllabs[train_idx], return_counts=True)
    weights = 1/counts
    sample_weights = [weights[label] for label in alllabs[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(alllabs[train_idx]), replacement=True)
    
    # Create dataloaders
    dataset_train = TensorDataset(X_train_fold, mask_train_fold, y_train_fold)
    dataset_val = TensorDataset(X_val_fold, mask_val_fold, y_val_fold)
    
    train_dataloader = DataLoader(dataset_train, batch_size=16, sampler=sampler)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=False)
    
    # Load fresh model for this fold
    from transformers import DistilBertConfig
    
    config = DistilBertConfig.from_pretrained('distilbert-base-cased',
        dropout=0.2,
        attention_dropout=0.2,
        seq_classif_dropout=0.4
    )
    
    model = DistilBertForSequenceClassification.from_pretrained(
        "bert_dist_cache/models--distilbert-base-cased/snapshots/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0", 
        config=config
    )
    
    # Freeze all BERT layers, train only the classifier head
    for param in model.distilbert.parameters():
        param.requires_grad = False
    
    # Unfreeze only the last N encoder layers
    for layer in model.distilbert.transformer.layer[-4:]:  # last 4 layers
        for param in layer.parameters():
            param.requires_grad = True
    
    model.to(dev)
    
    class_weights = torch.tensor(weights, dtype=torch.float).to(dev)
    loss_fn = CrossEntropyLoss(weight=class_weights)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 5
    
    fold_epochs_evolution = {}
    
    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        all_train_logits = []
        
        for i, batch in enumerate(train_dataloader):
            b_input_ids, b_attention_mask, b_labels = [t.to(dev) for t in batch]
            model.zero_grad()
            
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
            )
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
        train_f1        = f1_score(all_train_labels, all_train_preds, average="macro")
        train_precision = precision_score(all_train_labels, all_train_preds, average="macro")
        train_recall    = recall_score(all_train_labels, all_train_preds, average="macro")
        
        all_logits_tensor = torch.cat(all_train_logits, dim=0)
        train_probs = torch.softmax(all_logits_tensor, dim=1)[:, 1].numpy()
        train_auc = roc_auc_score(all_train_labels, train_probs)
        train_ap  = average_precision_score(all_train_labels, train_probs)
        
        # ── Validation metrics ───────────────────────────────────────────────
        model.eval()
        all_val_preds = []
        all_val_labels = []
        all_val_logits = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_attention_mask, b_labels = [t.to(dev) for t in batch]
                
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(b_labels.cpu().numpy())
                all_val_logits.append(logits.detach().cpu())
                
                del b_input_ids, b_attention_mask, b_labels
                del outputs, logits, preds
                torch.cuda.empty_cache()
                gc.collect()
        
        # Validation metrics
        val_f1        = f1_score(all_val_labels, all_val_preds, average="macro")
        val_precision = precision_score(all_val_labels, all_val_preds, average="macro")
        val_recall    = recall_score(all_val_labels, all_val_preds, average="macro")
        
        all_val_logits_tensor = torch.cat(all_val_logits, dim=0)
        val_probs = torch.softmax(all_val_logits_tensor, dim=1)[:, 1].numpy()
        val_auc = roc_auc_score(all_val_labels, val_probs)
        val_ap  = average_precision_score(all_val_labels, val_probs)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f} | F1: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | AUC: {train_auc:.4f} | AP: {train_ap:.4f}")
        print(f"  Val   - F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | AUC: {val_auc:.4f} | AP: {val_ap:.4f}")
        
        fold_epochs_evolution[f"epoch_{epoch+1}"] = {
            "train_loss": avg_train_loss,
            "train_f1": train_f1,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_auc": train_auc,
            "train_ap": train_ap,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auc": val_auc,
            "val_ap": val_ap,
        }
        
        del all_train_logits, all_logits_tensor, train_probs
        del all_val_logits, all_val_logits_tensor, val_probs
        del all_train_preds, all_train_labels, all_val_preds, all_val_labels
        torch.cuda.empty_cache()
        gc.collect()
    
    cv_results[f"fold_{fold_idx}"] = fold_epochs_evolution
    
    # Cleanup after fold
    del model, optimizer, loss_fn
    del dataset_train, dataset_val, train_dataloader, val_dataloader
    torch.cuda.empty_cache()
    gc.collect()

# Save cross-validation results
json.dump(cv_results, open("cv_results.json", "w"), indent=2)
print(f"\n{'='*80}")
print("Cross-validation complete! Results saved to cv_results.json")
print(f"{'='*80}")
