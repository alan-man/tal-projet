# not used for movies

from preprocessing_class import Preprocessing, load_pres
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import json
from datetime import datetime
from scipy.sparse import csr_matrix
import spacy

from preprocessing_class import Preprocessing, load_pres, load_movies

DATASET = "pres"
INPUT_TYPE = 'pos_tags'  # 'pos_tags', 'tfidf', 'count'
RNN_TYPE = 'gru'  # 'lstm' or 'gru'
DROPOUT = 0.4      
EARLY_STOPPING_PATIENCE = 10  # stop if val F1 doesn't improve for N epochs
SAVING_FILE_NAME =  DATASET + "_pos_tag"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if DATASET == "pres":

    alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)
    alllabs = np.where(alllabs == 1, 0, 1)
    print(f"Size train: {len(alltxt)}")

else:
    alltxt,alllabs = load_movies("./Dataset/movies1000/")
    alltxt, alllabs = np.array(alltxt), np.array(alllabs)
    print(f"Size train: {len(alltxt)}")


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

if INPUT_TYPE == 'pos_tags':
    if DATASET == "pres":
        nlp = spacy.load("fr_core_news_sm")        
    else:
        nlp = spacy.load("en_core_web_sm")  

    def pos_with_tense(text):
        """
        Returns a list of POS tags for each token.
        For verbs, append tense from morph features (e.g., VERB_Pres)
        """
        doc = nlp(text)
        pos_tags = []
        for token in doc:
            pos = token.pos_  # POS tag
            # If token is a verb, add tense if available
            if pos == "VERB":
                tense = token.morph.get("Tense")
                tense_str = tense[0] if tense else "X" 
                pos = f"{pos}_{tense_str}"  # e.g., VERB_Pres
            pos_tags.append(pos)
        return pos_tags
    
    print("POS tagging train")
    X_train_features = [pos_with_tense(str(t)) for t in X_train_sub]
    print("POS tagging val")
    X_val_features   = [pos_with_tense(str(t)) for t in X_val]
    print("POS tagging test")
    X_test_features  = [pos_with_tense(str(t)) for t in X_test]
    
elif INPUT_TYPE in ['tfidf', 'count']:
    # Use TF-IDF or Count vectorizer
    stpw = "french" if DATASET == "pres" else "english"
    print("vect", INPUT_TYPE)    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        max_df=0.95,
        min_df=2,
        stop_words=stpw
    ) if INPUT_TYPE == 'tfidf' else CountVectorizer(
        max_features=5000,
        max_df=0.95,
        min_df=2,
        stop_words=stpw
    )
    X_train_vec = vectorizer.fit_transform(X_train_sub)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)
    
    num_features = X_train_vec.shape[1]
    
    def sparse_to_sequences(X_sparse, top_k=50, shift=2):
        """
        Convert sparse TF-IDF/Count matrix to sequences.
        Each document becomes a sequence of top-k feature indices (shifted by 2).
        """
        sequences = []
        for i in range(X_sparse.shape[0]):
            row = X_sparse[i].toarray().flatten()
            # Get indices of top-k non-zero values
            top_indices = np.argsort(-row)[:top_k]
            top_indices = top_indices[row[top_indices] > 0]  # Only keep non-zero
            shifted_indices = (top_indices + shift).tolist() if len(top_indices) > 0 else [0]
            sequences.append(shifted_indices)
        return sequences
    
    X_train_features = sparse_to_sequences(X_train_vec)
    X_val_features   = sparse_to_sequences(X_val_vec)
    X_test_features  = sparse_to_sequences(X_test_vec)

if INPUT_TYPE == 'pos_tags':
    def build_vocab(all_sequences):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for seq in all_sequences:
            for item in seq:
                if item not in vocab:
                    vocab[item] = len(vocab)
        return vocab
    
    vocab = build_vocab(X_train_features)
else:
    vocab = {i: i for i in range(num_features + 2)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

print(f"Vocab size: {len(vocab)}")

def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded  = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, vocab):
        self.labels = labels
        self.vocab  = vocab
        self.encoded = [
            torch.tensor([vocab.get(t, vocab["<UNK>"]) for t in seq], dtype=torch.long)
            for seq in sequences
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


dataset_train = SequenceDataset(X_train_features, y_train_sub, vocab)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True,  collate_fn=collate_fn)

dataset_val = SequenceDataset(X_val_features, y_val, vocab)
dataloader_val = DataLoader(dataset_val,   batch_size=32, shuffle=False, collate_fn=collate_fn)

dataset_test = SequenceDataset(X_test_features, y_test, vocab)
dataloader_test = DataLoader(dataset_test,  batch_size=32, shuffle=False, collate_fn=collate_fn)


class POSClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Choose RNN type: LSTM or GRU
        rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        
        self.rnn = rnn_class(
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
        
        if self.rnn_type == 'lstm':
            _, (hn, _) = self.rnn(packed)
        else:  # GRU
            _, hn = self.rnn(packed)
        
        hn     = torch.cat([hn[-2], hn[-1]], dim=1)  # concat forward + backward
        return self.classifier(self.dropout(hn))

model = POSClassifier(
    vocab_size=len(vocab),
    embed_dim=32,
    hidden_dim=64,
    num_classes=2,
    num_layers=2,
    dropout=DROPOUT,
    rnn_type=RNN_TYPE,
).to(device)                          

print(model)

def compute_class_weights(labels):
    counts  = torch.bincount(torch.tensor(labels)).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights

class_weights = compute_class_weights(y_train_sub).to(device)  # fix: use train_sub, move to device
criterion     = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def compute_metrics(all_labels, all_preds, all_probs):
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    if DATASET == "pres":
        return {
            "f1": float(f1_score(all_labels, all_preds, average="binary")),
            "avg_precision": float(average_precision_score(all_labels, all_probs)),
            "roc_auc": float(roc_auc_score(all_labels, all_probs)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds)),
        }
    return {
            "f1": float(f1_score(all_labels, all_preds, average="binary")),
            "avg_precision": float(average_precision_score(all_labels, all_probs)),
            "roc_auc": float(roc_auc_score(all_labels, all_probs)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds)),
            "accuracy": float(accuracy_score(all_labels, all_preds)),
        }

def run_epoch(model, loader, optimizer, criterion, training=True):
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

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = float(total_loss / len(loader))
    return metrics

history = []         
best_f1 = 0.0
best_model_state = None
patience_counter = 0  

for epoch in range(50):
    train_m = run_epoch(model, dataloader_train, optimizer, criterion, training=True)
    val_m   = run_epoch(model, dataloader_val,   optimizer, criterion, training=False)

    # Early stopping: check if validation F1 improved
    if val_m["f1"] > best_f1:
        best_f1          = val_m["f1"]
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0  # reset patience counter
        print("improved")
        
    else:
        patience_counter += 1
        print(f"(patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")

    # store epoch results
    history.append({
        "epoch": epoch + 1,
        "train": train_m,
        "val":   val_m,
    })

    print(
        f"loss train {train_m['loss']:.4f}  val {val_m['loss']:.4f} | "
        f"f1 train {train_m['f1']:.3f}  val {val_m['f1']:.3f} | "
        f"auc train {train_m['roc_auc']:.3f}  val {val_m['roc_auc']:.3f} | "
        f"ap train {train_m['avg_precision']:.3f}  val {val_m['avg_precision']:.3f} | "
        f"prec train {train_m['precision']:.3f}  val {val_m['precision']:.3f} | "
        f"rec train {train_m['recall']:.3f}  val {val_m['recall']:.3f}"
    )

    # Early stopping: break if patience exceeded
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest val F1: {best_f1:.3f} (at epoch {len(history) - patience_counter})")

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), SAVING_FILE_NAME + "_model.pth")

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
if DATASET != "pres":
    print(f"accuracy {test_m['accuracy']:.3f} ")

results = {
    "run_date":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device":    str(device),
    "input_type": INPUT_TYPE,
    "model_config": {
        "rnn_type": RNN_TYPE,
        "embed_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": DROPOUT,
        "weight_decay": 1e-5,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    },
    "best_val_f1": best_f1,
    "epochs_trained": len(history),
    "history":   history,            
    "test":      test_m,           
}

with open(SAVING_FILE_NAME + ".json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to " + SAVING_FILE_NAME + ".json")
