"""
Loads a trained/best RNN encoder checkpoint and performs classification on test set.
"""

import os
import string, codecs, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

from python.preprocessing_class import Preprocessing, load_pres, load_movies

# CONFIGURATION !!!!

# choose dataset
DATASET = "movies" # "pres"
MAX_LENGHT_TOKEN = 256

# same split params as training
RANDOM_STATE = 42
TEST_SIZE = 0.20

# same model params as training file
BATCH_SIZE = 16
RNN_TYPE = "gru" # "lstm" or "gru"
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.4

SAVING_FILE_NAME = f"{DATASET}_rnn_encoder_{RNN_TYPE}"

# transformer encoder config
TRANSFORMER_MODEL = "intfloat/multilingual-e5-base"
FREEZE_TRANSFORMER = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"\nLoading transformer encoder ({TRANSFORMER_MODEL}) ")
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
encoder = AutoModel.from_pretrained(TRANSFORMER_MODEL)

if FREEZE_TRANSFORMER:
    for param in encoder.parameters():
        param.requires_grad = False
    print("Transformer encoder frozen")

EMBED_DIM = encoder.config.hidden_size
print(f"Embedding dimension: {EMBED_DIM}")

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

def load_movies_test(path2data):
    alltxts = []
    c = 0
    with open(path2data, 'r') as file:
        for line in file: 
            alltxts.append(line)
    return alltxts


if DATASET == "pres":
    alltxt = load_pres_test("Test_set/corpus.tache1.test.utf8")
    alltxt = np.array(alltxt)
else:
    alltxt = load_movies_test("Test_set/testSentiment.txt")
    alltxt = np.array(alltxt)

print(f"Test size: {len(alltxt)}")

punc = set(string.punctuation + "\n\r\t")
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
    urls=False,
)

print("Preprocessing test texts")
X_test_prep = [prep.process(t) for t in alltxt]


def encode_texts_to_word_embeddings(texts, tokenizer, encoder, max_length=MAX_LENGHT_TOKEN):
    embeddings_list = []
    valid_indices = []

    print(f"Encoding {len(texts)} test texts ")
    for idx, text in enumerate(texts):
        if (idx + 1) % 500 == 0:
            print(f" Encoded {idx + 1}/{len(texts)}")

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )

        with torch.no_grad():
            outputs = encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            token_embeddings = outputs.last_hidden_state[0]

        word_embeddings = token_embeddings[1:-1]
        if len(word_embeddings) > 0:
            embeddings_list.append(word_embeddings.detach().cpu())
            valid_indices.append(idx)

    return embeddings_list, valid_indices


X_test_embeddings, valid_idx = encode_texts_to_word_embeddings(X_test_prep, tokenizer, encoder)


def collate_fn(batch):
    lengths = torch.tensor([len(emb) for emb in batch], dtype=torch.long)
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded, lengths


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class RNNClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.3, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

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

        if self.rnn_type == "lstm":
            _, (hn, _) = self.rnn(packed)
        else:
            _, hn = self.rnn(packed)

        hn = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(self.dropout(hn))


dataset_test = EmbeddingDataset(X_test_embeddings)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = RNNClassifier(
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rnn_type=RNN_TYPE,
).to(device)

model_path = f"{SAVING_FILE_NAME}.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}. Train or continue training first."
    )

print(f"\nLoading best RNN encoder from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_preds, all_probs = [], []

with torch.no_grad():
    for embeddings, lengths in dataloader_test:
        embeddings = embeddings.to(device)

        logits = model(embeddings, lengths)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

preds_path = f"{SAVING_FILE_NAME}_test_preds.npy"
probs_path = f"{SAVING_FILE_NAME}_test_probs.npy"
np.save(preds_path, all_preds)
np.save(probs_path, all_probs)
print("distribution", np.unique(all_preds, return_counts=True))

print(f"\nSaved predictions to {preds_path}")
print(f"Saved probabilities to {probs_path}")

