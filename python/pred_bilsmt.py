"""
Use the LSTM for probabilty smoothung to predict
"""

import os
import numpy as np
import torch
import torch.nn as nn

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3

MODEL_PATH = "bilstm_prob_corrector_all_train_smoothed.pth"
PROBS_FILE = "npy/probs_sub3_smoothed.npy"
OUTPUT_DIR = "corrected_outputs"
OUTPUT_PRED = "predictions_sub4_smoothed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMpostproc(nn.Module): 
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, 1) 

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x) 
        out = self.dropout(out)
        logits = self.head(out) 
        return logits.squeeze(-1) 

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}")
    model = LSTMpostproc(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")

    print(f"\nLoading probabilities from {PROBS_FILE}")
    probs = np.load(PROBS_FILE).astype(np.float32)
    print(f"Loaded {len(probs)} frames")

    print("Predictions")
    with torch.no_grad():
        probs_tensor = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) 
        logits = model(probs_tensor) 
        corrected = torch.sigmoid(logits).squeeze().cpu().numpy() 
        preds = (corrected > 0.5).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PRED}.npy")
    np.save(out_path, corrected)
    print("\nDone.")