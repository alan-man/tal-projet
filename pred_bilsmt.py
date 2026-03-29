"""
Inference Script: Apply Trained BiLSTM Model to Raw Probabilities
------------------------------------------------------------------
Loads a trained BiLSTM model and applies it to probability files
to generate corrected probabilities.
"""

import os
import numpy as np
import torch
import torch.nn as nn

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3

MODEL_PATH = "bilstm_prob_corrector_all_train_smoothed.pth"
PROBS_FILE = "probs_sub3_smoothed.npy"
OUTPUT_DIR = "corrected_outputs"
OUTPUT_PRED = "predictions_sub4_smoothed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LSTMpostproc(nn.Module):
    """
    Bidirectional LSTM: raw prob sequence → corrected logits.
    Input:  [B, 1, T]
    Output: [B, T]
    """
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
        self.head    = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        x = x.permute(0, 2, 1)          # [B, 1, T] → [B, T, 1]
        out, _ = self.lstm(x)            # [B, T, hidden*2]
        out = self.dropout(out)
        logits = self.head(out)          # [B, T, 1]
        return logits.squeeze(-1)        # [B, T]


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

    print("\nRunning inference...")
    with torch.no_grad():
        probs_tensor = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
        logits = model(probs_tensor)                                    # [1, T]
        corrected = torch.sigmoid(logits).squeeze().cpu().numpy()       # [T]
        preds = (corrected > 0.5).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PRED}.npy")
    np.save(out_path, corrected)

    print(f"\nResults:")
    print(f"  shape:     {corrected.shape}")
    print(f"  mean prob: {corrected.mean():.3f}")
    print(f"  pred 1s:   {preds.sum()}/{len(preds)}")
    print(f"  saved →    {out_path}")
    print("\nDone.")