import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mouth_extractor import preprocess_video
import argparse

# =======================
# Parse command-line args
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help="Video filename to process")
args = parser.parse_args()

# =====================
# Configuration & paths
# =====================
UPLOAD_DIR = "/app/uploadModelB"  # Path from volume mount in Docker

video_filename = args.video
video_path = os.path.join(UPLOAD_DIR, video_filename)
# ===== Load vocab =====
with open("/app/utility/vocab.json", "r") as f:
    char2idx = json.load(f)
idx2char = {v: k for k, v in char2idx.items()}

# ===== Model definition =====
class LipModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, (3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 128, (3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))
        )
        dummy = torch.zeros(1, 1, 70, 112, 112)
        cnn_out = self.cnn(dummy)
        self.rnn_input_size = cnn_out.shape[1]

        self.rnn = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W) → (B, C, T, H, W)
        x = self.cnn(x)
        x = x.mean([-1, -2])         # global average pool (B, C, T)
        x = x.permute(0, 2, 1)       # (B, T, C)
        x, _ = self.rnn(x)
        x = self.fc(x)               # (B, T, num_classes)
        return F.log_softmax(x.permute(1, 0, 2), dim=2)  # (T, B, C)

# ============================
# Extract mouth region frames
# ============================
frames_path = preprocess_video(video_path)
if not frames_path or not os.path.exists(frames_path):
    print("❌ Failed to extract mouth frames.")
    exit(1)

print(f"✅ Frame file saved at: {frames_path}")

frames = np.load(frames_path).astype(np.float32)  # shape: (T, 1, 112, 112)
input_tensor = torch.tensor(frames).unsqueeze(0)  # shape: (1, T, 1, 112, 112)

# =================
# Load trained model
# =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipModel(num_classes=len(char2idx)).to(device)

checkpoint_path = "/app/lip_model_epoch299.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# =================
# Inference
# =================
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)  # (T, B, C)
    output = output[:, 0, :]      # (T, C)

    predictions = torch.argmax(output, dim=-1).cpu()

# =================
# CTC decoding
# =================
def ctc_decode(preds, idx2char, blank=0):
    result = []
    previous = None
    for p in preds:
        p = p.item()
        if p != previous and p != blank:
            result.append(idx2char.get(p, '?'))
        previous = p
    return ''.join(result)

transcript = ctc_decode(predictions, idx2char)
print("📜 Transcript:", transcript)

# ============================
# Save transcript as JSON file
# ============================
transcript_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(video_filename)[0]}_transcript.json")
with open(transcript_path, "w") as f:
    json.dump({"transcript": transcript}, f)

print(f"💾 Transcript saved at: {transcript_path}")
