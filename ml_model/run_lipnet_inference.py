# run_lipnet_inference.py

import torch
import cv2
import dlib
import numpy as np
from imutils import face_utils
from lipnet import LipNet, extract_mouth_frames, indices_to_text
import sys
# ------------------ Config ------------------
MODEL_PATH = "model_epoch100.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.mpg"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Load Model ------------------
vocab = ['-'] + list("abcdefghijklmnopqrstuvwxyz ")
model = LipNet(num_classes=len(vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------ Load Dlib Models ------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ------------------ Run Inference ------------------
print(f"🔍 Extracting mouth frames from: {VIDEO_PATH}")
frames = extract_mouth_frames(VIDEO_PATH, detector, predictor)

if frames is None or len(frames) == 0:
    print("❌ No mouth frames found.")
    exit()

frames = frames.astype(np.float32) / 255.0
frames = np.expand_dims(frames, axis=1)  # (T, 1, H, W)
frames_tensor = torch.tensor(frames).unsqueeze(0).to(DEVICE)  # (1, T, 1, H, W)

with torch.no_grad():
    output = model(frames_tensor).log_softmax(2)  # (time, 1, num_classes)

# ------------------ Decode ------------------
def greedy_decoder(output):
    output = output.cpu().numpy()[:, 0, :]  # (time, num_classes)
    decoded = []
    prev = -1
    for t in range(output.shape[0]):
        best = output[t].argmax()
        if best != prev and best != 0:
            decoded.append(best)
        prev = best
    return indices_to_text(decoded)

prediction = greedy_decoder(output)
print("✅ Predicted Transcript:", prediction)
