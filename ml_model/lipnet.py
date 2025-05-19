import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from imutils import face_utils

# ---------- Model Definition ----------
vocab = ['-'] + list("abcdefghijklmnopqrstuvwxyz ")
num_classes = len(vocab)

def text_to_indices(text):
    return [vocab.index(ch) for ch in text if ch in vocab]

def indices_to_text(indices):
    return ''.join([vocab[i] for i in indices if i != 0])

class LipNet(nn.Module):
    def __init__(self, num_classes):
        super(LipNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.cnn(x)
        x = x.mean(dim=[3,4])
        x = x.permute(0,2,1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.transpose(0,1)

# ---------- Inference Helper ----------
def extract_mouth_from_frame(frame, detector, predictor, margin=10):
    if frame is None: return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector(rgb_frame, 1)
    if len(rects) == 0: return None
    shape = predictor(rgb_frame, rects[0])
    shape = face_utils.shape_to_np(shape)
    mouth = shape[48:68]
    (x, y, w, h) = cv2.boundingRect(np.array(mouth))
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    return frame[y:y+h+margin, x:x+w+margin]

def extract_mouth_frames(video_path, detector, predictor):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        mouth = extract_mouth_from_frame(frame, detector, predictor)
        if mouth is not None:
            mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth = cv2.resize(mouth, (100, 50))
            frames.append(mouth)
    cap.release()
    return np.array(frames)
