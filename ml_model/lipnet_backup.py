#!/usr/bin/env python
# coding: utf-8
# %%

# # Lip Reading with LipNet in Google Colab
# 
# This notebook demonstrates how to build an offline, speaker-independent lip reading model from scratch using a LipNet-inspired architecture. The notebook will:
# 
# - Download and prepare a sample of the GRID dataset (or let you upload your own data)
# - Extract raw video frames and crop the mouth region from each frame using dlib
# - Build and train a LipNet model (CNN + Bi-GRU + CTC) on the processed data
# - Run inference on sample videos
# 


# %%


# %%


import dlib
import numpy
print("dlib version:", dlib.__version__)
print("numpy version:", numpy.__version__)


# ## 1. Download Required Files and (Sample) Dataset
# 
# We need to download the dlib shape predictor (68 landmarks). For the GRID dataset, you can either automatically download a small sample (if available) or upload your own dataset.
# 
# If you have your own data, use the file upload widget below.

# %%

'''
# Download dlib's shape predictor if not already present
import os
import gdown
if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
    get_ipython().system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    get_ipython().system('bzip2 -d shape_predictor_68_face_landmarks.dat.bz2')

# For the GRID dataset, if you have a sample archive, you can upload it below
from google.colab import files

#uploaded = files.upload()
'''

'''
if uploaded:
    # Assume the uploaded file is a zip archive containing a folder with .mpg and .align files
    import zipfile
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('./data')
    print('Dataset uploaded and extracted to ./data')
else:
    print('No dataset uploaded. Please upload a GRID sample archive (.zip) with .mpg videos and .align files.')
'''


# ## 2. Preprocessing: Extract Mouth Region from Video Frames
# 
# We use dlib to detect the face and then extract the mouth region (landmarks 48–67). The following code defines helper functions to extract the mouth frames from a video file.

# %%


import cv2
import dlib
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt

# Initialize dlib's face detector and load the shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_mouth_from_frame(frame, detector, predictor, margin=10):
    if frame is None:
        print("Warning: received None frame")
        return None

    # Debug: Print original frame info
    #print("Original frame dtype:", frame.dtype, "shape:", frame.shape)

    # Convert from BGR to RGB using slicing and ensure contiguity
    rgb = frame[..., ::-1].copy()  # This reverses the last axis (BGR -> RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=['C'])
    #print("Converted rgb dtype:", rgb.dtype, "shape:", rgb.shape)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
      rects = detector(rgb_frame, 1)
    except Exception as e:
        print("Error during detection:", e)
        return None

    if len(rects) == 0:
        print("No faces detected in this frame.")
        return None

    # Use first detected face for landmark prediction.
    shape = predictor(rgb, rects[0])
    shape = face_utils.shape_to_np(shape)

    # Mouth landmarks are 48-67
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
        if not ret:
            break
        mouth = extract_mouth_from_frame(frame, detector, predictor)
        if mouth is not None:
            # Convert to grayscale and resize for consistency
            mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth = cv2.resize(mouth, (100, 50))
            frames.append(mouth)
    cap.release()
    return np.array(frames)  # shape: (num_frames, height, width)

# Test on a sample video if available
sample_video = None
import glob
video_files = glob.glob('./data/**/*.mpg', recursive=True)
if len(video_files) > 0:
    sample_video = video_files[0]
    print('Using sample video:', sample_video)
    sample_frames = extract_mouth_frames(sample_video, detector, predictor)
    print('Extracted', len(sample_frames), 'frames.')
    plt.figure(figsize=(10,4))
    if len(sample_frames) > 0:
        plt.imshow(sample_frames[0], cmap='gray')
        plt.title('First Mouth Frame')
        plt.axis('off')
        plt.show()
else:
    print('No video files found in ./data. Please check your dataset upload.')


# ## 3. Data Preparation: PyTorch Dataset
# 
# The following Dataset class assumes that for every video file (`.mpg`) there is a corresponding transcript file (`.align`). The transcript file contains lines with start time, end time, and the spoken token. Here we simply concatenate the tokens to form the full sentence.
# 
# For a real project, you may wish to align the predictions frame-by-frame using the timestamps.

# %%


import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class LipReadingDataset(Dataset):
    def __init__(self, video_dir, align_dir, detector, predictor, transform=None):
        self.samples = []
        video_files = sorted(glob.glob(os.path.join(video_dir, '**', '*.mpg'), recursive=True))
        align_files = sorted(glob.glob(os.path.join(align_dir, '**', '*.align'), recursive=True))

        for video_path in video_files:
            base = os.path.splitext(os.path.basename(video_path))[0]
            align_path = next((f for f in align_files if base in os.path.basename(f)), None)

            frames = extract_mouth_frames(video_path, detector, predictor)
            if frames is None or len(frames) == 0:
                continue
            frames = frames.astype(np.float32) / 255.0
            frames = np.expand_dims(frames, axis=1)  # (T, 1, H, W)

            transcript = ""
            if align_path and os.path.exists(align_path):
                with open(align_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            transcript += parts[2] + ' '
                transcript = transcript.strip()

            self.samples.append({
                'frames': torch.tensor(frames),
                'transcript': transcript
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Example: create dataset (adjust paths as necessary)
video_dir = './data'
align_dir = './data'
dataset = LipReadingDataset(video_dir, align_dir, detector, predictor)
print('Number of samples in dataset:', len(dataset))


def collate_fn(batch):
    # This simple collate function assumes batch_size=1 for demo simplicity
    # In practice, you'll want to pad sequences to the same length
    frames = [item['frames'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    return frames, transcripts


# Create a simple DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


# ## 4. Define the LipNet Model (PyTorch)
# 
# Below is a simplified version of LipNet. The model first processes the input video frames with a 3D CNN, applies global average pooling, then passes the features through bidirectional GRU layers. Finally, a linear layer projects the output to the number of classes. We will use CTC loss to train the network.
# 
# For the vocabulary, we use lowercase letters and space. The CTC blank token is implicitly index 0.

# %%


import torch.nn as nn
import torch
# Define the vocabulary
vocab = ['-'] + list("abcdefghijklmnopqrstuvwxyz ")
num_classes = len(vocab)

def text_to_indices(text):
    text = text.lower()
    return [vocab.index(ch) for ch in text if ch in vocab]

def indices_to_text(indices):
    return ''.join([vocab[i] for i in indices if i != 0])

class LipNet(nn.Module):
    def __init__(self, num_classes):
        super(LipNet, self).__init__()

        # 3D CNN for spatiotemporal feature extraction
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )

        # Bidirectional GRU
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        # x: (batch, time, channels, height, width)
        # Rearrange for 3D CNN: (batch, channels, time, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        cnn_out = self.cnn(x)  # (batch, channels, time, h, w)
        # Global average pooling over spatial dims
        cnn_out = cnn_out.mean(dim=[3,4])  # (batch, channels, time)
        cnn_out = cnn_out.permute(0,2,1)  # (batch, time, channels)

        rnn_out, _ = self.rnn(cnn_out)  # (batch, time, hidden*2)
        out = self.fc(rnn_out)  # (batch, time, num_classes)
        # For CTC loss, transpose to (time, batch, num_classes)
        out = out.transpose(0,1)
        return out

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LipNet(num_classes).to(device)
print(model)


# ## 5. Training Setup
# 
# We use the CTC loss for training. The training loop below demonstrates how to process a batch from the dataset, convert the transcript to indices, and compute the loss. For demonstration, we run only a few epochs on the available (small) dataset.

# %%
import torch
import torch.optim as optim
from torch.nn import CTCLoss
from tqdm import tqdm
import os

# 1. Build model & move to device
model = LipNet(num_classes).to(device)

# 2. Re-create optimizer & loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ctc_loss  = CTCLoss(blank=0, zero_infinity=True)

# 3. Load the epoch‑50 weights
checkpoint_path = "model_weights/model_epoch50.pt"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
print(f"✅ Loaded weights from {checkpoint_path}")

# 4. Continue training for 50 more epochs
start_epoch = 51
end_epoch   = 100  # inclusive
save_dir    = "model_weights"
os.makedirs(save_dir, exist_ok=True)

model.train()
for epoch in range(start_epoch, end_epoch + 1):
    running_loss = 0.0
    num_updates  = 0
    loop         = tqdm(dataloader, desc=f"Epoch {epoch}/{end_epoch}", leave=False)

    for batch_idx, (frames_list, transcripts) in enumerate(loop):
        frames     = frames_list[0].unsqueeze(0).to(device)
        transcript = transcripts[0]
        if not transcript:
            continue

        # forward
        outputs = model(frames).log_softmax(2)

        # prepare targets
        target_indices = torch.tensor(text_to_indices(transcript),
                                      dtype=torch.long, device=device)
        input_length   = torch.tensor([outputs.size(0)],     device=device)
        target_length  = torch.tensor([target_indices.size(0)], device=device)

        # loss + backward
        loss = ctc_loss(outputs,
                        target_indices.unsqueeze(0),
                        input_length,
                        target_length)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        num_updates  += 1
        loop.set_postfix(loss=loss.item())

    # save checkpoint
    save_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
    torch.save(model.state_dict(), save_path)
    avg_loss = running_loss / max(num_updates, 1)
    print(f"[✔] Saved {save_path} — avg loss: {avg_loss:.4f}")

print("✅ Continued training complete.")



# %%


def greedy_decoder(output):
    # output: (time, batch, num_classes) -> assume batch size 1
    output = output.cpu().detach().numpy()
    output = output[:, 0, :]
    decoded = []
    prev = -1
    for t in range(output.shape[0]):
        best = output[t].argmax()
        if best != prev and best != 0:  # 0 is the CTC blank
            decoded.append(best)
        prev = best
    return indices_to_text(decoded)

# Set model to evaluation mode
model.eval()

# Run inference on a sample video
if sample_video is not None:
    sample_frames = extract_mouth_frames(sample_video, detector, predictor)
    sample_frames = sample_frames.astype(np.float32) / 255.0
    sample_frames = np.expand_dims(sample_frames, axis=1)  # (T, 1, H, W)
    sample_tensor = torch.tensor(sample_frames).unsqueeze(0).to(device)  # (1, T, 1, H, W)

    with torch.no_grad():
        output = model(sample_tensor)  # (time, 1, num_classes)

    prediction = greedy_decoder(output)
    print("Predicted transcript:", prediction)
else:
    print('No sample video available for inference.')


# ## 7. (Optional) Save Inference Results in `.align` Format
# 
# If desired, you can post-process the model's output to create an `.align` file. For example, assuming each predicted character corresponds to a segment of time, you could map frame indices to timestamps. This is left as an exercise to tailor the timing resolution to your needs.

# ## Conclusion
# 
# This notebook provides a simplified end-to-end pipeline for building and training a LipNet-inspired lip reading model. For a production-ready system, consider:
# 
# - Expanding the dataset (using the full GRID corpus or similar)
# - Refining the preprocessing (e.g. better mouth tracking, data augmentation)
# - Tuning the model architecture and hyperparameters
# - Implementing a more sophisticated decoder (e.g. beam search with an external language model) for improved accuracy
# 
# Happy coding!
