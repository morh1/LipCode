"""
Mouth Extraction and Preprocessing Module
This module extracts the mouth region from each frame of a given video using dlib facial landmarks,
resizes it to 112x112, and saves the preprocessed frames as a .npy file for further processing
in lip reading models.

Outputs:
- Saved .npy file containing normalized grayscale mouth frames of shape (T, 1, 112, 112)
"""
import os
import cv2
import dlib
import numpy as np

# Paths relative to current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
PREDICTOR_PATH = os.path.join("utility", "shape_predictor_68_face_landmarks.dat")
# Ensure required predictor model exists
if not os.path.isfile(PREDICTOR_PATH):
    raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat")
# Ensure processed directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

"""
    Extracts and resizes the mouth region from a given video frame using dlib facial landmarks.
    Args:
        frame (np.ndarray): A single video frame in BGR format (as read by OpenCV).
    Returns:
        np.ndarray or None: Grayscale resized (112x112) image of the mouth region,
                            or None if no face or mouth region was detected.
"""
def extract_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    # Detect facial landmarks on the first face
    shape = predictor(gray, faces[0])
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
    # Get mouth bounding box with margin
    x, y, w, h = cv2.boundingRect(points)
    margin = 10
    y1, y2 = max(0, y - margin), y + h + margin
    x1, x2 = max(0, x - margin), x + w + margin
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (112, 112))

"""
    Processes a video to extract and normalize mouth region frames.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        str or None: Path to the saved .npy file containing processed frames,
                     or None if no valid frames were found or the video could not be read.
"""
def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mouth = extract_mouth(frame)
        if mouth is not None:
            frames.append(mouth)

    cap.release()

    if not frames:
        print("❌ No valid mouth frames found.")
        return None

    frames = np.stack(frames)[:, np.newaxis, :, :].astype(np.float32) / 255.0

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_frames.npy")
    np.save(out_path, frames)
    return out_path
