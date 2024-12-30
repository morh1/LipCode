import cv2
import dlib
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load a pre-trained facial landmark predictor from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def preprocess_video(video_path, transcription, output_dir, video_name):
    """
    Preprocess a video to extract mouth regions (frames) and save them with the transcription.

    Args:
        video_path (str): Path to the input video file.
        transcription (str): The words spoken in the video (the transcription).
        output_dir (str): Directory to save the processed mouth regions and corresponding transcription.
        video_name (str): The name of the video file (used for naming the saved files).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []  # List to store processed mouth regions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the mouth region from the frame using the extract_mouth_region function.
        mouth_frame = extract_mouth_region(frame)

        # If a valid mouth region was detected, add it to the frames list.
        if mouth_frame is not None:
            frames.append(mouth_frame)

    cap.release()

    # Convert the frames list into a NumPy array (to make it easy to save as .npy file).
    processed_frames = np.array(frames)

    # Save the processed frames and corresponding transcription in a dictionary.
    video_data = {'frames': processed_frames, 'transcription': transcription}

    # Save the video data as a .npz file (a compressed NumPy format that stores multiple arrays).
    np.savez(f"{output_dir}/{video_name}.npz", **video_data)


def extract_mouth_region(frame):
    """
    Detect and crop the mouth region from a video frame.

    Args:
        frame (ndarray): A single video frame (in BGR format).

    Returns:
        ndarray: The cropped and resized mouth region (128x128 pixels) or None if no face is detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    for face in faces:
        landmarks = predictor(gray, face)
        mouth_points = [landmarks.part(i) for i in range(48, 68)]
        x_min = min([p.x for p in mouth_points])
        x_max = max([p.x for p in mouth_points])
        y_min = min([p.y for p in mouth_points])
        y_max = max([p.y for p in mouth_points])

        mouth_region = frame[y_min:y_max, x_min:x_max]
        if mouth_region.size == 0:
            continue

        mouth_region_resized = cv2.resize(mouth_region, (128, 128))
        return mouth_region_resized

    return None


def process_and_save_data(videos_dir, transcriptions_file, train_dir, val_dir):
    """
    Process the dataset by extracting frames from videos and saving them with transcriptions
    into separate directories for training and validation.

    Args:
        videos_dir (str): Path to the directory containing video files.
        transcriptions_file (str): Path to the CSV file containing video filenames and transcriptions.
        train_dir (str): Directory to save the training data.
        val_dir (str): Directory to save the validation data.
    """
    # Ensure the output directories exist. If not, create them.
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Read the transcriptions from the CSV file.
    with open(transcriptions_file, 'r') as f:
        reader = csv.reader(f)

        # Loop through each row in the CSV file
        for row in reader:
            # Extract the video filename and the transcription (sentence spoken in the video)
            video_filename = row[0]
            transcription = row[1]

            # Construct the full path to the video file
            video_path = os.path.join(videos_dir, video_filename)

            # Use the video filename (without extension) as the name for the saved file
            video_name = os.path.splitext(video_filename)[0]

            # Split data into train and validation sets (80% train, 20% validation)
            if np.random.rand() < 0.8:  # 80% for training, 20% for validation
                output_dir = train_dir
            else:
                output_dir = val_dir

            # Process the video and save the frames and transcription using the preprocess_video function
            preprocess_video(video_path, transcription, output_dir, video_name)


def load_data(data_dir):
    """
    Load processed data from .npz files in the given directory.

    Args:
        data_dir (str): Path to the directory containing the .npz files.

    Returns:
        frames_list (list): List of frames (mouth regions) from all videos.
        transcriptions_list (list): List of transcriptions corresponding to each video.
    """
    frames_list = []
    transcriptions_list = []

    # Loop through all .npz files in the data directory
    for npz_file in os.listdir(data_dir):
        if npz_file.endswith(".npz"):
            data = np.load(os.path.join(data_dir, npz_file))
            frames_list.append(data['frames'])  # Extract frames
            transcriptions_list.append(data['transcription'])  # Extract transcription

    # Convert lists to numpy arrays for ease of handling in models
    frames_array = np.array(frames_list)
    transcriptions_array = np.array(transcriptions_list)

    return frames_array, transcriptions_array


def preprocess_for_training(frames, transcriptions):
    """
    Preprocess the data for training the model.

    Args:
        frames (ndarray): Array of frames (mouth regions).
        transcriptions (ndarray): Array of transcriptions corresponding to each video.

    Returns:
        frames (ndarray): Normalized frames.
        transcriptions (ndarray): Encoded transcriptions.
        X_train, X_val, y_train, y_val (arrays): Training and validation splits.
    """
    # Normalize the frames to the range [0, 1]
    frames = frames.astype('float32') / 255.0

    # Encode the transcriptions (words/sentences) into numerical labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate(transcriptions))  # Fit on all transcriptions

    # Convert each transcription into a sequence of integers (numerical labels)
    encoded_transcriptions = [label_encoder.transform(list(transcription)) for transcription in transcriptions]

    # Split the data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(frames, encoded_transcriptions, test_size=0.2, random_state=42)

    return frames, encoded_transcriptions, X_train, X_val, y_train, y_val, label_encoder


if __name__ == "__main__":
    """
    Main entry point of the script to process the dataset.
    This part is executed when the script is run directly.
    """
    # Directory paths
    videos_dir = "data/raw"  # Directory where video files are stored
    transcriptions_file = "data/transcriptions.csv"  # Path to CSV containing video filenames and transcriptions
    train_dir = "data/train"  # Directory to save the training data
    val_dir = "data/validation"  # Directory to save the validation data

    # Process the dataset by extracting frames and saving them with corresponding transcriptions.
    process_and_save_data(videos_dir, transcriptions_file, train_dir, val_dir)

    # Load the processed data
    frames, transcriptions = load_data("data/train")

    # Preprocess the data for training
    frames, encoded_transcriptions, X_train, X_val, y_train, y_val, label_encoder = preprocess_for_training(frames,
                                                                                                            transcriptions)

    # Now, the data is ready to be fed into the model
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
