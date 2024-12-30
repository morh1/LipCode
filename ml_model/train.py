import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import functions from prepro.py
from prepro import process_and_save_data, load_data, preprocess_for_training

# Import the model function from model.py
from model import create_lip_reading_model

# Directory paths
VIDEOS_DIR = "san_tests/raw"  # Directory where the videos are stored
TRANSCRIPTIONS_FILE = "san_tests/transcriptions.csv"  # CSV with video names and transcriptions
TRAIN_DIR = "san_tests/train"  # Directory to store training data
VAL_DIR = "san_tests/validation"  # Directory to store validation data
PROCESSED_DATA_DIR = "san_tests/processed"  # Processed .npz files

# Ensure necessary directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Step 1: Process and Save Data
print("Processing and saving data...")
process_and_save_data(VIDEOS_DIR, TRANSCRIPTIONS_FILE, TRAIN_DIR, VAL_DIR)
print("Data processing complete!")

# Step 2: Load the Processed Data
print("Loading processed data...")
X_train_raw, y_train_raw = load_data(TRAIN_DIR)
X_val_raw, y_val_raw = load_data(VAL_DIR)

# Step 3: Preprocess Data for Training
print("Preprocessing data for training...")
frames, encoded_transcriptions, X_train, X_val, y_train, y_val, label_encoder = preprocess_for_training(
    np.concatenate(X_train_raw),
    np.concatenate(y_train_raw),
)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Number of classes: {len(label_encoder.classes_)}")

# Step 4: Create the Model
print("Creating the model...")
input_shape = (128, 128, 3)  # Assuming each frame is 128x128 RGB
num_classes = len(label_encoder.classes_)  # Number of unique transcription classes
model = create_lip_reading_model(input_shape=input_shape, num_classes=num_classes)

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Print the model summary
model.summary()

# Step 5: Define Callbacks for Training
checkpoint_path = "lip_reading_model_best.h5"
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
    EarlyStopping(patience=5, monitor="val_loss", verbose=1),
]

# Step 6: Train the Model
print("Training the model...")
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# Step 7: Evaluate the Model
print("Evaluating the model...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Step 8: Save the Final Model
final_model_path = "lip_reading_model_final.h5"
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")
