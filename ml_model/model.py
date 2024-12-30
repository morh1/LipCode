import tensorflow as tf
from tensorflow.keras import layers, models


# Define the model architecture (CNN + Transformer)
def create_lip_reading_model(input_shape, num_classes, num_transformer_layers=4, d_model=256, num_heads=8, ff_dim=512):
    """
    Create a lip-reading model using CNN + Transformer for sequence modeling.

    Args:
        input_shape (tuple): Shape of the input frames (height, width, channels).
        num_classes (int): Number of unique transcription classes (words/characters).
        num_transformer_layers (int): Number of transformer encoder layers.
        d_model (int): Dimensionality of the transformer model.
        num_heads (int): Number of attention heads in each transformer layer.
        ff_dim (int): Feed-forward dimensionality in transformer layers.

    Returns:
        model: Keras model instance for lip-reading.
    """
    # Define the model
    inputs = layers.Input(shape=input_shape)

    # Convolutional layers for feature extraction from the mouth frames
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the CNN output to feed into transformer
    x = layers.Flatten()(x)
    x = layers.Dense(d_model)(x)  # Convert to the model's dimensionality

    # Transformer Encoder layers
    for _ in range(num_transformer_layers):
        # Multi-head self-attention mechanism
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        # Add & Normalize
        attention = layers.LayerNormalization()(attention + x)

        # Feed-forward layer
        ff = layers.Dense(ff_dim, activation='relu')(attention)
        ff_output = layers.Dense(d_model)(ff)
        # Add & Normalize
        x = layers.LayerNormalization()(ff_output + attention)

    # Global Average Pooling to get a single representation for the sequence
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers for final prediction
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = models.Model(inputs=inputs, outputs=output)

    return model


# Define input shape and number of classes
input_shape = (128, 128, 3)  # Assuming each frame is 128x128 RGB image
num_classes = len(label_encoder.classes_)  # Number of classes based on transcriptions

# Create the model
model = create_lip_reading_model(input_shape=input_shape, num_classes=num_classes)

# **Compile the Model** - This is where we compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer-labeled transcriptions
              metrics=['accuracy'])

# Print model summary to check the architecture
model.summary()

# **Train the Model**
# Train the model using your training data (X_train, y_train) and validation data (X_val, y_val)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# **Evaluate the Model**
# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
