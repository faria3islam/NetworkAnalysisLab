from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import tensorflow as tf

# Load preprocessed training data
data = pd.read_csv("preprocessed_training_data.csv")
sequence_length = 10

# Prepare sequences for LSTM
sequences = []
for i in range(len(data) - sequence_length):
    sequences.append(data.iloc[i:i+sequence_length].values)

X_train = np.array(sequences)
y_train = np.zeros((X_train.shape[0], 1))  # Assuming all data is normal for unsupervised training

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1, activation='sigmoid')  # Output a single score for anomaly detection
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model as .h5 file
model.save("anomaly_detector.h5")
print("Model training complete. Model saved as 'anomaly_detector.h5'.")

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("anomaly_detector.tflite", "wb") as f:
    f.write(tflite_model)
print("TensorFlow Lite model saved as 'anomaly_detector.tflite'.")
