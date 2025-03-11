from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import tensorflow as tf

# Load preprocessed training data
data = pd.read_csv("data/preprocessed_training_data.csv")

# Prepare sequences for LSTM
sequence_length = 10
sequences = [data.iloc[i:i+sequence_length].values for i in range(len(data) - sequence_length)]
X_train = np.array(sequences)
y_train = np.zeros((X_train.shape[0], 1))  # Assuming all data is normal

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1, activation='sigmoid')  # Anomaly score
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the full model as .h5
model.save("models/anomaly_detector.h5")
print("Model saved as 'models/anomaly_detector.h5'.")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("models/anomaly_detector.tflite", "wb") as f:
    f.write(tflite_model)
print("Model saved as 'models/anomaly_detector.tflite'.")
