from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed validation data
data = pd.read_csv("validation_data.csv")  # Replace with your actual validation data file

# Normalize the data (just like during training)
scaler = MinMaxScaler()
data['length'] = scaler.fit_transform(data[['length']])

# Prepare sequences for LSTM model
sequence_length = 10  # Adjust this based on your model's design
sequences = [data.iloc[i:i+sequence_length].values for i in range(len(data) - sequence_length)]
sequences = np.array(sequences)

# Load the pre-trained model
model = load_model("models/anomaly_detector.h5")

# Predict anomalies
predictions = model.predict(sequences)

# Print a few anomaly scores for reference
print("First 10 Anomaly Scores:")
print(predictions[:10])

# Plot anomaly scores
plt.figure(figsize=(10, 6))
plt.plot(predictions, label="Anomaly Scores")
plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
plt.title("Anomaly Detection Results")
plt.xlabel("Sequence Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.show()
