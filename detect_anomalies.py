import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the preprocessed data
data = pd.read_csv("data/preprocessed_data.csv")

# Prepare the data for the LSTM model
sequence_length = 10  # Adjust the sequence length as needed
sequences = [data.iloc[i:i+sequence_length].values for i in range(len(data) - sequence_length)]
sequences = np.array(sequences)

# Load the pre-trained TensorFlow model
model = load_model("models/anomaly_detector.h5")

# Predict anomalies
predictions = model.predict(sequences)
print("Anomaly Scores:", predictions[:10])  # Print the first 10 anomaly scores

# Visualize anomaly scores
plt.plot(predictions, label="Anomaly Scores")
plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold")
plt.title("Anomaly Detection Scores")
plt.xlabel("Sequence Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.show()
