import tflite_runtime.interpreter as tflite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load validation data
data = pd.read_csv("data/validation_data.csv")

# Prepare sequences
sequence_length = 10
sequences = [data.iloc[i:i+sequence_length].values for i in range(len(data) - sequence_length)]
sequences = np.array(sequences, dtype=np.float32)

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="models/anomaly_detector.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
predictions = []
for sequence in sequences:
    interpreter.set_tensor(input_details[0]['index'], [sequence])
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])

# Plot anomaly scores
plt.figure(figsize=(10, 6))
plt.plot(predictions, label="Anomaly Scores")
plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
plt.title("Validation Results - Anomaly Detection")
plt.xlabel("Sequence Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.show()
