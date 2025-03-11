import pyshark
import tflite_runtime.interpreter as tflite
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="models/anomaly_detector.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define feature preprocessing
scaler = MinMaxScaler()

def preprocess_packet(packet):
    try:
        features = [
            int(packet.length),
            hash(packet.ip.src) % 1000,
            hash(packet.ip.dst) % 1000,
            hash(packet.transport_layer) % 10
        ]
        return scaler.fit_transform([features])[0]
    except AttributeError:
        return None

# Capture live packets
print("Starting live packet capture. Press Ctrl+C to stop.")
capture = pyshark.LiveCapture(interface="eth0")

for packet in capture.sniff_continuously(packet_count=10):
    features = preprocess_packet(packet)
    if features is not None:
        interpreter.set_tensor(input_details[0]['index'], [features])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"Anomaly Score: {prediction}")
