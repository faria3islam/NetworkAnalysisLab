import pyshark
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained TensorFlow model
model = load_model("models/anomaly_detector.h5")

# Define a scaler for normalizing the data
scaler = MinMaxScaler()

# Function to preprocess each packet
def preprocess_packet(packet):
    try:
        # Extract basic features: length, source IP, destination IP, and protocol
        features = [
            int(packet.length),  # Packet length
            hash(packet.ip.src) % 1000,  # Hashed source IP
            hash(packet.ip.dst) % 1000,  # Hashed destination IP
            hash(packet.transport_layer) % 10  # Hashed protocol
        ]
        # Normalize features
        return scaler.fit_transform([features])[0]
    except AttributeError:
        # Skip packets without IP or transport layer
        return None

# Start live capture
print("Starting live packet capture. Press Ctrl+C to stop.")
capture = pyshark.LiveCapture(interface="eth0")

for packet in capture.sniff_continuously(packet_count=10):
    features = preprocess_packet(packet)
    if features is not None:
        # Reshape features for the LSTM model
        features = np.array(features).reshape(1, -1, 1)
        prediction = model.predict(features)
        print(f"Anomaly Score: {prediction[0][0]}")
