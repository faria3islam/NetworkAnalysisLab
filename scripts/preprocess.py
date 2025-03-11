import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load raw traffic data
data = pd.read_csv("data/traffic_data.csv", names=["time", "src_ip", "dst_ip", "length", "protocols"])

# Convert 'time' to datetime and then to a numerical timestamp
data['time'] = pd.to_datetime(data['time'], errors='coerce')  # Convert to datetime
data['time'] = data['time'].astype('int64') // 10**9  # Convert to Unix timestamp (seconds)

# Drop non-numeric columns (IP addresses) since they are categorical
data = data.drop(columns=["src_ip", "dst_ip"])

# Convert categorical 'protocols' into numerical features using one-hot encoding
data = pd.get_dummies(data, columns=["protocols"])

# Normalize numerical columns ('time' and 'length') using MinMaxScaler
scaler = MinMaxScaler()
data[['time', 'length']] = scaler.fit_transform(data[['time', 'length']])

# Save the processed data for training and detection
data.to_csv("data/preprocessed_data.csv", index=False)
print("Preprocessing complete. File saved to 'data/preprocessed_data.csv'")
