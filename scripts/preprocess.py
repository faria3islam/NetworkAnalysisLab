import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV file
data = pd.read_csv("data/traffic_data.csv", names=["time", "src_ip", "dst_ip", "length", "protocols"])

# Convert categorical 'protocols' into numerical features using one-hot encoding
data = pd.get_dummies(data, columns=["protocols"])

# Normalize 'length' using MinMaxScaler
scaler = MinMaxScaler()
data['length'] = scaler.fit_transform(data[['length']])

# Save the preprocessed data to a new CSV file
data.to_csv("data/preprocessed_data.csv", index=False)
print("Preprocessing complete. File saved to 'data/preprocessed_data.csv'")
