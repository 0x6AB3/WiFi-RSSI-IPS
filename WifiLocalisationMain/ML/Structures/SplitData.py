import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Training/Testing data split for classical models
def split_data_classical(ports, csv, testsplit):
    # Retrieving all RSSI values from each available port
    rssi = csv[[f"rssi_for_port{port}" for port in ports]]
    # Retrieving the locations for the given RSSIs
    distance = csv[["x", "y"]]
    x_train, x_test, y_train, y_test = train_test_split(rssi, distance, test_size=testsplit, random_state=1)
    # Casting to numpy arrays to remove automatic cast warnings
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

# Feedforward Neural Network data split (uses tensors instead of numpy arrays)
def split_data_nn(csv, ports, testsplit):
    xy = csv.groupby(['x', 'y'])

    x_train, y_train, x_test, y_test = [], [], [], []
    for (x_position, y_position), location in xy:
        X = location[[f"rssi_for_port{port}" for port in ports]].values
        Y = location[["x", "y"]].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsplit, random_state=1)

        # Append to lists
        x_train.append(X_train)
        y_train.append(Y_train)
        x_test.append(X_test)
        y_test.append(Y_test)

    x_train_combined = np.vstack(x_train)
    y_train_combined = np.vstack(y_train)
    x_test_combined = np.vstack(x_test)
    y_test_combined = np.vstack(y_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_combined)
    x_test_scaled = scaler.transform(x_test_combined)

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_combined, dtype=torch.float32)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor