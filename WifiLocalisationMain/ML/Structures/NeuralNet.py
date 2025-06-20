import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class RandomRSSINN(nn.Module):
    def __init__(self, hidden_layer_count, min_neurons=32, max_neurons=128):
        super().__init__()
        self.layers = nn.ModuleList()

        # Random size for the first hidden layer
        prev_size = 3
        for _ in range(hidden_layer_count):
            layer_size = random.randint(min_neurons, max_neurons)
            self.layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size

        # Final output layer
        self.layers.append(nn.Linear(prev_size, 2))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# Uniform Neural Network: same number of neurons per hidden layer
class UniformRSSINN(nn.Module):
    def __init__(self, hidden_layer_count, neurons_per_layer):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3, neurons_per_layer))
        for _ in range(hidden_layer_count):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        self.layers.append(nn.Linear(neurons_per_layer, 2))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# Pyramid Neural Network: decreasing neurons per hidden layer
class PyramidRSSINN(nn.Module):
    def __init__(self, hidden_layer_count, hidden_first_layer_count=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3, hidden_first_layer_count))

        hidden_sizes = [
            int(hidden_first_layer_count * (hidden_layer_count - i) / hidden_layer_count)
            for i in range(hidden_layer_count)
        ]
        for i in range(hidden_layer_count - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], 2))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

def randomObjective(trial, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
    hidden_layers = trial.suggest_int("hidden_layers", 2, 10)
    min_neurons = trial.suggest_int("min_neurons", 32, 64)
    max_neurons = trial.suggest_int("max_neurons", 65, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    model = RandomRSSINN(hidden_layers, min_neurons, max_neurons)
    train_loss, test_loss = train_nn(model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, lr=learning_rate)
    return test_loss

# Optuna objective function for tuning PyramidRSSINN
def pyramidObjective(trial, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
    hidden_layers = trial.suggest_int("hidden_layers", 2, 10)
    max_neurons = trial.suggest_int("neurons_per_layer", 32, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    model = PyramidRSSINN(hidden_layers, max_neurons)
    train_loss, test_loss = train_nn(model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, lr=learning_rate)
    return test_loss

# Optuna objective function for tuning UniformRSSINN
def uniformObjective(trial, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
    hidden_layers = trial.suggest_int("hidden_layers", 2, 10)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 32, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    model = UniformRSSINN(hidden_layers, neurons_per_layer)
    train_loss, test_loss = train_nn(model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, lr=learning_rate)
    return test_loss

# Training function with RMSE reporting
def train_nn(model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, epochs=1000, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train_tensor, y_train_tensor = x_train_tensor.to(device), y_train_tensor.to(device)
    x_test_tensor, y_test_tensor = x_test_tensor.to(device), y_test_tensor.to(device)

    print("Model is on:", next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    rmse_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_tensor)
        mse_loss = criterion(predictions, y_train_tensor)
        mse_loss.backward()
        optimizer.step()

        # Store RMSE (sqrt of MSE)
        rmse_losses.append(math.sqrt(mse_loss.item()))

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], RMSE: {rmse_losses[-1]:.4f}")

    # Plot RMSE loss
    plt.figure()
    plt.plot(range(epochs), rmse_losses)
    plt.title("Training RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (cm)")
    plt.grid(True)
    plt.show()

    # Final RMSE evaluation
    model.eval()
    with torch.no_grad():
        train_rmse = math.sqrt(criterion(model(x_train_tensor), y_train_tensor).item())
        test_rmse = math.sqrt(criterion(model(x_test_tensor), y_test_tensor).item())
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    return train_rmse, test_rmse
