import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples = 1000  # Total number of samples
seq_length = 10   # Length of each input sequence
n_features = 1    # Number of features (e.g., 1 for a single time series)

# Generate synthetic data
def generate_data(n_samples, seq_length):
    X = []
    y = []
    for i in range(n_samples):
        # Generate a sequence of numbers
        seq = np.arange(i, i + seq_length)
        X.append(seq)
        y.append(i + seq_length)  # Next number in the sequence
    return np.array(X), np.array(y)

# Generate dataset
X, y = generate_data(n_samples, seq_length)

# Reshape data for RNN input: (batch_size, seq_length, n_features)
X = X.reshape((n_samples, seq_length, n_features)).astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass through RNN layer
        out, _ = self.rnn(x)
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Hyperparameters
input_size = n_features
hidden_size = 32  # Number of RNN units
output_size = 1   # Predicting the next number

# Initialize model
model = RNNModel(input_size, hidden_size, output_size)

# Hyperparameters
learning_rate = 0.01
num_epochs = 100

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test with a new sequence
test_seq = np.array([[90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
test_seq = test_seq.reshape((1, seq_length, n_features)).astype(np.float32)
test_seq_tensor = torch.from_numpy(test_seq)

# Get the prediction
with torch.no_grad():
    predicted = model(test_seq_tensor)
    print(f'Input sequence: {test_seq.flatten()}')
    print(f'Predicted next number: {predicted.item()}')