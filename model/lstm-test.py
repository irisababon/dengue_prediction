import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy
import torch
import torch.nn as nn

# https://medium.com/@mike.roweprediger/using-pytorch-to-train-an-lstm-forecasting-model-e5a04b6e0e67
# Load the data
column_names = ['YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH']
mdata = pandas.read_csv('data/historical/meteorological_cleaned.csv', names=column_names, header=0)
mdata.head()
mdata = mdata.drop(columns = ['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'WIND_SPEED', 'WIND_DIRECTION'])

# Prepare the history data
history = list(mdata['RAINFALL'])

# Scale the data
scaler = MinMaxScaler()
history_scaled = scaler.fit_transform(numpy.array(history).reshape(-1, 1))  # Scale the data

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return numpy.array(xs), numpy.array(ys)

# Preprocess the historical data
seq_length = 100
target_index = mdata.columns.get_loc('RAINFALL')
X, y = create_sequences(history_scaled, seq_length)  # Use scaled data

# Split the data into training and testing sets
train_size = int(len(y) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Reshape the input to match LSTM's expected input shape
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)  # [batch_size, seq_length, input_size]
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)  # [batch_size, seq_length, input_size]

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

# Initialize the LSTM model
input_size = X_train.shape[2]
hidden_size = 16
num_layers = 1
output_size = 1
model = LSTM(input_size, hidden_size, num_layers, output_size)

# Set training parameters
learning_rate = 0.1
num_epochs = 500

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
bestLoss = 1e9
for epoch in range(num_epochs):
    outputs = model(X_train).squeeze()  # Pass the input through the model
    optimizer.zero_grad()  # Zero the gradients
    loss = criterion(outputs, y_train.squeeze())  # Calculate the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    # Print the loss every 10 epochs
    curLoss = loss.item()
    if(curLoss > bestLoss):
        continue
    bestLoss = min(curLoss, curLoss)
    if(bestLoss == curLoss):
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(X_test).squeeze()  # Get the test predictions
    test_loss = criterion(outputs, y_train.squeeze())  # Squeeze y_train to match the dimensions

    print(f"Test Loss: {test_loss.item():.4f}")

# Concatenate the training and test predictions for plotting
with torch.no_grad():
    train_outputs = model(X_train).squeeze().numpy()
    test_outputs = model(X_test).squeeze().numpy()

all_outputs = numpy.concatenate((train_outputs, test_outputs))

# Calculate the index where the test set starts
test_start_index = len(history_scaled) - len(y_test) - seq_length

# Plot the true values and the predictions
plt.plot(history_scaled, label="True Values")
plt.plot(range(seq_length, seq_length + len(all_outputs)), all_outputs, label="Predictions")
plt.axvline(x=test_start_index, color='gray', linestyle='--', label="Test set start")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("LSTM Predictions vs True Values")
plt.show()