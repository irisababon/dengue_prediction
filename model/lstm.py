import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error

print("BEGIN TRAINING")

device = ""
if torch.cuda.is_available(): device = "cuda"
elif torch.mps.is_available(): device = "mps"
else: device="cpu"
device

# https://medium.com/@mike.roweprediger/using-pytorch-to-train-an-lstm-forecasting-model-e5a04b6e0e67
# data processing ===================================================================================================
column_names = ['date','Cases','Rainfall','Temperature','RH','searches1','searches2']
mdata = pandas.read_csv('data/historical/csv_files/finalSmooth.csv', names=column_names, header=0)
mdata.head()
mdata['date'] = pandas.to_datetime(mdata['date'])
mdata.set_index('date', inplace=True)

history = mdata[['Cases', 'Rainfall', 'Temperature', 'RH', 'searches1', 'searches2']].values

scaler = MinMaxScaler()
history_scaled = scaler.fit_transform(history)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return numpy.array(xs), numpy.array(ys)

seq_length = 30
target_index = mdata.columns.get_loc('Cases')
X, y = create_sequences(history_scaled, seq_length)  # Use scaled data

train_size = int(len(y) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_train = X_train.reshape(X_train.shape[0], seq_length, X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], seq_length, X_test.shape[2])

X_train_gpu = X_train.to(device)
y_train_gpu = y_train.to(device)

# lstm model ======================================================================================================

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        self.dropout_layer = nn.Dropout(dropout)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 6
dropout = 0.5   # probability of dropout, so this should be in [0,1]
model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
learning_rate = 0.001
num_epochs = 180

# tracking loss  ====================================================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Tracking loss to make sure we are not overfitting!
train_losses = []
val_losses = []

# training loop =====================================================================================================
bestLoss = 1e9
patience = 10 # if it goes down this may times in a row we should stop before it goes insane
tick = 0
for epoch in range(num_epochs):
    outputs = model(X_train).squeeze()  # Pass the input through the model
    optimizer.zero_grad()  # Zero the gradients
    loss = criterion(outputs, y_train)  # Calculate the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    # track loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test).squeeze()
        val_loss = criterion(val_outputs, y_test.squeeze())

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    # if(epoch > 15):
    #     if(val_losses[-1] > val_losses[-2]):
    #         print("FAIL. BREAKING")
    #         break

    # print(tick)
    curLoss = loss.item()
    if(curLoss > bestLoss):
        tick += 1
    if(bestLoss > curLoss): # there has been an improvement!
        tick = 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Improvement: {bestLoss - curLoss:.8f}")
    bestLoss = min(curLoss, curLoss)
    # stops training if validation loss goes up =====================================================================
    # if(tick >= patience):
    #     print("PATIENCE EXHAUSTED. ENDING HERE.")
    #     break

X_train_cpu = X_train_gpu.cpu()
y_train_cpu = y_train_gpu.cpu()

# evaluation ========================================================================================================

with torch.no_grad():
    test_outputs = model(X_test).squeeze()  # Get the test predictions
    test_outputs_cases = test_outputs[:, 0].numpy()
    y_test_cases = y_test[:, 0].numpy()
    test_loss = criterion(torch.tensor(test_outputs_cases), torch.tensor(y_test_cases))  # Squeeze y_train to match the dimensions

    print(f"Test Loss: {test_loss.item():.4f}")

# Concatenate the training and test predictions for plotting
with torch.no_grad():
    train_outputs = model(X_train_cpu).squeeze().numpy()
    test_outputs = model(X_test).squeeze().numpy()

    train_outputs_cases = train_outputs[:, 0]
    test_outputs_cases = test_outputs[:, 0]

all_outputs = numpy.concatenate((train_outputs_cases, test_outputs_cases))


test_start_index = len(history_scaled) - len(y_test) - seq_length

torch.save(model.state_dict(), "model/forecasting_test1.pth")

mae = mean_absolute_error(y_test.numpy(), test_outputs)
mse = mean_squared_error(y_test.numpy(), test_outputs)
rmse = numpy.sqrt(mse)

# forecasting =======================================================================================================

num_forecast_steps = 500
sequence_to_plot = X_test.squeeze().cpu().numpy()
historical_data = sequence_to_plot[-1]

forecasted_values = []
with torch.no_grad():
    for _ in range(num_forecast_steps):
        historical_data_tensor = torch.as_tensor(historical_data).float().unsqueeze(0)
        predicted_value = model(historical_data_tensor).numpy()[0, 0]
        forecasted_values.append(predicted_value)
        historical_data = numpy.roll(historical_data, shift=-1)
        historical_data[-1] = predicted_value

last_date = mdata.index[-1]
future_dates = pandas.date_range(start=last_date + pandas.DateOffset(1), periods=num_forecast_steps)


# for testing =======================================================================================================

print("==========================")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print("==========================")
print(f'Mean of Cases: {mdata["Cases"].mean()}')

history_scaled_cases = history_scaled[:, 0]

test_outputs_cases_reshaped = test_outputs_cases.reshape(-1, 1)
# Inverse transform only the 'Cases' column of the scaled data
lstm_predictions = scaler.inverse_transform(numpy.hstack([test_outputs_cases_reshaped, numpy.zeros((test_outputs_cases_reshaped.shape[0], 5))]))[:, 0]

target_index = mdata.columns.get_loc('Cases')
target_min = scaler.data_min_[target_index]
target_max = scaler.data_max_[target_index]
forecasted_cases = numpy.array(forecasted_values) * (target_max - target_min) + target_min


# Plot the true values
plt.plot(mdata['Cases'], label="True Values")
plt.plot(mdata.index[-1:].append(future_dates), numpy.concatenate([mdata['Cases'][-1:].values, forecasted_cases]),label='forecasted values', color='red')

#plt.plot(range(test_start_index, test_start_index + len(test_outputs_cases)), test_outputs_cases, label="Test Predictions", color="orange")
#plt.axvline(x=test_start_index, color='gray', linestyle='--', label="Test Set Start")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("LSTM Predictions")
plt.show()

# print("Losses:")
# print(train_losses)
# print(val_losses)
# Plotting loss (for overfitting checks)
# generally you want them both to descend smoothly
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss Curve")
plt.show()