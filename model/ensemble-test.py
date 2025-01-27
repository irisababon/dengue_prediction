import matplotlib.pyplot as plt
import pandas
import numpy
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# https://medium.com/@mike.roweprediger/using-pytorch-to-train-an-lstm-forecasting-model-e5a04b6e0e67
# Load the data
column_names = ['YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH', 'dengue']
mdata = pandas.read_csv('data/historical/csv_files/weather_searches.csv', names=column_names, header=0)
mdata.head()
mdata = mdata.drop(columns = ['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'WIND_SPEED', 'WIND_DIRECTION'])

mdata2 = pandas.read_csv('data/historical/csv_files/weather_searches.csv', names=column_names, header=0)
mdata2.head()
mdata2 = mdata2.drop(columns = ['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'WIND_SPEED', 'WIND_DIRECTION'])

# Prepare the history data
history = list(mdata['TMEAN'])

# Scale the data
scaler = MinMaxScaler()
history_scaled = scaler.fit_transform(numpy.array(history).reshape(-1, 1))  # Scale the data

# LSTM model ==================================================================================================================
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return numpy.array(xs), numpy.array(ys)

seq_length = 100
target_index = mdata.columns.get_loc('TMEAN')
X, y = create_sequences(history_scaled, seq_length)  # Use scaled data

train_size = int(len(y) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)

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

input_size = X_train.shape[2]
hidden_size = 16
num_layers = 1
output_size = 1
lstm_model = LSTM(input_size, hidden_size, num_layers, output_size)
learning_rate = 0.1
num_epochs = 500

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

lstm_model.load_state_dict(torch.load('lstm_model_full.pth', weights_only=True))
lstm_model.eval()  # Set to evaluation mode
print("LSTM model loaded.")

with torch.no_grad():
    train_outputs = lstm_model(X_train).squeeze().numpy()
    test_outputs = lstm_model(X_test).squeeze().numpy()

lstm_predictions = scaler.inverse_transform(test_outputs.reshape(-1, 1)).flatten()

# holt winters es =============================================================================================================
mdata2.index.freq = 'D'
train_size = int(len(mdata2) * 0.7)
hw_train, hw_test = mdata2[:train_size], mdata2[train_size:]
m = 12
alpha = 1/(2*m)

fitted_model = ExponentialSmoothing(hw_train['TMEAN'], trend='add', seasonal='add', seasonal_periods=365).fit()
hw_predictions = fitted_model.forecast(len(hw_test))

hw_test = hw_test['TMEAN']
hw_predictions.index = hw_test.index
hw_predictions = hw_predictions[:-30]

# =============================================================================================================================

ensemble1 = (lstm_predictions + hw_predictions)/2
ensemble2 = (lstm_predictions * 0.7 + hw_predictions * 0.3)

#hw_train['TMEAN'].plot(legend=True, label='TRAIN')
hw_test.plot(legend=True, label='TEST')
ensemble1.plot(legend=True, label='ENSEMBLE 1')
ensemble2.plot(legend=True, label='ENSEMBLE 2')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()

mae1 = mean_absolute_error(hw_test[:-30], ensemble1)
mse1 = mean_squared_error(hw_test[:-30], ensemble1)
rmse1 = numpy.sqrt(mse1)

mae2 = mean_absolute_error(hw_test[:-30], ensemble2)
mse2 = mean_squared_error(hw_test[:-30], ensemble2)
rmse2 = numpy.sqrt(mse2)

mae_baseline = mean_absolute_error(y_test.numpy(), [numpy.mean(history)] * len(test_outputs))
mse_baseline = mean_squared_error(y_test.numpy(), [numpy.mean(history)] * len(test_outputs))
rmse_baseline = math.sqrt(mse_baseline)

print("ensemble 1 ==========================")
print(f"MAE: {mae1}")
print(f"Baseline MAE: {mae_baseline}")
print("==========================")
print(f"MSE: {mse1}")
print(f"Baseline MSE: {mse_baseline}")
print("==========================")
print(f"RMSE: {rmse1}")
print(f"Baseline RMSE: {rmse_baseline}")
print("==========================")
print(f'Mean of TMEAN: {mdata["TMEAN"].mean()}')

print("ensemble 2 ==========================")
print(f"MAE: {mae2}")
print(f"Baseline MAE: {mae_baseline}")
print("==========================")
print(f"MSE: {mse2}")
print(f"Baseline MSE: {mse_baseline}")
print("==========================")
print(f"RMSE: {rmse2}")
print(f"Baseline RMSE: {rmse_baseline}")
print("==========================")
print(f'Mean of TMEAN: {mdata["TMEAN"].mean()}')