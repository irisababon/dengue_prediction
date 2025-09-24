import matplotlib.pyplot as plt
import pandas
import seaborn
import numpy
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import savgol_filter    

print("BEGIN")

column_names = ['date','Cases','Rainfall','Temperature','RH','searches1','searches2']
mdata = pandas.read_csv('data/historical/csv_files/websiteSmooth.csv', names=column_names, header=0)
mdata.head()

mdata['date'] = pandas.to_datetime(mdata['date'])
mdata.set_index('date', inplace=True)

history = mdata[['Cases', 'Rainfall', 'Temperature', 'RH', 'searches1', 'searches2']].values

mdata2 = pandas.read_csv('data/historical/csv_files/websiteSmooth.csv', names=column_names, header=0)
mdata2.head()

mdata2['date'] = pandas.to_datetime(mdata2['date'])
mdata2.set_index('date', inplace=True)

history2 = mdata[['Cases', 'Rainfall', 'Temperature', 'RH', 'searches1', 'searches2']].drop(columns=['searches1', 'searches2']).values
cases_filtered = mdata2[["Cases"]].apply(savgol_filter,  window_length=31, polyorder=2)

# scaler = MinMaxScaler()
# history_scaled = scaler.fit_transform(history)
# history_scaled2 = scaler.fit_transform(history2)

scaler1 = MinMaxScaler()
history_scaled = scaler1.fit_transform(history) 

scaler2 = MinMaxScaler()
history_scaled2 = scaler2.fit_transform(history2) 

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
seq_length = 30
target_index = mdata.columns.get_loc('Cases')
X, y = create_sequences(history_scaled, seq_length)  # Use scaled data
X2, y2 = create_sequences(history_scaled2, seq_length)  # Use scaled data

# Split the data into training and testing sets
train_size = int(len(y) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train2, X_test2 = X2[:train_size], X2[train_size:]
y_train2, y_test2 = y2[:train_size], y2[train_size:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_train2 = torch.from_numpy(X_train2).float()
y_train2 = torch.from_numpy(y_train2).float()
X_test2 = torch.from_numpy(X_test2).float()
y_test2 = torch.from_numpy(y_test2).float()

# Reshape the input to match LSTM's expected input shape
X_train = X_train.reshape(X_train.shape[0], seq_length, X_train.shape[2])  # [batch_size, seq_length, input_size]
X_test = X_test.reshape(X_test.shape[0], seq_length, X_test.shape[2])  # [batch_size, seq_length, input_size]

X_train2 = X_train2.reshape(X_train2.shape[0], seq_length, X_train2.shape[2])  # [batch_size, seq_length, input_size]
X_test2 = X_test2.reshape(X_test2.shape[0], seq_length, X_test2.shape[2])  # [batch_size, seq_length, input_size]

# lstm model ========================================================================================================
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        if self.training:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 6
dropout = 0.5   # probability of dropout, so this should be in [0,1]
lstm_model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
lstm_model2 = LSTM(4, hidden_size, num_layers, 4, dropout)
learning_rate = 0.001
num_epochs = 100

# tracking loss  ====================================================================================================
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(lstm_model2.parameters(), lr=learning_rate)

lstm_model.load_state_dict(torch.load('model/backups/100epochs_withSearches.pth', weights_only=False))
lstm_model.eval()  # Set to evaluation mode
print("LSTM model loaded.")

lstm_model2.load_state_dict(torch.load('model/backups/100epochs_withoutSearches.pth', weights_only=False))
lstm_model2.eval()  # Set to evaluation mode
print("LSTM model #2 loaded.")


with torch.no_grad():
    train_outputs = lstm_model(X_train).squeeze().numpy()
    test_outputs = lstm_model(X_test).squeeze().numpy()

with torch.no_grad():
    train_outputs = lstm_model(X_train).squeeze()
    test_outputs = lstm_model(X_test).squeeze()
    train_outputs_cases = train_outputs[:, 0].numpy()
    test_outputs_cases = test_outputs[:, 0].numpy()

with torch.no_grad():
    train_outputs2 = lstm_model2(X_train2).squeeze().numpy()
    test_outputs2 = lstm_model2(X_test2).squeeze().numpy()

with torch.no_grad():
    train_outputs2 = lstm_model2(X_train2).squeeze()
    test_outputs2 = lstm_model2(X_test2).squeeze()
    train_outputs_cases2 = train_outputs2[:, 0].numpy()
    test_outputs_cases2 = test_outputs2[:, 0].numpy()

# Reshape test outputs to match the input size
test_outputs_cases_reshaped = test_outputs_cases.reshape(-1, 1)
test_outputs_cases_reshaped2 = test_outputs_cases2.reshape(-1, 1)
# Inverse transform only the 'Cases' column of the scaled data
# lstm_predictions = scaler.inverse_transform(numpy.hstack([test_outputs_cases_reshaped, numpy.zeros((test_outputs_cases_reshaped.shape[0], 3))]))[:, 0]
# lstm_predictions2 = scaler.inverse_transform(numpy.hstack([test_outputs_cases_reshaped2, numpy.zeros((test_outputs_cases_reshaped2.shape[0], 3))]))[:, 0]

# For model with searches (6 columns)
lstm_predictions = scaler1.inverse_transform(
    numpy.hstack([
        test_outputs_cases_reshaped,
        numpy.zeros((test_outputs_cases_reshaped.shape[0], 5))  # 6-1
    ])
)[:, 0]

# For model without searches (4 columns)
lstm_predictions2 = scaler2.inverse_transform(
    numpy.hstack([
        test_outputs_cases_reshaped2,
        numpy.zeros((test_outputs_cases_reshaped2.shape[0], 3))  # 4-1
    ])
)[:, 0]


# holt winters es =============================================================================================================
mdata2.index.freq = 'D'
train_size = int(len(mdata2) * 0.7)
hw_train, hw_test = mdata2[:train_size], mdata2[train_size:]
m = 12
alpha = 1/(2*m)

fitted_model = ExponentialSmoothing(hw_train['Cases'], trend='add', seasonal='add', seasonal_periods=365).fit()
hw_predictions = fitted_model.forecast(len(hw_test))

hw_test = hw_test['Cases']
hw_predictions.index = hw_test.index
hw_predictions = hw_predictions[:-30]

# =============================================================================================================================


ensemble1 = lstm_predictions[:-21] * 1 + hw_predictions * (1-1)
# ensemble2 = lstm_predictions[:-21] * 0.95629 + hw_predictions * (1-0.95629)
# ensemble3 = lstm_predictions[:-21] * 0.50 + hw_predictions * (1-0.50)

# #mae1 = mean_absolute_error(hw_test[:-30], ensembleMAE)
rmse1 = numpy.sqrt(mean_squared_error(hw_test[:-30], ensemble1))
# rmse2 = numpy.sqrt(mean_squared_error(hw_test[:-30], ensemble2))
# rmse3 = numpy.sqrt(mean_squared_error(hw_test[:-30], ensemble3))

# rmse_list = []

# for i in range(1, 100):
#     ensemble = lstm_predictions[:-21] * i/100 + hw_predictions * (1 - i/100)
#     rmse = numpy.sqrt(mean_squared_error(hw_test[:-30], ensemble))
#     rmse_list.append(rmse)

print(rmse1)

# mae2 = mean_absolute_error(hw_test[:-30], ensembleMSE)
# mse2 = mean_squared_error(hw_test[:-30], ensembleMSE)
# rmse2 = numpy.sqrt(mse2)

# forecasting =======================================================================================================

num_forecast_steps = 30
sequence_to_plot = X_test.squeeze().cpu().numpy()
historical_data = sequence_to_plot[-1]

forecasted_values = []
with torch.no_grad():
    for _ in range(num_forecast_steps):
        historical_data_tensor = torch.as_tensor(historical_data).float().unsqueeze(0)
        predicted_value = lstm_model(historical_data_tensor).numpy()[0, 0]
        forecasted_values.append(predicted_value)
        historical_data = numpy.roll(historical_data, shift=-1)
        historical_data[-1] = predicted_value

last_date = mdata.index[-1]
future_dates = pandas.date_range(start=last_date + pandas.DateOffset(1), periods=num_forecast_steps)


target_index = mdata.columns.get_loc('Cases')
target_min = scaler1.data_min_[target_index]
target_max = scaler1.data_max_[target_index]
forecasted_cases = numpy.array(forecasted_values) * (target_max - target_min) + target_min

plt.plot(hw_test.index[:-30], lstm_predictions[:-21], label='LSTM with searches', color='darkorange')
# plt.plot(hw_test.index[:-30], lstm_predictions2[:-21], label='LSTM without searches', color='teal')

# plt.plot(hw_predictions.index, hw_predictions, label='Holt-Winters')

plt.plot(hw_test.index, mdata2[["Cases"]][train_size:], label='Actual Cases', linestyle='-')

plt.title('Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.tight_layout()
plt.show()

# # plt.plot(mdata['Cases'], label="True Values")
# #lstm_predictions = pandas.Series(ensemble1, index=hw_test.index)
# plt.plot(lstm_predictions, legend=True, label='LSTM')
# plt.plot(lstm_predictions2, legend=True, label='LSTM')
# # pandas.Series(ensemble2, index=hw_test.index).plot(legend=True, label='5% Holt Winters')
# # pandas.Series(ensemble3, index=hw_test.index).plot(legend=True, label='50% Holt Winters')

plt.plot(
    mdata.index[-1:].append(future_dates), 
    numpy.concatenate([mdata['Cases'][-1:].values, forecasted_cases]),
    label='forecasted values', 
    color='orange'
)

# # plt.plot(rmse_list, label = "")
# # plt.xlabel("LSTM Percentage")
# # plt.ylabel("RMSE")

# #plt.plot(lstm_forecast, label="Predicted")
# plt.legend()
# plt.show()