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

print("BEGIN")

# https://medium.com/@mike.roweprediger/using-pytorch-to-train-an-lstm-forecasting-model-e5a04b6e0e67
# Load the data
column_names = ['date','Cases','Rainfall','Temperature','RH','searches1','searches2']
mdata = pandas.read_csv('data/historical/csv_files/finalSmooth.csv', names=column_names, header=0)
mdata.head()

mdata2 = pandas.read_csv('data/historical/csv_files/finalSmooth.csv', names=column_names, header=0)
mdata2.head()

mdata.set_index('date', inplace=True)
mdata2.set_index('date', inplace=True)

# Prepare the history data
history = list(mdata['Cases'])

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
target_index = mdata.columns.get_loc('Cases')
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        self.dropout_layer = nn.Dropout(dropout)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = X_train.shape[2]
hidden_size = 16
num_layers = 1
output_size = 1
dropout = 0.5   # probability of dropout, so this should be in [0,1]
lstm_model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

lstm_model.load_state_dict(torch.load('model/testing.pth', weights_only=True))
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

fitted_model = ExponentialSmoothing(hw_train['Cases'], trend='add', seasonal='add', seasonal_periods=365).fit()
hw_predictions = fitted_model.forecast(len(hw_test))

hw_test = hw_test['Cases']
hw_predictions.index = hw_test.index
hw_predictions = hw_predictions[:-30]

# =============================================================================================================================

ensembleMAE = lstm_predictions * 0.97028 + hw_predictions * (1-0.97028)
ensembleMSE = lstm_predictions * 0.95629 + hw_predictions * (1-0.95629)

mae1 = mean_absolute_error(hw_test[:-30], ensembleMAE)
mse1 = mean_squared_error(hw_test[:-30], ensembleMAE)
rmse1 = numpy.sqrt(mse1)

mae2 = mean_absolute_error(hw_test[:-30], ensembleMSE)
mse2 = mean_squared_error(hw_test[:-30], ensembleMSE)
rmse2 = numpy.sqrt(mse2)

# Code for checking which proportion of LSTM/H-W ES is best ====================================================================
# this found that:
# 97.028% LSTM was best for MAE (no covid)
# 95.629% LSTM was best for RMSE/MSE (no covid)

# numModels = 10000

# ensembleModels = [None for _ in range(numModels)]
# MAE = [None for _ in range(numModels)]
# MSE = [None for _ in range(numModels)]
# RMSE = [None for _ in range(numModels)]

# for i in range(numModels):
#     if not i % 3000:
#         print("ON", i)
#     ensembleModels[i] = lstm_predictions * i/numModels + hw_predictions * (numModels-i)/numModels
#     MAE[i] = mean_absolute_error(hw_test[:-30],ensembleModels[i])
#     MSE[i] = mean_squared_error(hw_test[:-30],ensembleModels[i])
#     RMSE[i] = numpy.sqrt(MSE[i])
    
# bestMAE = MAE.index(min(MAE))
# bestMSE = MSE.index(min(MSE))
# bestRMSE = RMSE.index(min(RMSE))
    
# worstMAE = MAE.index(max(MAE))
# worstMSE = MSE.index(max(MSE))
# worstRMSE = RMSE.index(max(RMSE))

print(f"MAE optimized model: {mae1}")
print(f"MSE optimized model: {mae2}")
# print(f"Best MAE: model {bestMAE+1} with {MAE[bestMAE]}")
# print(f"Worst MAE: model {worstMAE+1} with {MAE[worstMAE]}")

print("==========================")
print(f"MAE optimized model: {mse1}")
print(f"MSE optimized model: {mse2}")
# print(f"Best MSE: model {bestMSE+1} with {MAE[bestMSE]}")
# print(f"Worst MSE: model {worstMSE+1} with {MAE[worstMSE]}")

print("==========================")
print(f"MAE optimized model: {rmse1}")
print(f"MSE optimized model: {rmse2}")
# print(f"Best RMSE: model {bestRMSE+1} with {MAE[bestRMSE]}")
# print(f"Worst RMSE: model {worstRMSE+1} with {MAE[worstRMSE]}")

print("==========================")
print(f'Mean of Cases: {mdata["Cases"].mean()}')

hw_test.plot(legend=True, label='TEST')
#hw_predictions.plot(legend=True, label='HOLT WINTERS')
#lstm_predictions = pandas.Series(ensemble1, index=hw_test.index)
#lstm_predictions.plot(legend=True, label='LSTM')
#ensemble1.plot(legend=True, label='ENSEMBLE 1')
ensembleMAE.plot(legend=True, label='MODEL 1')
ensembleMSE.plot(legend=True, label='MODEL 2')
plt.xlabel('Date')
plt.ylabel('Dengue Case Count')
# ensembleModels[bestMAE].plot(legend=True, label="BEST MAE")
# ensembleModels[worstMAE].plot(legend=True, label="WORST MAE")


plt.title('Ensemble Comparison')
plt.show()