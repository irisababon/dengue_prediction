import pandas
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error

# data ========================================================================================================================
column_names = ['YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH']
mdata = pandas.read_csv('data/historical/csv_files/2010meteorological_cleaned.csv', names=column_names, header=0)
mdata.head()
mdata = mdata.drop(columns = ['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'WIND_SPEED', 'WIND_DIRECTION'])

# parameters ==================================================================================================================
mdata.index.freq = 'D'
train_size = int(len(mdata) * 0.7)
temp_train, temp_test = mdata[:train_size], mdata[train_size:]
m = 12
alpha = 1/(2*m)

# holt winters es =============================================================================================================
fitted_model = ExponentialSmoothing(temp_train['TMEAN'], trend='add', seasonal='add', seasonal_periods=365).fit()
test_predictions = fitted_model.forecast(len(temp_test))
temp_train['TMEAN'].plot(legend=True, label='TRAIN')
temp_test['TMEAN'].plot(legend=True, label='TEST')
test_predictions.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()

temp_test = temp_test['TMEAN']
test_predictions.index = temp_test.index

# evaluation ==================================================================================================================
print(f'Mean Absolute Error = {mean_absolute_error(temp_test,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(temp_test,test_predictions)}')
# baselines
baseline_mae = mean_absolute_error(temp_test, [temp_train['TMEAN'].mean()] * len(temp_test))
baseline_mse = mean_squared_error(temp_test, [temp_train['TMEAN'].mean()] * len(temp_test))
print(f"Baseline MAE (mean): {baseline_mae}")
print(f"Baseline MSE (mean): {baseline_mse}")
print(f'Mean of TMEAN: {mdata["TMEAN"].mean()}')