import matplotlib.pyplot as plt
import pandas
import seaborn
import numpy
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.feature_selection import r_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Load the data
column_names = ['date','Cases','Rainfall','Temperature','RH','searches1','searches2']
mdata = pandas.read_csv('data/historical/csv_files/finalSmooth.csv', names=column_names, header=0)
mdata.head()

mdata['date'] = pandas.to_datetime(mdata['date'])
mdata.set_index('date', inplace=True)
plt.plot(mdata['Cases'], antialiased=True, linewidth=1, label="Dengue Cases")
plt.plot(mdata['searches1'], label="Searches for 'dengue'")
plt.plot(mdata['searches2'], color='green', label="Searches for 'dengue symptoms'")
plt.legend(loc="upper right")
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()