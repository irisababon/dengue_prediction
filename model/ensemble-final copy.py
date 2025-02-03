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
column_names = ['date', 'YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH', 'dengue', 'Cases', 'searches']
mdata = pandas.read_csv('data/historical/csv_files/final.csv', names=column_names, header=0)
mdata.head()

mdata["monthday"] = mdata['MONTH'].map(str) + mdata['DAY'].map(str)
print(mdata['monthday'])

group_by_both = mdata.groupby(["YEAR", "monthday"]).agg({'Cases': 'sum'})
mdata['YEAR'] = pandas.Categorical(mdata['YEAR'], categories=pandas.unique(mdata['YEAR']))
mdata['monthday'] = pandas.Categorical(mdata['monthday'], categories=pandas.unique(mdata['monthday']))
heatmap1_data = pandas.pivot_table(group_by_both, values='Cases',
                     index='YEAR',
                     columns='monthday')
print(heatmap1_data)

sns_plot = seaborn.heatmap(heatmap1_data, cmap="YlGnBu")
plt.show()