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
column_names = ['date', 'YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH', 'dengue', 'Cases', 'searches']
mdata = pandas.read_csv('data/historical/csv_files/final.csv', names=column_names, header=0)
mdata.head()

mdata['Cases_Smooth'] = ExponentialSmoothing(mdata['Cases'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
mdata['Rainfall_Smooth'] = ExponentialSmoothing(mdata['RAINFALL'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
mdata['Temperature_Smooth'] = ExponentialSmoothing(mdata['TMEAN'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
mdata['RH_Smooth'] = ExponentialSmoothing(mdata['RH'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
mdata['searches1_Smooth'] = ExponentialSmoothing(mdata['dengue'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
mdata['searches2_Smooth'] = ExponentialSmoothing(mdata['searches'], trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues

mdata.set_index('date', inplace=True)

mdataSmooth = pandas.DataFrame({'Cases': mdata['Cases_Smooth'],
                                'Rainfall': mdata['Rainfall_Smooth'],
                                'Temperature': mdata['Temperature_Smooth'],
                                'RH': mdata['RH_Smooth'],
                                'searches1': mdata['searches1_Smooth'],
                                'searches2': mdata['searches2_Smooth']})

mdataSmooth.to_csv('data/historical/csv_files/finalSmooth.csv')

plt.show()

"""
mdata['DAY'] = mdata['DAY'].apply(lambda x: '{0:0>2}'.format(x))
mdata["monthday"] = ("0" + mdata['MONTH'].map(str) + "." + mdata['DAY'].map(str)).map(float)
print(mdata['monthday'])

group_by_both = mdata.groupby(["YEAR", "monthday"]).agg({'Cases': 'sum'})
heatmap1_data = pandas.pivot_table(group_by_both, values='Cases',
                     index='YEAR',
                     columns='monthday')
print(heatmap1_data)

sns_plot = seaborn.heatmap(heatmap1_data, cmap="YlGnBu")

plt.xlabel("Month-Day")
plt.ylabel("Year")
plt.show()
"""