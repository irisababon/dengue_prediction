import matplotlib.pyplot as plt
import pandas 
import numpy

column_names = ['YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH']
mdata = pandas.read_csv('data/historical/csv_files/meteorological.csv', names=column_names, header=0)
mdata.head()

mdata.replace(-999, numpy.nan, inplace=True)
mdata.fillna(method='ffill', inplace=True)
mdata.to_csv('data/historical/csv_files/meteorological_cleaned.csv', index=False)

#*=====================================*
#*   data visualization for testing    *
#*=====================================*

mdata['DATE'] = pandas.to_datetime(mdata[['YEAR', 'MONTH', 'DAY']])

#rainfall
plt.figure(figsize=(15, 6))
plt.plot(mdata['DATE'], mdata['RAINFALL'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
plt.title('Rainfall', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

#mean temperature
plt.figure(figsize=(15, 6))
plt.plot(mdata['DATE'], mdata['TMEAN'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
plt.title('Mean Temperature', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (Celsius)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

#relative humidity
plt.figure(figsize=(15, 6))
plt.plot(mdata['DATE'], mdata['RH'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
plt.title('Relative Humidity', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Relative Humidity (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()