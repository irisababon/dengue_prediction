import matplotlib.pyplot as plt
import pandas 
import numpy

# reading csv files
column_names = ['YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH']
mdata = pandas.read_csv('data/historical/csv_files/2010meteorological_cleaned.csv', names=column_names, header=0)
mdata.head()

mdata['date'] = pandas.to_datetime(mdata[['YEAR', 'MONTH', 'DAY']])
gdata = pandas.read_csv('data/historical/csv_files/dengueSearchesfinal.csv', names=['date', 'dengue'], header=0, parse_dates=['date'])
gdata.head()

mdata.set_index('date', inplace=True)
gdata.set_index('date', inplace=True)

gdaily = gdata.resample('D').interpolate()

combined_data = pandas.merge(mdata, gdaily, left_index=True, right_index=True, how='left')

combined_data.to_csv('data/historical/csv_files/weather_searches.csv')

#*=====================================*
#*   data visualization for testing    *
#*=====================================*

plt.figure(figsize=(15, 6))
plt.plot(combined_data.index, combined_data['dengue'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
combined_data['RH'].plot(legend=True, label='TEST', color='orange')
plt.title('dengue search results', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('dengue search result', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()