import matplotlib.pyplot as plt
import pandas 
import numpy

# reading csv files
column_names = ['date', 'YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'WIND_SPEED', 'WIND_DIRECTION', 'RH', 'dengue']
wsdata = pandas.read_csv('data/historical/csv_files/weather_searches.csv', names=column_names, header=0)
wsdata.head()

ddata = pandas.read_csv('data/historical/csv_files/dengue_cleaned.csv', names=['Date', 'Cases'], header=0)
ddata.head()

combined_data = pandas.merge(wsdata, ddata, left_index=True, right_index=True, how='left')
combined_data.to_csv('data/historical/csv_files/final.csv')

#*=====================================*
#*   data visualization for testing    *
#*=====================================*

plt.figure(figsize=(15, 6))
plt.plot(combined_data.index, combined_data['dengue'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
combined_data['Cases'].plot(legend=True, label='TEST', color='orange')
plt.title('cases', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('cases', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()