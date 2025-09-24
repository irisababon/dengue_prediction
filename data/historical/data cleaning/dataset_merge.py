import matplotlib.pyplot as plt
import pandas 
import numpy

import pandas as pd

# mdata = pd.read_csv('data/historical/csv_files/may19dengue_searches.csv', names=['date', 'dengue'], header=0, parse_dates=['date'])
# mdata['date'] = pd.to_datetime(mdata['date'], errors='coerce')
# print(mdata)

# Read first file (daily weather data)
column_names = ['date', 'dengue']
mdata = pd.read_csv('data/historical/csv_files/may19_finalsearches.csv', 
                    names=['date', 'searches1', 'searches2'], header=0, parse_dates=['date'])
mdata['date'] = pd.to_datetime(mdata['date'])
mdata.set_index('date', inplace=True)

# Read second file (monthly dengue cases)
gdata = pd.read_csv('data/historical/csv_files/websiteSmooth.csv', 
                    names=['date','Cases','Rainfall','Temperature','RH','searches1','searches2'], header=0, parse_dates=['date'])
gdata.set_index('date', inplace=True)

print(gdata['searches1'].iloc[-1])
print(gdata['searches2'].iloc[-1])

print("==================")

date='2025-01-01'
print(mdata.loc[date, 'searches1'])
print(mdata.loc[date, 'searches2'])

# print(mdata.index[:5])
# print(gdata.index[:5])

# # Merge with mdata
# combined_data = pd.merge(mdata, gdata, left_index=True, right_index=True, how='left')

# # Save the merged data
# combined_data.to_csv('data/historical/csv_files/may19_finalsearches.csv')

# # # Preview result
# print(combined_data.tail())

# # Remove duplicate headers if any
# gdata = gdata[gdata['Month'] != 'Month']

# # Convert 'Month' column to datetime format
# gdata['Month'] = pd.to_datetime(gdata['Month'], errors='coerce')

# # Drop rows where date parsing failed
# gdata = gdata.dropna(subset=['Month'])

# Convert 'dengue' column to numeric
# mdata['dengue'] = pd.to_numeric(mdata['dengue'], errors='coerce')
# gdata['dengue'] = pd.to_numeric(gdata['dengue_symptoms'], errors='coerce')

# # Set Month as index
# gdata.set_index('Month', inplace=True)

# # Ensure the date range includes all days until Dec 31
# full_date_range = pd.date_range(start=gdata.index.min(), end='2024-12-31', freq='D')
# gdata = gdata.reindex(full_date_range)

# # Interpolate missing values
# gdaily = gdata.interpolate()


# #*=====================================*
# #*   data visualization for testing    *
# #*=====================================*

# plt.figure(figsize=(15, 6))
# plt.plot(combined_data.index, combined_data['dengue'], marker='.', linestyle='-', markersize=1, alpha=0.7, color='b')
# combined_data['RH'].plot(legend=True, label='TEST', color='orange')
# plt.title('dengue search results', fontsize=16)

# plt.xlabel('Date', fontsize=12)
# plt.ylabel('dengue search result', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()