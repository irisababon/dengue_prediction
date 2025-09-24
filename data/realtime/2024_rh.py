from meteostat import Hourly, Stations
from datetime import datetime

stations = Stations()
station = stations.nearby(14.6333, 121.0167).fetch(1)  # Science Garden
station_id = station.index[0]

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31, 23, 59)

# Fetch hourly data
data = Hourly(station_id, start, end)
data = data.fetch()

# Convert hourly humidity to daily average
daily_humidity = data[['rhum']].resample('D').mean()
daily_humidity.to_csv('data/historical/csv_files/2024rh.csv')