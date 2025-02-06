import requests
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

url = "https://api.tomorrow.io/v4/timelines"
load_dotenv()

api_key = os.getenv("API_KEY")

latitude = 14.6760
longitude = 121.0437

params = {
    "apikey": api_key,
    "location": f"{latitude},{longitude}",
    "fields": ["temperature"], 
    "timesteps": "1h",  
    "startTime": "2025-02-05T17:09:50Z",
    "endTime": "2025-02-06T17:09:50Z",
    "units": "metric"
}

headers = {
    "accept": "application/json",
    "Accept-Encoding": "gzip"
}

response = requests.get(url, params=params, headers=headers)

if response.status_code == 200:
    data = response.json()

    intervals = data["data"]["timelines"][0]["intervals"]

    csv_data = [["Time", "Temperature (°C)"]]
    for interval in intervals:
        time = interval["startTime"]
        temperature = interval["values"]["temperature"]
        csv_data.append([time, temperature])

    csv_file_name = "weather_data.csv"
    with open(csv_file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"Data successfully saved to {csv_file_name}")

else:
    print("Error:", response.status_code, response.text)