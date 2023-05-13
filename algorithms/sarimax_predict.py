from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
from matplotlib import pyplot as plt

from tokens import open_weather_map_token


def sarimax_predict(model, location):
    startPrediction = datetime.now()
    if startPrediction.minute >= 30:
        startPrediction = startPrediction.replace(second=0, microsecond=0, minute=0, hour=(startPrediction.hour + 1) % 24)
    else:
        startPrediction = startPrediction.replace(second=0, microsecond=0, minute=0)

    endPrediction = startPrediction + timedelta(hours=40)


    # API CALL
    geolocation = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct?q=" + location + "&appid=" + open_weather_map_token)
    geolocation = geolocation.json()

    geolocationData = geolocation[0]
    longitude = geolocationData['lon']
    latitude = geolocationData['lat']

    weatherForecast = requests.get(
        "https://api.openweathermap.org/data/3.0/onecall?lat=" + str(latitude) + "&lon=" + str(
            longitude) + "&exclude=minutely,daily&appid=" + open_weather_map_token)
    weatherForecast = weatherForecast.json()
    weatherForecastHourlyData = weatherForecast["hourly"]

    hourlyTimeData=[]
    hourlyTempData = []
    for i in range(48):
        hourlyData = weatherForecastHourlyData[i]["dt"]

        timestamp = pd.to_datetime(hourlyData, utc=False, unit='s')
        if(timestamp<startPrediction):
            continue
        elif(timestamp>endPrediction):
            continue

        weatherForecastHourlyData[i]["dt"] = timestamp.strftime("%d-%m-%Y, %H:%M:%S")

        hourlyTemp = weatherForecastHourlyData[i]["temp"]

        hourlyTimeData.append(timestamp)
        hourlyTempData.append(hourlyTemp)

    # Combine the lists using zip
    hourlyWeatherData = list(zip(hourlyTimeData, hourlyTempData))

    # Create a DataFrame from the combined list
    exogTemperatur = pd.DataFrame(hourlyWeatherData, columns=['Time', 'Temperatur'])
    exogTemperatur.asfreq('H')
    exogTemperatur.set_index('Time')

    print(exogTemperatur)


    timezone = pytz.timezone('Europe/Madrid')
    startPrediction = timezone.localize(startPrediction)
    endPrediction = timezone.localize(endPrediction)

    predictionFuture = model.predict(startPrediction, endPrediction, exog=exogTemperatur['Temperatur']).rename('Prediction')

    print(predictionFuture)

    predictionFuture.plot(legend=True)
    # ---------------------------------------------------------------------------------


    plt.show()