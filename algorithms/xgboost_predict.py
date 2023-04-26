from tokens import open_weather_map_token

import pandas as pd
import requests
from matplotlib import pyplot as plt
from algorithms.xgboost_train import *
from datetime import datetime

def xgboost_predict(model, location):
    # API CALL
    geolocation = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct?q=" + location + "&appid=" + open_weather_map_token)
    geolocation = geolocation.json()
    print(geolocation)
    geolocationData = geolocation[0]
    longitude = geolocationData['lon']
    latitude = geolocationData['lat']

    weatherForecast = requests.get(
        "https://api.openweathermap.org/data/3.0/onecall?lat=" + str(latitude) + "&lon=" + str(
            longitude) + "&exclude=minutely,daily&appid=" + open_weather_map_token)
    weatherForecast = weatherForecast.json()
    weatherForecastHourlyData = weatherForecast["hourly"]

    hourlyTimeData = []
    hourlyTempData = []
    for i in range(48):
        hourlyData = weatherForecastHourlyData[i]["dt"]

        timestamp = pd.to_datetime(hourlyData, utc=False, unit='s')
        weatherForecastHourlyData[i]["dt"] = timestamp.strftime("%d-%m-%Y, %H:%M:%S")
        hourlyTemp = round(weatherForecastHourlyData[i]["temp"] - 273.15, 2)

        hourlyTimeData.append(timestamp)
        hourlyTempData.append(hourlyTemp)

    # Combine the lists using zip
    hourlyWeatherData = list(zip(hourlyTimeData, hourlyTempData))

    # Create a DataFrame from the combined list
    df_future_dates = pd.DataFrame(hourlyWeatherData, columns=['time', 'temp'])
    df_future_dates.asfreq('H')
    df_future_dates.set_index('time')

    print(df_future_dates)

    # -----------------------------------------------------------------------------
    #future prediction
    #dti = pd.date_range(datetime.now(), periods=48, freq="H")
    #df_future_dates = pd.DataFrame([dti, exogTemperatur], columns=['time', 'temp'])
    df_future_dates['predicted_load'] = np.nan
    df_future_dates.index = pd.to_datetime(df_future_dates['time'], format='%Y-%m-%d %H:%M:%S')
    df_future_dates_copy = df_future_dates.copy()
    X_test_future, y_test_future = xgboost_train.create_features(df_future_dates, label='predicted_load')

    print(df_future_dates)

    y_future = model.predict(X_test_future)

    df_future_dates['predicted_load'] = y_future
    plt.figure(figsize=(20, 8))
    plt.plot(list(df_future_dates['time']), list(df_future_dates['predicted_load']))
    plt.title("Predicted")
    plt.ylabel("load")
    plt.xlabel("time")
    plt.legend(('predicted'))
    plt.show()
