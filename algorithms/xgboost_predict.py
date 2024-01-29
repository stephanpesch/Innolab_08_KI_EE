from tokens import open_weather_map_token

import pandas as pd
import numpy as np
import requests
from matplotlib import pyplot as plt
from algorithms.xgboost_train import *
import re
from datetime import datetime

def xgboost_predict(model, location, checked_weather_columns, weather_col_names):
    useWeatherColumns = []
    i = 0

    # Select weather columns based on the user's selection
    for column in checked_weather_columns:
        if column in weather_col_names:  # Überprüfen, ob die Spalte in col_names vorhanden ist
            useWeatherColumns.append(column)

    useWeatherColumns[0] = "time"

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

    hourlyDateTime = []
    hourlyWeatherData = np.zeros((48, len(useWeatherColumns) - 1))
    for i in range(48):
        hourlyData = weatherForecastHourlyData[i]["dt"]

        timestamp = pd.to_datetime(hourlyData, utc=False, unit='s')
        weatherForecastHourlyData[i]["dt"] = timestamp.strftime("%d-%m-%Y, %H:%M:%S")

        for j in range(len(useWeatherColumns) - 1):
            if useWeatherColumns[j + 1] == "temp":
                hourlyWeatherData[i][j] = round(weatherForecastHourlyData[i][useWeatherColumns[j + 1]] - 273.15, 2)
            else:
                hourlyWeatherData[i][j] = round(weatherForecastHourlyData[i][useWeatherColumns[j + 1]], 2)

        hourlyDateTime.append(timestamp)

    # Create a DataFrame
    df_time = pd.DataFrame(hourlyDateTime, columns=['time'])
    useWeatherColumns.remove("time")
    df_weather = pd.DataFrame(hourlyWeatherData, columns=useWeatherColumns)

    df_future_dates = pd.concat([df_time, df_weather], axis=1)
    df_future_dates.asfreq('H')


    print(df_future_dates)

    # -----------------------------------------------------------------------------
    #future prediction
    df_future_dates['predicted_load'] = np.nan
    df_future_dates.index = pd.to_datetime(df_future_dates['time'], format='%Y-%m-%d %H:%M:%S')
    df_future_dates.drop(["time"], axis=1, inplace=True)
    df_future_dates_copy = df_future_dates.copy()
    X_test_future, y_test_future = create_features(df_future_dates, label='predicted_load')

    print(df_future_dates)
    model = load_model('trained/rnn.h5')
    y_future = model.predict(X_test_future)

    df_future_dates['predicted_load'] = y_future
    plt.figure(figsize=(20, 8))
    plt.plot(list(df_future_dates.index), list(df_future_dates['predicted_load']))
    plt.title("Predicted")
    plt.ylabel("load")
    plt.xlabel("time")
    plt.legend(('predicted'))
    plt.show()
