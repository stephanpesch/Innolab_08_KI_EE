from tokens import open_weather_map_token

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import r2_score
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import requests

def rnn_predict(model, location, file, weather_file, checked_columns, checked_weather_columns,
                  col_names, weather_col_names):
    useColumns = []
    i = 0

    # Select columns based on the user's selection
    for column in checked_columns:
        if (checked_columns[i].get() == 1):
            useColumns.append(col_names[i])
        i = i + 1

    # Read energy data from file
    df_energy = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    # Sort and preprocess energy data
    df_energy = df_energy.fillna(method='ffill')
    df_energy = df_energy.sort_values(by=useColumns[0], ascending=True)
    df_energy.dropna(axis=0, how='any', subset=None, inplace=True)
    df_energy = df_energy.asfreq('H')
    print(df_energy)

    useWeatherColumns = []
    i = 0

    # Select weather columns based on the user's selection
    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1

    # Read weather data from file
    useWeatherColumns = ["dt_iso", "temp"]
    df_weather = pd.read_csv(weather_file, usecols=useWeatherColumns, index_col=useWeatherColumns[0], parse_dates=True)
    df_weather = df_weather.fillna(method='ffill')
    df_weather = df_weather.asfreq('H')
    print(df_weather)

    # Concatenate energy and weather data
    df = pd.concat([df_weather, df_energy], axis=1)
    seq_len = 20
    useWeatherColumns = []
    i = 0

    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1
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
                hourlyWeatherData[i][j] = round(weatherForecastHourlyData[i][useWeatherColumns[j + 1]], 2)
            else:
                hourlyWeatherData[i][j] = round(weatherForecastHourlyData[i][useWeatherColumns[j + 1]], 2)

        hourlyDateTime.append(timestamp)

    # Create a DataFrame
    df_time = pd.DataFrame(hourlyDateTime, columns=['time'])
    useWeatherColumns.remove("time")
    df_weather = pd.DataFrame(hourlyWeatherData, columns=useWeatherColumns)

    df_future_dates = pd.concat([df_time, df_weather], axis=1)
    df_future_dates.asfreq('H')
    df_future_dates.set_index("time", inplace=True)

    df_future_dates

    column_names = df_future_dates.columns.tolist()

    def normalize_future_data(new_df, initial_df, scaler=None):
        if scaler is None:
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(initial_df[column_names].values.reshape(-1, len(column_names)))
        
        normalized_data = new_df.copy()
        normalized_data[column_names] = scaler.transform(normalized_data[column_names].values.reshape(-1, len(column_names)))
        
        return normalized_data
    df_future_norm = normalize_future_data(df_future_dates, df)

    X_test_future = []
    for i in range(seq_len, len(df_future_norm)):
        X_test_future.append(df_future_norm.iloc[i - seq_len: i, 0])
        
    # Convert to numpy array
    X_test_future = np.array(X_test_future)

    # Reshape data for RNN input
    X_test_future = np.reshape(X_test_future, (X_test_future.shape[0], seq_len, 1))
    
    rnn_future_predictions = model.predict(X_test_future)
    
    time_index = df_future_norm.index[seq_len:]
    plt.figure(figsize=(16, 7))
    time_index = time_index[-len(rnn_future_predictions):]
    plt.plot(time_index, rnn_future_predictions, alpha=0.7, color='orange', label='Predicted power consumption data')
    plt.title("Predicted power consumption for the next 48 hours")
    plt.xlabel('Time in hours')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show(block=False)