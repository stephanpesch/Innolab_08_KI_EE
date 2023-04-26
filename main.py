import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests
from datetime import datetime
from tokens import *
from algorithms.rnn import *
from algorithms.lstm import *
from algorithms.sarimax_train import *
from algorithms.sarimax_predict import *
from algorithms.xgboost_predict import *
from algorithms.xgboost_train import *

matplotlib.use('TkAgg')

# activate interactive mode, so algorithms shouldn't get interrupted when plt.show()
# plt.ion()
root = tk.Tk()
root.geometry("800x500")
label = tk.Label(root, text="Energy Consumption Forecast",
                 fg="light green",
                 bg="dark green",
                 font="Helvetica 16 bold italic")
label.grid(row=1, column=2)
subtitle = tk.Label(root,
                    text="To train a model with your data, please select an algorithm and upload your data as CSV-file",
                    font="Helvetica 10 bold italic")
subtitle.grid(row=2, column=2)

global sarimaxModel


def run_algorithm():
    global model
    # print(checked_columns[0].get())  the checked boxes can be accessed like this, 1 = > checked / 0 => not checked
    if (var1.get() == "LSTM"):
        print("I will now run the " + var1.get() + " algorithm")
        lstm_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "XGBOOST"):
        print("I will now run the " + var1.get() + " algorithm")
        model = xgboost_train(file, weather_file, checked_columns, checked_weather_columns,
                      col_names, weather_col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "SARIMAX"):
        print("I will now run the " + var1.get() + " algorithm")
        model = sarimax_train(file, weather_file, checked_columns, checked_weather_columns,
                              col_names, weather_col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "RNN"):
        print("I will now run the " + var1.get() + " algorithm")
        rnn_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")


# function to open a new window
# on a button click
def open_new_window():
    # Toplevel object which will
    # be treated as a new window
    new_window = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    new_window.title("Data selection")

    # sets the geometry of toplevel
    new_window.geometry("800x500")
    column_selection_text = tk.Label(new_window,
                                     text="Select Columns",
                                     font="Helvetica 10 bold italic")
    column_selection_text.grid(row=2, column=0)
    global checked_columns
    global checked_weather_columns
    checked_columns = []
    checked_weather_columns = []
    row_counter = 2
    weather_row_counter = 2
    for col_name in col_names:
        checked_columns.append(tk.IntVar())
        row_counter += 1
        tk.Checkbutton(new_window, text=col_name, variable=checked_columns[-1],
                       onvalue=1, offvalue=0, height=1,
                       width=20, anchor=tk.W).grid(row=row_counter, column=1)

    for weather_col_name in weather_col_names:
        checked_weather_columns.append(tk.IntVar())
        weather_row_counter += 1
        tk.Checkbutton(new_window, text=weather_col_name, variable=checked_weather_columns[-1],
                       onvalue=1, offvalue=0, height=1,
                       width=20, anchor=tk.W).grid(row=weather_row_counter, column=4)

    tk.Button(new_window, text='Train Model', command=run_algorithm).grid(row=8, column=2)


def open_forecast_window():
    location = location_field.get()
    # Toplevel object which will
    # be treated as a new window
    forecast_window = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    forecast_window.title("Forecast Area")

    # sets the geometry of toplevel
    forecast_window.geometry("800x500")

    if (var1.get() == "LSTM"):
        print("I will now run the " + var1.get() + " algorithm")
        lstm_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "XGBOOST"):
        print("Prediction will be done using the XGBOOST model")
        xgboost_predict(model, location)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "SARIMAX"):
        print("Prediction will be done using the SARIMAX model")
        sarimax_predict(model, location)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "RNN"):
        print("I will now run the " + var1.get() + " algorithm")
        rnn_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")

    tk.Button(forecast_window, text='Predict values').grid(row=8, column=2)


def open_weather_window():
    location = location_field.get()
    weather_window = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    weather_window.title("Weather Forecast")

    # sets the geometry of toplevel
    weather_window.geometry("800x500")

    label_weather = tk.Label(weather_window, text="Weather Forecast",
                             fg="light green",
                             bg="dark green",
                             font="Helvetica 16 bold italic").grid(row=1, column=2)

    column_name = tk.Label(weather_window,
                           text="Time",
                           font="Helvetica 8 bold italic")
    column_name.grid(row=2, column=1)

    column_name2 = tk.Label(weather_window,
                            text="Degree[°C]",
                            font="Helvetica 8 bold italic")
    column_name2.grid(row=2, column=2)

    column_name3 = tk.Label(weather_window,
                            text="Humidity",
                            font="Helvetica 8 bold italic")
    column_name3.grid(row=2, column=3)

    column_name4 = tk.Label(weather_window,
                            text="Wind Speed",
                            font="Helvetica 8 bold italic")
    column_name4.grid(row=2, column=4)

    # API CALL
    city = location
    geolocation = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct?q=" + city + "&appid=" + open_weather_map_token)
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

    for i in range(48):
        hourlyData = weatherForecastHourlyData[i]["dt"]

        timestamp = pd.to_datetime(hourlyData, utc=True, unit='s')
        weatherForecastHourlyData[i]["dt"] = timestamp.strftime("%d-%m-%Y, %H:%M:%S")
        time = tk.Label(weather_window, text=weatherForecastHourlyData[i]["dt"],
                        font="Helvetica 8").grid(row=i + 3, column=1)
        degree = tk.Label(weather_window, text=round(weatherForecastHourlyData[i]["temp"] - 273.15, 2),
                          font="Helvetica 8").grid(row=i + 3, column=2)
        hum = tk.Label(weather_window, text=round(weatherForecastHourlyData[i]["humidity"], 2),
                       font="Helvetica 8").grid(row=i + 3, column=3)
        wind = tk.Label(weather_window, text=round(weatherForecastHourlyData[i]["wind_speed"], 2),
                        font="Helvetica 8").grid(row=i + 3, column=4)

    # weatherForecastHourlyData is a list with all the necessary data for the next 48 hours


tk.Button(root, text='Train Model', command=open_new_window).grid(row=8, column=2)

tk.Button(root, text='Forecast Area', command=open_forecast_window).grid(row=9, column=2)

locationLabel = tk.Label(text='Enter location')
locationLabel.grid(row=5, column=2)
location_field = tk.Entry(root)
location_field.grid(row=6, column=2)

tk.Button(root, text='Weather Forecast', command=lambda: open_weather_window()).grid(row=10, column=2)


####CSV File Upload

def upload_consumption_file():
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    global file
    file = filedialog.askopenfilename(filetypes=f_types)
    # l1.config(text=file) # display the path
    df = pd.read_csv(file)  # create DataFrame
    global col_names
    col_names = list(df.columns)
    print(col_names)
    label1['text'] = file.split('/')[len(file.split('/')) - 1]  # display filename


def upload_weather_file():
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    global weather_file
    weather_file = filedialog.askopenfilename(filetypes=f_types)
    # l1.config(text=file) # display the path
    df = pd.read_csv(weather_file)  # create DataFrame
    global weather_col_names
    weather_col_names = list(df.columns)
    print(weather_col_names)
    label2['text'] = weather_file.split('/')[len(weather_file.split('/')) - 1]  # display filename


b1 = tk.Button(root, text='Upload CSV-File',
               width=20, command=lambda: upload_consumption_file())
b1.grid(row=5, column=3)
label1 = tk.Label(text='Please choose a file')
label1.grid(row=6, column=3)

b2 = tk.Button(root, text='Upload Weather',
               width=20, command=lambda: upload_weather_file())
b2.grid(row=8, column=3)
label2 = tk.Label(text='Please choose a weather file, if not already included in consumption data')
label2.grid(row=9, column=3)

# so root window stays interactive (maybe later)
# tk.Button(root, text='Run the algorithm', command=threading.Thread(target=run_algorithm).start).grid(row=2, column=1)

var1 = tk.StringVar(root, "LSTM")  # Create a variable for strings, and initialize the variable


def printResults():
    print(var1.get())


tk.Radiobutton(root, text="LSTM", variable=var1, value="LSTM", command=printResults).grid(row=4, column=0)
tk.Radiobutton(root, text="XGBOOST", variable=var1, value="XGBOOST", command=printResults).grid(row=5, column=0)
tk.Radiobutton(root, text="SARIMAX", variable=var1, value="SARIMAX", command=printResults).grid(row=6, column=0)
tk.Radiobutton(root, text="RNN", variable=var1, value="RNN", command=printResults).grid(row=7, column=0)

root.mainloop()
