from tkinter import filedialog

import matplotlib
import json
from keras.models import load_model
import pickle

from algorithms.lstm import *
from algorithms.rnn_train import *
from algorithms.rnn_predict import *
from algorithms.sarimax_predict import *
from algorithms.sarimax_train import *
from algorithms.xgboost_predict import *
from algorithms.xgboost_train import *

matplotlib.use('TkAgg')


# activate interactive mode, so algorithms shouldn't get interrupted when plt.show()
# plt.ion()

def run_algorithm():
    global model
    algorithm_functions = {
        "LSTM": lstm_algorithm,
        "XGBOOST": xgboost_train,
        "SARIMAX": sarimax_train,
        "RNN": rnn_train
    }

    print("I will now run the " + var1.get() + " algorithm")
    model = algorithm_functions[var1.get()](file, weather_file, checked_columns, checked_weather_columns,
                                            col_names, weather_col_names, root, grid_var)
    print(var1.get() + " algorithm completed")


# function to open a new window
# on a button click
def open_new_window():
    new_window = tk.Toplevel(root)
    new_window.geometry("1200x800")
    new_window.minsize(1200, 800)

    n_rows = 60
    n_columns = 10
    for i in range(n_rows):
        new_window.grid_rowconfigure(i, weight=1)
    for i in range(n_columns):
        new_window.grid_columnconfigure(i, weight=1)

    new_window.title("Data selection")

    column_selection_text = ttk.Label(new_window, text="Select Columns", font=("Arial", 15))
    column_selection_text.grid(row=1, column=3)

    target_header = ttk.Label(new_window, text="Target Value", font=("Arial", 12, "bold"))
    target_header.grid(row=4, column=2)

    weather_header = ttk.Label(new_window, text="Weather Features", font=("Arial", 12, "bold"))
    weather_header.grid(row=4, column=8)

    separator = ttk.Separator(new_window, orient="vertical")
    separator.grid(row=5, column=6, rowspan=25, sticky="ns")

    global checked_columns
    global checked_weather_columns
    checked_columns = []
    checked_weather_columns = []
    row_counter = 5
    weather_row_counter = 5
    col_add = 0
    for col_name in col_names:
        checked_columns.append(tk.IntVar())
        row_counter += 1
        if row_counter > 20:
            row_counter = 6
            col_add = 3
        check_button = ttk.Checkbutton(new_window, text=col_name, variable=checked_columns[-1], onvalue=1, offvalue=0)
        check_button.grid(row=row_counter, column=0 + col_add)
        
        # set index to true by default
        if col_name == "time":
            checked_columns[-1].set(1)  
        else:
            checked_columns[-1].set(0)  

    for weather_col_name in weather_col_names:
        checked_weather_columns.append(tk.IntVar())
        weather_row_counter += 1
        check_button = ttk.Checkbutton(new_window, text=weather_col_name, variable=checked_weather_columns[-1], onvalue=1, offvalue=0)
        check_button.grid(row=weather_row_counter, column=8)
        
        # set index to true by default
        if weather_col_name == "dt_iso":
            checked_weather_columns[-1].set(1)  
        else:
            checked_weather_columns[-1].set(0)  

    global grid_var
    grid_var = tk.IntVar()
    ttk.Checkbutton(new_window, text="Perform Grid Search", variable=grid_var, onvalue=1, offvalue=0).grid(row=48,
                                                                                                           column=3)
    ttk.Button(new_window, text='Train Model', command=run_algorithm).grid(row=50, column=3)


def open_forecast_window():
    location = location_field.get()
    # Toplevel object which will
    # be treated as a new window
    forecast_window = tk.Toplevel(root)

    # sets the geometry of toplevel
    forecast_window.geometry("1200x600")
    forecast_window.minsize(1200, 600)

    n_rows = 60
    n_columns = 10
    for i in range(n_rows):
        forecast_window.grid_rowconfigure(i, weight=1)
    for i in range(n_columns):
        forecast_window.grid_columnconfigure(i, weight=1)

    # sets the title of the
    # Toplevel widget
    forecast_window.title("Forecast Area")

    if var1.get() == "LSTM":
        print("I will now run the " + var1.get() + " algorithm")
        lstm_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif var1.get() == "XGBOOST":
        print("Prediction will be done using the XGBOOST model")
        model_filename = 'trained/xgboost_model.pkl'
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)

        file_path_energy = 'trained/spaltennamen_xgboost_energy.json'
        with open(file_path_energy, 'r') as file_to_read:
            checked_columns = json.load(file_to_read)

        file_path_weather = 'trained/spaltennamen_xgboost_weather.json'
        with open(file_path_weather, 'r') as file_to_read:
            checked_weather_columns = json.load(file_to_read)

        xgboost_predict(model, location, checked_weather_columns, weather_col_names)
        print(var1.get() + " algorithm completed")
    elif var1.get() == "SARIMAX":
        print("Prediction will be done using the SARIMAX model")
        model_filename = 'trained/sarimax_model.pkl'
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
            
        sarimax_predict(model, location)
        print(var1.get() + " algorithm completed")
    elif var1.get() == "RNN":
        print("I will now run the " + var1.get() + " algorithm")
        model = load_model('trained/rnn.h5')

        file_path_energy = 'trained/spaltennamen_rnn_energy.json'
        with open(file_path_energy, 'r') as file_to_read:
            checked_columns = json.load(file_to_read)

        file_path_weather = 'trained/spaltennamen_rnn_weather.json'
        with open(file_path_weather, 'r') as file_to_read:
            checked_weather_columns = json.load(file_to_read)
        

        rnn_predict(model, location, file, weather_file, checked_columns, checked_weather_columns,
                  col_names, weather_col_names)
        print(var1.get() + " algorithm completed")

    tk.Button(forecast_window, text='Predict values').grid(row=35, column=5)


def open_weather_window():
    location = location_field.get()
    weather_window = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    weather_window.title("Weather Forecast")

    # sets the geometry of toplevel
    weather_window.geometry("1200x600")
    weather_window.minsize(1200, 600)

    n_rows = 40
    n_columns = 10
    for i in range(n_rows):
        weather_window.grid_rowconfigure(i, weight=1)
    for i in range(n_columns):
        weather_window.grid_columnconfigure(i, weight=1)

    ttk.Label(weather_window, text="Weather Forecast",
              font=("Arial", 15)).grid(row=1, column=1)

    column_name = ttk.Label(weather_window,
                            text="Time",
                            font=("Arial", 8))
    column_name.grid(row=3, column=1)

    column_name2 = ttk.Label(weather_window,
                             text="Degree[°C]",
                             font=("Arial", 8))
    column_name2.grid(row=3, column=2)

    column_name3 = ttk.Label(weather_window,
                             text="Humidity",
                             font=("Arial", 8))
    column_name3.grid(row=3, column=3)

    column_name4 = ttk.Label(weather_window,
                             text="Wind Speed",
                             font=("Arial", 8))
    column_name4.grid(row=3, column=4)

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
        time = ttk.Label(weather_window, text=weatherForecastHourlyData[i]["dt"],
                         font=("Arial", 5)).grid(row=i + 4, column=1)
        degree = ttk.Label(weather_window, text=round(weatherForecastHourlyData[i]["temp"] - 273.15, 2),
                           font=("Arial", 5)).grid(row=i + 4, column=2)
        hum = ttk.Label(weather_window, text=round(weatherForecastHourlyData[i]["humidity"], 2),
                        font=("Arial", 5)).grid(row=i + 4, column=3)
        wind = ttk.Label(weather_window, text=round(weatherForecastHourlyData[i]["wind_speed"], 2),
                         font=("Arial", 5)).grid(row=i + 4, column=4)

    # weatherForecastHourlyData is a list with all the necessary data for the next 48 hours


####CSV File Upload

def upload_consumption_file():
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    global file
    file = filedialog.askopenfilename(filetypes=f_types)
    # l1.config(text=file) # display the path
    df = pd.read_csv(file)  # create Dataroot
    global col_names
    col_names = list(df.columns)
    print(col_names)
    label1['text'] = file.split('/')[len(file.split('/')) - 1]  # display filename


def upload_weather_file():
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    global weather_file
    weather_file = filedialog.askopenfilename(filetypes=f_types)
    # l1.config(text=file) # display the path
    df = pd.read_csv(weather_file)  # create Dataroot
    global weather_col_names
    weather_col_names = list(df.columns)
    print(weather_col_names)
    label2['text'] = weather_file.split('/')[len(weather_file.split('/')) - 1]  # display filename


def print_results():
    print(var1.get())


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x600")
    root.minsize(1200, 600)
    root.tk.call("source", "Azure/azure.tcl")
    root.tk.call("set_theme", "dark")

    n_rows = 40
    n_columns = 10
    for i in range(n_rows):
        root.grid_rowconfigure(i, weight=1)
    for i in range(n_columns):
        root.grid_columnconfigure(i, weight=1)

    label = ttk.Label(root, text="Energy Consumption Forecast", font=("Arial", 18))
    label.grid(row=1, column=5)
    subtitle = ttk.Label(root,
                         text="To train a model with your data, please select an algorithm and upload your data as CSV-file",
                         font=("Arial", 13))
    subtitle.grid(row=2, column=5)

    ttk.Button(root, text='Train Model', command=open_new_window).grid(row=15, column=5)
    ttk.Button(root, text='Forecast Area', command=open_forecast_window).grid(row=16, column=5)
    locationLabel = tk.Label(text='Enter location')
    locationLabel.grid(row=18, column=5)
    location_field = tk.Entry(root)
    location_field.grid(row=19, column=5)
    ttk.Button(root, text='Weather Forecast', command=lambda: open_weather_window()).grid(row=20, column=5)

    b1 = ttk.Button(root, text='Upload CSV-File', width=20, command=lambda: upload_consumption_file())
    b1.grid(row=15, column=7)
    label1 = ttk.Label(text='Consumption Data')
    label1.grid(row=16, column=7)
    b2 = ttk.Button(root, text='Upload Weather', width=20, command=lambda: upload_weather_file())
    b2.grid(row=18, column=7)
    label2 = ttk.Label(text='Weather Data')
    label2.grid(row=19, column=7)

    var1 = tk.StringVar(root, "XGBOOST")  # Create a variable for strings, and initialize the variable

    ttk.Radiobutton(root, text="XGBOOST", variable=var1, value="XGBOOST", command=print_results).grid(row=15, column=2)
    ttk.Radiobutton(root, text="SARIMAX", variable=var1, value="SARIMAX", command=print_results).grid(row=16, column=2)
    ttk.Radiobutton(root, text="LSTM", variable=var1, value="LSTM", command=print_results).grid(row=17, column=2)
    ttk.Radiobutton(root, text="RNN", variable=var1, value="RNN", command=print_results).grid(row=18, column=2)

    ttk.Label(root, text="Selected Algorithm:").grid(row=20, column=2)
    ttk.Label(root, textvariable=var1).grid(row=20, column=3)

    ttk.Button(root, text='Run Algorithm', command=run_algorithm).grid(row=22, column=5)

    root.mainloop()
