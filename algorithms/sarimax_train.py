from datetime import timedelta, datetime
from tkinter import ttk

import pytz
from dateutil import tz
import tkinter as tk


def sarimax_train(file, weather_file, checked_columns, checked_weather_columns, col_names, weather_col_names, rootWindow):
    from pmdarima.arima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tools.eval_measures import rmse
    # %matplotlib inline

    useColumns = []
    i = 0

    for column in checked_columns:
        if (checked_columns[i].get() == 1):
            useColumns.append(col_names[i])
        i = i + 1

    energy_df = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    energy_df.sort_values(by=useColumns[0], ascending=True)
    energy_df = energy_df.fillna(method='ffill')
    energy_df = energy_df.sort_values(by=useColumns[0], ascending=True)
    energy_df.dropna(axis=0, how='any', subset=None, inplace=True)

    energy_df = energy_df[~energy_df.index.duplicated(keep='first')]

    energy_df = energy_df.asfreq('H')
    energy_df = energy_df.fillna(method='ffill')

    # ---------------------------------------------------------------------------------

    useWeatherColumns = []
    i = 0

    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1

    weather_df = pd.read_csv(weather_file, usecols=useWeatherColumns, index_col=useWeatherColumns[0], parse_dates=True)

    weather_df = weather_df.asfreq('H')
    energy_weather_df = pd.concat([energy_df, weather_df], axis=1)

    # ---------------------------------------------------------------------------------
    end_date = datetime.now() - timedelta(hours=39)
    if end_date.minute >= 30:
        end_date = end_date.replace(second=0, microsecond=0, minute=0, hour=(end_date.hour + 1) % 24)

    else:
        end_date = end_date.replace(second=0, microsecond=0, minute=0)

    start_date = end_date - timedelta(hours=532)

    timezone = pytz.timezone('Europe/Madrid')
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    energy_weather_df.index = energy_weather_df.index.tz_convert('Europe/Madrid')

    train_df = energy_weather_df.loc[start_date:end_date - timedelta(hours=1)]
    # train_df = energy_weather_df.iloc[len(energy_weather_df) - 532:len(energy_weather_df) - 48]

    test_df = energy_weather_df.loc[end_date:end_date + timedelta(hours=39)]
    # test_df = energy_weather_df.iloc[len(energy_weather_df) - 48:]

    exog_train = train_df['temp']
    exog_forecast = test_df['temp']

    # ---------------------------------------------------------------------------------

    ##SARIMAX (2, 0, 2)X(0, 1, 2, 24)

    modGRID = SARIMAX(train_df[useColumns[1]], order=(2, 0, 2), exog=exog_train,
                      seasonal_order=(0, 1, 2, 24))
    print(start_date)
    print(end_date)
    resGRID = modGRID.fit(maxiter=200)

    startGRID = end_date
    endGRID = end_date + timedelta(hours=39)
    print(startGRID)
    print(endGRID)

    predictionGRID = resGRID.predict(startGRID, endGRID, exog=exog_forecast).rename('Prediction')
    ax = test_df[useColumns[1]].plot(legend=True, figsize=(16, 8))
    predictionGRID.plot(legend=True)

    print(predictionGRID)
    predictionDF = pd.DataFrame(predictionGRID)

    rmseValue = str(rmse(test_df['total load actual'], predictionGRID))

    print("rmse: " + rmseValue)

    trainResultWindow = tk.Toplevel(rootWindow)

    # sets the title of the
    # Toplevel widget
    trainResultWindow.title("Training results")

    # sets the geometry of toplevel
    trainResultWindow.geometry("800x500")

    # create a Listbox widget
    my_listbox = tk.Listbox(trainResultWindow)
    # add each element of the Series to the Listbox
    # add each row of the DataFrame to the Listbox
    for index, row in predictionDF.iterrows():
        my_listbox.insert(tk.END, f"{row}")
    # pack the Listbox widget onto the window
    labelResults = tk.Label(trainResultWindow, text="Results")
    labelResults.grid(row=1, column=2)
    my_listbox.grid(row=2, column=2, sticky="nswe")

    trainResultWindow.rowconfigure(2, weight=1)
    trainResultWindow.columnconfigure(2, weight=1)

    my_var = tk.StringVar()
    my_var.set(rmseValue)

    # create a label to display the variable
    my_label = tk.Label(trainResultWindow, textvariable=my_var)
    labelRMSE = tk.Label(trainResultWindow, text="Root-Mean-Squared-Error")
    my_label.grid(row=2, column=4)
    labelRMSE.grid(row=1, column=4)

    # ---------------------------------------------------------------------------------

    end_date = datetime.now() - timedelta(hours=1)
    if end_date.minute >= 30:
        end_date = end_date.replace(second=0, microsecond=0, minute=0, hour=(end_date.hour + 1) % 24)
    else:
        end_date = end_date.replace(second=0, microsecond=0, minute=0)

    start_date = end_date - timedelta(hours=550)

    timezone = pytz.timezone('Europe/Madrid')
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    train_df = energy_weather_df.loc[start_date:end_date]
    exog_train = train_df['temp']

    newModel = SARIMAX(train_df[useColumns[1]], order=(2, 0, 2), exog=exog_train,
                       seasonal_order=(0, 1, 2, 24))

    returnModel = newModel.fit(maxiter=200)

    plt.show(block=False)
    print("Model is ready")

    return returnModel
