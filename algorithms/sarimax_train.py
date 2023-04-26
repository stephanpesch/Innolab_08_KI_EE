from datetime import timedelta, datetime

import pytz
from dateutil import tz


def sarimax_train(file, weather_file, checked_columns, checked_weather_columns, col_names, weather_col_names):
    from pmdarima.arima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tools.eval_measures import rmse
    # %matplotlib inline

    useColumns=[]
    i=0

    for column in checked_columns:
        if(checked_columns[i].get()==1):
           useColumns.append(col_names[i])
        i=i+1





    energy_df = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    print(energy_df)
    
    energy_df.sort_values(by=useColumns[0], ascending = True)
    energy_df= energy_df.fillna(method='ffill')
    energy_df=energy_df.sort_values(by= useColumns[0], ascending = True)
    energy_df.dropna(axis=0,how='any',subset=None,inplace=True)

    energy_df = energy_df[~energy_df.index.duplicated(keep='first')]

    energy_df= energy_df.asfreq('H')
    energy_df= energy_df.fillna(method='ffill')



    # ---------------------------------------------------------------------------------

    useWeatherColumns = []
    i = 0

    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1


    weather_df = pd.read_csv(weather_file, usecols=useWeatherColumns, index_col=useWeatherColumns[0], parse_dates=True)

    weather_df = weather_df.asfreq('H')
    print(weather_df)
    energy_weather_df=pd.concat([energy_df, weather_df], axis=1)

    # ---------------------------------------------------------------------------------
    end_date = datetime.now() - timedelta(hours=48)
    if end_date.minute >= 30:
        end_date = end_date.replace(second=0, microsecond=0, minute=0, hour=(end_date.hour + 1) % 24)
    else:
        end_date = end_date.replace(second=0, microsecond=0, minute=0)

    start_date = end_date - timedelta(hours=532)

    timezone = pytz.timezone('Europe/Madrid')
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    energy_weather_df.index = energy_weather_df.index.tz_convert('Europe/Madrid')

    # check if the timezones are the same
    if energy_weather_df.index.tz == start_date.tzinfo and energy_weather_df.index.tz == end_date.tzinfo:
        print('The timezone of the index is the same as the timezones of start_date and end_date.')
    else:
        print("Timezone Index: " + str(energy_weather_df.index.tz))
        print("Timezone Start date: " + str(start_date.tzinfo))
        print('The timezone of the index is different from the timezones of start_date and end_date.')

    train_df = energy_weather_df.loc[start_date:end_date - timedelta(hours=1)]
    #train_df = energy_weather_df.iloc[len(energy_weather_df) - 532:len(energy_weather_df) - 48]
    test_df = energy_weather_df.loc[end_date:end_date + timedelta(hours=40)]
    #test_df = energy_weather_df.iloc[len(energy_weather_df) - 48:]

    exog_train = train_df['temp']
    exog_forecast = test_df['temp']

    # ---------------------------------------------------------------------------------

    ##SARIMAX (2, 0, 2)X(0, 1, 2, 24)

    modGRID = SARIMAX(train_df[useColumns[1]], order=(2, 0, 2), exog=exog_train,
                      seasonal_order=(0, 1, 2, 24))
    print(start_date)
    print(end_date)
    resGRID = modGRID.fit(maxiter=200)

    print(start_date)
    print(end_date)
    startGRID = end_date
    endGRID = end_date + timedelta(hours=40)


    print(exog_forecast.info())
    print(test_df.info())

    predictionGRID = resGRID.predict(startGRID, endGRID, exog=exog_forecast).rename('Prediction')
    ax = test_df[useColumns[1]].plot(legend=True, figsize=(16, 8))
    predictionGRID.plot(legend=True)

    plt.show(block=False)
    print("rmse: " + str(rmse(test_df['total load actual'], predictionGRID)))

    return modGRID

    
    # ---------------------------------------------------------------------------------

