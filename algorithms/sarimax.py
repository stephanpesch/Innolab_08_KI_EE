def sarimax_algorithm(file, weather_file, checked_columns, checked_weather_columns, col_names, weather_col_names):
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

    train_df = energy_weather_df.iloc[len(energy_weather_df) - 532:len(energy_weather_df) - 48]
    test_df = energy_weather_df.iloc[len(energy_weather_df) - 48:]

    exog_train = train_df['temp']
    exog_forecast = test_df['temp']

    # ---------------------------------------------------------------------------------

    ##SARIMAX (2, 0, 2)X(0, 1, 2, 24)

    modGRID = SARIMAX(train_df[useColumns[1]], order=(2, 0, 2), exog=exog_train,
                      seasonal_order=(0, 1, 2, 24))

    resGRID = modGRID.fit(maxiter=100)

    startGRID = len(train_df)
    endGRID = len(train_df) + len(test_df) - 1

    print("predict")
    predictionGRID = resGRID.predict(startGRID, endGRID, exog=exog_forecast).rename('Prediction')
    print("plot")
    ax = test_df[useColumns[1]].plot(legend=True, figsize=(16, 8))
    predictionGRID.plot(legend=True)

    plt.show()

    print("Fertig")
    # ---------------------------------------------------------------------------------

    print("Root mean squared error: "+rmse(test_df['total load actual'], predictionGRID))

    
    # ---------------------------------------------------------------------------------

    startPrediction= len(energy_weather_df)
    endPrediction= len(energy_weather_df) + 100

    predictionFuture = resGRID.predict(startPrediction, endPrediction).rename('Prediction')
    ax = test_df[useColumns[1]].plot(legend=True, figsize=(16,8))
    predictionFuture.plot(legend=True)
    # ---------------------------------------------------------------------------------

    print(predictionFuture)

    plt.show()