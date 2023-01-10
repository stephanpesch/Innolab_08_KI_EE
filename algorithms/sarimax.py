def sarimax_algorithm(df, checked_columns):
    from pmdarima.arima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline

    col_list = ["time", "total load actual"]
    energy_df = pd.read_csv("csv_files/sarimax/energy_dataset.csv", usecols=col_list, index_col='time', parse_dates=True)

    energy_df= energy_df.asfreq('H')

    energy_df= energy_df.fillna(method='ffill')
    energy_df.dropna(
        axis=0,
        how='any',
        subset=None,
        inplace=True
    )

    ##energy_df['total load actual'].plot(figsize=(80,80))

    from statsmodels.tsa.seasonal import seasonal_decompose
    decompose_data = seasonal_decompose(energy_df, model="additive", period=24)
    decompose_data.plot();

    ##auto_arima(energy_df['total load actual'], m=7, trace=True).summary()

    # ---------------------------------------------------------------------------------

    train_df= energy_df.iloc[:34900]
    test_df= energy_df.iloc[34900:]

    test_auto_arima=test_df[0:744]

    auto_arima(test_auto_arima['total load actual'], m=24, trace=True).summary()

    # ---------------------------------------------------------------------------------

    ##SARIMAX (0, 1, 2)X(1, 0, 1, 24)

    ##keine Ergebnisse in 20 min
    modelFirst= SARIMAX(energy_df['total load actual'], order=(0,1,2), seasonal_order=(1,0,1,24), enforce_stationarity=False)

    resFirst= modelFirst.fit()

    startFirst= len(train_df)
    endFirst= len(train_df) + len(test_df)-1

    predictionFirst = resFirst.predict(startFirst, endFirst).rename('Prediction')
    ax = test_df['total load actual'].plot(legend=True, figsize=(16,8))
    predictionFirst.plot(legend=True)

    # ---------------------------------------------------------------------------------

    from statsmodels.tools.eval_measures import rmse
    rmse(test_df['total load actual'], predictionFirst)

    # ---------------------------------------------------------------------------------

    ##SARIMAX (1, 1, 1)X(2, 0, 0, 24)
    mod= SARIMAX(energy_df['total load actual'], order=(1,1,1), seasonal_order=(2,0,0,24))

    res= mod.fit()

    start= len(train_df)
    end= len(train_df) + len(test_df)-1

    prediction = res.predict(start, end).rename('Prediction')
    ax = test_df['total load actual'].plot(legend=True, figsize=(16,8))
    prediction.plot(legend=True)
    
    # ---------------------------------------------------------------------------------

    col_list_weather = ["dt_iso", "temp", "pressure", "humidity", "wind_speed", "weather_description"]
    weather_df = pd.read_csv("csv_files/sarimax/weather_features.csv", usecols=col_list_weather, index_col='dt_iso', parse_dates=True, nrows=35145)
    energy_weather_df=pd.concat([energy_df, weather_df], axis=1)

    # ---------------------------------------------------------------------------------

    from statsmodels.tools.eval_measures import rmse
    rmse(test_df['total load actual'], prediction)

    # ---------------------------------------------------------------------------------

    df_auto = energy_weather_df[0:744]
    auto_arima(df_auto['total load actual'], exogenous=df_auto['temp'], m=24, trace=True).summary()

    # ---------------------------------------------------------------------------------

    train2_df= energy_weather_df.iloc[:34950]
    test2_df= energy_weather_df.iloc[34950:]

    # ---------------------------------------------------------------------------------

    ##SARIMAX (0, 1, 2)X(1, 0, 1, 24)

    model= SARIMAX(energy_weather_df['total load actual'], exog=energy_weather_df['temp'],  order=(0,1,2), seasonal_order=(1,0,1,24))
    result=model.fit()

    start2= len(train2_df)
    end2= len(train2_df) + len(test2_df)-1

    prediction2 = result.predict(start2, end2, exog=test2_df['temp']).rename('Prediction')
    ax = test2_df['total load actual'].plot(legend=True, figsize=(16,8))
    prediction2.plot(legend=True)

    # ---------------------------------------------------------------------------------

    from statsmodels.tools.eval_measures import rmse
    rmse(test2_df['total load actual'], prediction2)