def sarimax_algorithm(file, checked_columns, col_names):
    from pmdarima.arima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline

    useColumns=[]
    i=0

    for column in checked_columns:
        if(checked_columns[i].get()==1):
           useColumns.append(col_names[i])
        i=i+1





    energy_df = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    print(energy_df)
    
    energy_df= energy_df.fillna(method='ffill')
    energy_df.dropna(
        axis=0,
        how='any',
        subset=None,
        inplace=True
    )
    energy_df= energy_df.asfreq('H')

    


    # ---------------------------------------------------------------------------------

    col_list_weather = ["dt_iso", "temp", "pressure", "humidity", "wind_speed", "weather_description"]
    weather_df = pd.read_csv("csv_files/sarimax/weather_features.csv", usecols=col_list_weather, index_col='dt_iso', parse_dates=True)

    weather_df= weather_df.asfreq('H')
    energy_weather_df=pd.concat([energy_df, weather_df], axis=1)

    # ---------------------------------------------------------------------------------

    train_df= energy_weather_df.iloc[:len(energy_weather_df)-100]
    test_df= energy_weather_df.iloc[len(energy_weather_df)-100:]

    #exog_train = train_df['temp']
    #exog_forecast = test_df['temp']

    # ---------------------------------------------------------------------------------

    ##SARIMAX (1, 1, 1)X(2, 0, 0, 24)
    mod= SARIMAX(train_df['total load actual'], order=(1,1,1), seasonal_order=(2,0,0,24))

    res= mod.fit()

    start= len(train_df)
    end= len(train_df) + len(test_df)-1

    prediction = res.predict(start, end).rename('Prediction')
    ax = test_df['total load actual'].plot(legend=True, figsize=(16,8))
    prediction.plot(legend=True)

    # ---------------------------------------------------------------------------------

    from statsmodels.tools.eval_measures import rmse
    ##print("Root mean squared error: "+rmse(test_df['total load actual'], prediction))
    
    # ---------------------------------------------------------------------------------

    print(prediction)