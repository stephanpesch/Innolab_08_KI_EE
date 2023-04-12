def index_to_column(data, indx_col, power_col):
    import pandas as pd
    data = data.reset_index()
    data[indx_col] = pd.to_datetime(data[indx_col])
    data = data.sort_values(indx_col)

    data = data.rename(columns={indx_col: 'ds', power_col: 'y'})
    return data

def xgboost_algorithm(file, checked_columns, col_names):
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from prophet import Prophet
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.model_selection import GridSearchCV

    useColumns = []
    i = 0

    for column in checked_columns:
        if (checked_columns[i].get() == 1):
            useColumns.append(col_names[i])
        i = i + 1

    df = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    print(df)

    indx_col = useColumns[0]
    power_col = useColumns[1]

    df.sort_values(by=useColumns[0], ascending=True)
    df = df.fillna(method='ffill')
    df = df.sort_values(by=useColumns[0], ascending=True)
    df.dropna(axis=0, how='any', subset=None, inplace=True)

    df = df[~df.index.duplicated(keep='first')]

    df = df.asfreq('H')
    df = df.fillna(method='ffill')

    #############################################################################################
    #training section

    # Before building and training our model, let's split the data into training and testing
    #df_train, df_test = df[df.index < '2016-01-01'], df[df.index >= '2016-01-01']
    df_train, df_test = df[:], df[-22000:]

    X_train, y_train = date_transform(df_train, power_col)
    X_test, y_test = date_transform(df_test, power_col)

    print(X_train)
    print(y_train)

    print(X_test)
    print(y_test)

    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=10)
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    xgb_pred = xgb_model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, xgb_pred))  # root mean square error

    df_plot = pd.DataFrame({'y_test': y_test, 'xgb_pred': xgb_pred})

    plt.figure(figsize=(20, 8))

    df_plot['y_test'].plot(label='Actual')
    df_plot['xgb_pred'].plot(label='Predicted')
    plt.text(16770, 3250, 'RMSE: {}'.format(rmse), fontsize=20, color='red')
    plt.title('Testing Set Forecast', weight='bold', fontsize=25)
    plt.legend()

    #####################################################################################################
    #prediction section
    new_df = index_to_column(df, indx_col, power_col)
    new_df = remove_tz_from_dataframe(new_df)
    prophet_model2 = Prophet(interval_width=0.95)
    prophet_model2.fit(new_df)
    # 7 days to the future (7x24 = 168)
    future_dates = prophet_model2.make_future_dataframe(periods=168, freq='H')

    future_dates2 = future_dates.iloc[-168:, :].copy()

    future_dates2['ds'] = pd.to_datetime(future_dates2['ds'])
    future_dates2 = future_dates2.set_index('ds')

    future_dates2['Hour'] = future_dates2.index.hour
    future_dates2['Dayofweek'] = future_dates2.index.dayofweek
    future_dates2['Dayofmonth'] = future_dates2.index.day
    future_dates2['Dayofyear'] = future_dates2.index.dayofyear
    future_dates2['weekofyear'] = future_dates2.index.weekofyear
    future_dates2['Month'] = future_dates2.index.month
    future_dates2['Quarter'] = future_dates2.index.quarter
    future_dates2['Year'] = future_dates2.index.year

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    xgb_model2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    xgb_model2.fit(X, y)
    xgb_pred2 = xgb_model2.predict(future_dates2)

    df_plot2 = pd.DataFrame({'Hour': future_dates2['Hour'], 'xgb_pred2': xgb_pred2})

    #last_week = df['2018-07-01':'2018-08-15']
    last_week = df[-168:]

    plt.figure(figsize=(20, 8))

    last_week[power_col].plot()
    df_plot2['xgb_pred2'].plot()
    plt.title('7 Days Forecast', weight='bold', fontsize=25)
    plt.show()

def date_transform(data, power_col):
    df = data.copy()

    print(data)

    df['Hour'] = df.index.hour
    df['Dayofweek'] = df.index.dayofweek
    df['Dayofmonth'] = df.index.day
    df['Dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.weekofyear
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year

    X = df.drop(power_col, axis=1)
    y = df[power_col]

    return X, y

def remove_tz_from_dataframe(df_in):
    import pandas as pd
    df = df_in.copy()
    col_times = [ col for col in df.columns if any([isinstance(x, pd.Timestamp) for x in df[col]])]
    for col in col_times:
        df[col] = pd.to_datetime(
            df[col], infer_datetime_format=True)
        df[col] = df[col].dt.tz_localize(None)
    return df




