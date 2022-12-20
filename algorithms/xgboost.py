def xgboost_algorithm():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from prophet import Prophet
    from xgboost import XGBRegressor

    df = pd.read_csv('csv_files/xgboost/DAYTON_hourly.csv', index_col='Datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # The data that we are going to use
    df.head()

    # Before building and training our model, let's split the data into training and testing
    df_train, df_test = df[df.index < '2016-01-01'], df[df.index >= '2016-01-01']

    X_train, y_train = date_transform(df_train)
    X_test, y_test = date_transform(df_test)

    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=10)
    xgb_model.fit(X_train, y_train, eval_metric='mae', eval_set=[(X_train, y_train), (X_test, y_test)])

    def index_to_column(data):
        data = data.reset_index()
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.sort_values('Datetime')

        data = data.rename(columns={'Datetime': 'ds', 'DAYTON_MW': 'y'})
        return data

    new_df = index_to_column(df)
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
    xgb_model2.fit(X, y, eval_metric='mae')
    xgb_pred2 = xgb_model2.predict(future_dates2)

    df_plot2 = pd.DataFrame({'Hour': future_dates2['Hour'], 'xgb_pred2': xgb_pred2})

    last_week = df['2018-07-01':'2018-08-15']

    plt.figure(figsize=(20, 8))

    last_week['DAYTON_MW'].plot()
    df_plot2['xgb_pred2'].plot()
    plt.title('7 Days Forecast', weight='bold', fontsize=25)
    plt.show()

def date_transform(data):
    df = data.copy()

    df['Hour'] = df.index.hour
    df['Dayofweek'] = df.index.dayofweek
    df['Dayofmonth'] = df.index.day
    df['Dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.weekofyear
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year

    X = df.drop('DAYTON_MW', axis=1)
    y = df['DAYTON_MW']

    return X, y



