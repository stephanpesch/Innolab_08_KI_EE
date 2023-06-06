def xgboost_train(file, weather_file, checked_columns, checked_weather_columns, col_names, weather_col_names, grid_var):
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score

    useColumns = []
    i = 0

    for column in checked_columns:
        if (checked_columns[i].get() == 1):
            useColumns.append(col_names[i])
        i = i + 1

    df_energy = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)


    indx_col = useColumns[0]
    power_col = useColumns[1]

    df_energy.sort_values(by=useColumns[0], ascending=True)
    df_energy = df_energy.fillna(method='ffill')
    df_energy = df_energy.sort_values(by=useColumns[0], ascending=True)
    df_energy.dropna(axis=0, how='any', subset=None, inplace=True)

    df_energy = df_energy[~df_energy.index.duplicated(keep='first')]

    df_energy = df_energy.fillna(method='ffill')

    df_energy = df_energy.asfreq('H')
    print(df_energy)

    # ---------------------------------------------------------------------------------

    useWeatherColumns = []
    i = 0

    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1

    df_weather = pd.read_csv(weather_file, usecols=useWeatherColumns, index_col=useWeatherColumns[0], parse_dates=True)
    df_weather = df_weather.asfreq('H')
    print(df_weather)

    # -----------------------------------------------------------------------------------
    # training section

    # Before building and training our model, let's split the data into training and testing
    energy_weather_df=pd.concat([df_energy, df_weather], axis=1)

    train_data = energy_weather_df.iloc[:len(energy_weather_df) - 48]
    test_data = energy_weather_df.iloc[len(energy_weather_df) - 48:]

    X_train, y_train = create_features(train_data, label=power_col)
    X_test, y_test = create_features(test_data, label=power_col)

    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())

    print(X_train)

    # --------------------------------------------------------------------------------------
    # Grid Search

    global xgb_model_train
    global xgb_pred

    if grid_var.get() == 1:
        print("Performing Grid Search")

        param_grid = {
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }

        '''
        'learning_rate': [0.1, 0.01, 0.001],
            colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        '''

        xgb_model = xgb.XGBRegressor(n_estimators=300, colsample_bytree=0.8, learning_rate=0.05)

        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   verbose=3)
        grid_search.fit(X_train, y_train)

        xgb_model_train = grid_search.best_estimator_
        print(grid_search.best_params_)

        xgb_pred = xgb_model_train.predict(X_test)
    else:
        xgb_model_train = xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.05, max_depth=8,
                                     min_child_weight=4, n_estimators=1000, subsample=0.5)

        xgb_model_train.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=False)

        xgb_pred = xgb_model_train.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, xgb_pred))  # root mean square error
    print("rmse: " + str(rmse))
    r2_score = r2_score(y_test, xgb_pred)
    print("R2: ", r2_score)

    df_plot = pd.DataFrame({'y_test': y_test, 'xgb_pred': xgb_pred})
    plt.figure(figsize=(20, 8))
    df_plot['y_test'].plot(label='Actual')
    df_plot['xgb_pred'].plot(label='Predicted')
    plt.text(16770, 3250, 'RMSE: {}'.format(rmse), fontsize=20, color='red')
    plt.title('Testing Set Forecast', weight='bold', fontsize=25)
    plt.legend()
    plt.show()

    return xgb_model_train


def create_features(df, label):
    df['Hour'] = df.index.hour
    df['Dayofweek'] = df.index.dayofweek
    df['Dayofmonth'] = df.index.day
    df['Dayofyear'] = df.index.dayofyear
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year

    X = df.drop(label, axis=1)
    y = df[label]
    return X, y

