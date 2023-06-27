def rnn_train(file, weather_file, checked_columns, checked_weather_columns,
                  col_names, weather_col_names, rootWindow, grid_var):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.preprocessing
    from sklearn.metrics import r2_score
    from keras.layers import Dense, Dropout, SimpleRNN
    from keras.models import Sequential
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import GridSearchCV
    
    useColumns = []
    i = 0

    # Select columns based on the user's selection
    for column in checked_columns:
        if (checked_columns[i].get() == 1):
            useColumns.append(col_names[i])
        i = i + 1

    # Read energy data from file
    df_energy = pd.read_csv(file, usecols=useColumns, index_col=useColumns[0], parse_dates=True)

    # Sort and preprocess energy data
    df_energy = df_energy.fillna(method='ffill')
    df_energy = df_energy.sort_values(by=useColumns[0], ascending=True)
    df_energy.dropna(axis=0, how='any', subset=None, inplace=True)
    df_energy = df_energy.asfreq('H')
    print(df_energy)

    useWeatherColumns = []
    i = 0

    # Select weather columns based on the user's selection
    for column in checked_weather_columns:
        if (checked_weather_columns[i].get() == 1):
            useWeatherColumns.append(weather_col_names[i])
        i = i + 1

    # Read weather data from file
    useWeatherColumns = ["dt_iso", "temp"]
    df_weather = pd.read_csv(weather_file, usecols=useWeatherColumns, index_col=useWeatherColumns[0], parse_dates=True)
    df_weather = df_weather.fillna(method='ffill')
    df_weather = df_weather.asfreq('H')
    print(df_weather)

    # Concatenate energy and weather data
    df = pd.concat([df_weather, df_energy], axis=1)

    column_names = df.columns.tolist()
    column_to_predict = df.columns[-1]
    if 'total load actual' in column_names:
        column_to_predict = column_names.index('total load actual')
    number_of_columns = len(column_names)

    def normalize_data(df):
        scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_data = df.copy()
        normalized_data[column_names] = scaler.fit_transform(normalized_data[column_names].values.reshape(-1, len(column_names)))
        return normalized_data, scaler
    def denormalize_data(df, scaler):
        for column in column_names:
            if column in df.columns:
                column_data = df[column].values
                if len(column_data.shape) == 1:
                    column_data = column_data.reshape(-1, 2)
                df[column] = scaler.inverse_transform(column_data).flatten()
        return df

    def create_features(df, label=None):
        df['Hour'] = df.index.hour
        df['Dayofweek'] = df.index.dayofweek
        df['Dayofmonth'] = df.index.day
        df['Dayofyear'] = df.index.dayofyear
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year

        X = df.drop(label, axis=1)
        if label:
            y = df[label]
            return X, y
        return X

    # Normalize data
    df_norm, scaler = normalize_data(df)

    # Visualize data after normalization
    # df_norm.plot(figsize=(16, 7), legend=True)
    # plt.title('Hourly Consumption - AFTER NORMALIZATION')
    # plt.show(block=False)

    def load_data(stock, seq_len):
        X_train = []
        y_train = []
        for i in range(seq_len, len(stock)):
            X_train.append(stock.iloc[i - seq_len: i, 0])
            y_train.append(stock.iloc[i, column_to_predict])

        # Split data into train and test sets
        X_test = X_train[30648:]
        y_test = y_train[30648:]
        X_train = X_train[:30648]
        y_train = y_train[:30648]

        # Convert to numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Reshape data for RNN input
        X_train = np.reshape(X_train, (X_train.shape[0], seq_len, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

        return [X_train, y_train, X_test, y_test]

    # Create train and test data
    seq_len = 20  # Choose sequence length
    X_train, y_train, X_test, y_test = load_data(df_norm, seq_len)

    print('X_train.shape = ', X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', X_test.shape)
    print('y_test.shape = ', y_test.shape)

    # RNN model
    def create_rnn_model(neurons=60, dropout=0.1):
        model = Sequential()
        model.add(SimpleRNN(neurons, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(neurons, activation="tanh", return_sequences=True))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(neurons, activation="tanh", return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.summary()
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    rnn_model = KerasRegressor(build_fn=create_rnn_model, verbose=0)

    # Perform grid search if GridSearch was checked
    if grid_var == 1:
        # Define the hyperparameters to tune
        param_grid = {
        'neurons': [20, 40, 60],
        'dropout': [0.1, 0.2, 0.3]
        # takes 3+ hours with those parameters included, only 10min with 2 above
        # 'epochs': [5, 10, 15],
        # 'batch_size': [500, 1000, 2000]
        }

        grid = GridSearchCV(estimator=rnn_model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(X_train, y_train)

        # Print the best parameters and best score
        print("Best Parameters: ", grid_result.best_params_)
        print("Best Score: ", grid_result.best_score_)

        # Train the model with the best parameters
        best_model = create_rnn_model(neurons=grid_result.best_params_['neurons'],
                                    dropout=grid_result.best_params_['dropout'])
        history = best_model.fit(X_train, y_train,
                                epochs=grid_result.best_params_['epochs'],
                                batch_size=grid_result.best_params_['batch_size'],
                                validation_data=(X_test, y_test))
    else:
        # create RNN model with fixed parameters
        best_model = create_rnn_model(neurons=60, dropout=0.1)
        history = best_model.fit(X_train, y_train, epochs=15, batch_size=500, validation_data=(X_test, y_test))

    # Plot the training loss and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)

    rnn_predictions = best_model.predict(X_test)

    # R2 score for the values predicted by the RNN model
    rnn_score = r2_score(y_test, rnn_predictions)
    print("R2 Score of RNN model = ", rnn_score)

    df_denorm = denormalize_data(df_norm, scaler)
    y_test_denorm = denormalize_data(pd.DataFrame(y_test, columns=[column_names[column_to_predict]]), scaler)
    rnn_predictions_denorm = denormalize_data(pd.DataFrame(rnn_predictions, columns=[column_names[column_to_predict]]), scaler)

    def plot_predictions(test, predicted, title, time_index):
        plt.figure(figsize=(16, 7))
        time_index = time_index[-len(test):]
        plt.plot(time_index, test, color='blue', label='Actual power consumption data')
        plt.plot(time_index, predicted, alpha=0.7, color='orange', label='Predicted power consumption data')
        plt.title(title)
        plt.xlabel('Time in hours')
        plt.ylabel('Normalized power consumption scale')
        plt.legend()
        plt.show(block=False)
        
    def plot_predictions_hours(test, predicted, title, time_index):
        plt.figure(figsize=(16, 7))
        time_index = time_index[-len(test):]
        plt.plot(time_index[-48:], test[-48:], color='blue', label='Actual power consumption data')  # Display last 48 rows
        plt.plot(time_index[-48:], predicted[-48:], alpha=0.7, color='orange', label='Predicted power consumption data')  # Display last 24 rows
        plt.title(title)
        plt.xlabel('Time in hours')
        plt.ylabel('Normalized power consumption scale')
        plt.legend()
        plt.show(block=False)

    plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model", df_norm.index[seq_len:])
    plot_predictions_hours(y_test, rnn_predictions, "Predictions made by simple RNN model last 48 hours", df_norm.index[seq_len:])

    return best_model