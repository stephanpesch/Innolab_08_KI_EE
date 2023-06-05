def rnn_algorithm(file, weather_file, checked_columns, checked_weather_columns,
                  col_names, weather_col_names):
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import sklearn.preprocessing
    from sklearn.metrics import r2_score
    from keras.layers import Dense,Dropout,SimpleRNN,LSTM
    from keras.models import Sequential
    from keras.optimizers import SGD

    # print("\nChecked_columns:")
    # print(checked_columns)
    # print("\nchecked_weather_columns:")
    # print(checked_weather_columns)
    # print("\ncol_names")
    # print(col_names)
    # print("\nweather_col_names")
    # print(weather_col_names)
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

    df = pd.concat([df_energy, df_weather], axis=1)

    df["total load actual"].plot(figsize=(16,7),legend=True)

    plt.title('hourly consumption - before normalization')

    plt.show()

    column_names = df.columns.tolist()
    column_to_predict = 1
    if 'total load actual' in column_names:
        column_to_predict = column_names.index('total load actual')

    number_of_columns = len(column_names)


    def normalize_data(df):
        scaler = sklearn.preprocessing.MinMaxScaler()
        df[column_names]=scaler.fit_transform(df[column_names].values.reshape(-1,len(column_names)))
        return df

    df_norm = normalize_data(df)
    df_norm.shape

    #Visualize data after normalization

    df_norm.plot(figsize=(16,7),legend=True)

    plt.title('hourly consumption - AFTER NORMALIZATION')

    plt.show()

    def load_data(stock, seq_len):
        X_train = []
        y_train = []
        for i in range(seq_len, len(stock)):
            X_train.append(stock.iloc[i - seq_len: i, :])
            y_train.append(stock.iloc[i, column_to_predict]) # there is a difference if i type 1 or a variable? why?

        # 1 last 6 months for prediction
        X_test = X_train[30648:]
        y_test = y_train[30648:]

        # 2 first 3 1/2 years for training
        X_train = X_train[:30648]
        y_train = y_train[:30648]

        # 3 convert to numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # 4 reshape data to input into RNN models
        X_train = np.reshape(X_train, (X_train.shape[0], seq_len, number_of_columns))
        X_test = np.reshape(X_test, (X_test.shape[0], seq_len, number_of_columns))

        return [X_train, y_train, X_test, y_test]

    #create train, test data
    seq_len = 20 #choose sequence length

    X_train, y_train, X_test, y_test = load_data(df, seq_len)

    print('X_train.shape = ',X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', X_test.shape)
    print('y_test.shape = ',y_test.shape)

    #RNN model

    rnn_model = Sequential()

    rnn_model.add(SimpleRNN(40, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(Dense(1))

    rnn_model.summary()

    rnn_model.compile(optimizer="adam",loss="mean_squared_error")
    history = rnn_model.fit(X_train, y_train, epochs=10, batch_size=1000)

    # Plot the training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    rnn_predictions = rnn_model.predict(X_test)

    #r2 score for the values predicted by the above trained SIMPLE RNN model
    rnn_score = r2_score(y_test,rnn_predictions)
    print("R2 Score of RNN model = ",rnn_score)

    def plot_predictions(test, predicted, title, time_index):
        plt.figure(figsize=(16, 7))
        time_index = time_index[-len(test):]  # Select the corresponding time index for the last 'len(test)' elements
        plt.plot(time_index, test, color='blue', label='Actual power consumption data')
        plt.plot(time_index, predicted, alpha=0.7, color='orange', label='Predicted power consumption data')
        plt.title(title)
        plt.xlabel('Time in hours')
        plt.ylabel('Normalized power consumption scale')
        plt.legend()
        plt.show()
    plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model", df_norm.index[seq_len:])

    # Get the index range for temperature values corresponding to predicted power consumption
    temp_index = range(seq_len, seq_len + len(rnn_predictions))

    # Plot temperature against predicted power consumption
    plt.figure(figsize=(10, 6))
    plt.scatter(df_norm["temp"].values[temp_index], rnn_predictions.flatten(), color='orange', label='Predicted Power Consumption')
    plt.xlabel('Temperature')
    plt.ylabel('Predicted Power Consumption')
    plt.title('Predicted Power Consumption vs Temperature')
    plt.legend()
    plt.show()