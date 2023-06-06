import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential


def lstm_algorithm(file, weather_file, checked_columns, checked_weather_columns,
                   col_names, weather_col_names, grid_var):
    # Use only columns of col_names that have the value of 1 in checked_columns
    use_cols = [checked_column for col_name, checked_column in zip(checked_columns, col_names) if col_name.get() == 1]

    # Reduce size of dataset so my computer can handle it
    print(use_cols)
    df_long = pd.read_csv(file, header=0,
                          usecols=use_cols, parse_dates=[col_names[0]])
    print(df_long.head())
    print(df_long.info())
    df_long["month"] = df_long[col_names[0]].dt.month
    df_long["year"] = df_long[col_names[0]].dt.year
    df_long["day"] = df_long[col_names[0]].dt.day
    df = df_long.groupby(["year", "month", "day"]).mean()

    # fix random seed for reproducibility
    tf.random.set_seed(7)

    dataset = df.values
    dataset = dataset.astype('float32')

    # normalize the dataset between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    # reshape into X=t and Y=t+1
    look_back = 1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # epochs changed from 100 to 15 only for test purposes 
    model.fit(train_x, train_y, epochs=15, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # calculate root mean squared error
    train_score = np.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    test_score = np.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % test_score)

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict
    # plot baseline and predictions
    plt.figure(figsize=(6, 5))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show(block=False)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)
