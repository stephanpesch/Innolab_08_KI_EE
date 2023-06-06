import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential


def lstm_algorithm(file, weather_file, checked_columns, checked_weather_columns,
                   col_names, weather_col_names, grid_var):
    use_cols = get_selected_columns(checked_columns, col_names)

    df_long = read_csv_and_reduce(file, use_cols, col_names)
    df = group_by_and_compute_mean(df_long)

    dataset, scaler = preprocess_dataset(df)

    train, test = split_dataset(dataset)

    look_back = 1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    train_x = reshape_dataset(train_x)
    test_x = reshape_dataset(test_x)

    if grid_var.get() == 1:
        # Hyperparameters to search over
        param_grid = {
            'units': [4, 8, 16],
            'batch_size': [1, 2, 4],
            'epochs': [10, 15, 20]
        }

        # Create the KerasRegressor wrapper for the LSTM model
        model = KerasRegressor(build_fn=create_lstm_model, verbose=0)

        # Create the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=3
        )

        # Fit the GridSearchCV object to the training data
        grid_result = grid_search.fit(train_x, train_y)

        print(grid_result.best_params_)

        # Get the best model from the grid search
        best_model = grid_result.best_estimator_.model
    else:
        best_model = build_and_train_lstm_model(train_x, train_y, look_back)

    train_predict, test_predict = make_predictions(best_model, train_x, test_x)

    train_predict, train_y = inverse_transform_predictions(train_predict, train_y, scaler)
    test_predict, test_y = inverse_transform_predictions(test_predict, test_y, scaler)

    train_score = calculate_rmse(train_y[0], train_predict[:, 0])
    print('Train Score: %.2f RMSE' % train_score)
    test_score = calculate_rmse(test_y[0], test_predict[:, 0])
    print('Test Score: %.2f RMSE' % test_score)

    train_predict_plot, test_predict_plot = prepare_plot_data(dataset, train_predict, test_predict, look_back)

    plot_data(scaler.inverse_transform(dataset), train_predict_plot, test_predict_plot)
    return best_model


def get_selected_columns(checked_columns, col_names):
    return [checked_column for col_name, checked_column in zip(checked_columns, col_names) if col_name.get() == 1]


def read_csv_and_reduce(file, use_cols, col_names):
    df_long = pd.read_csv(file, header=0, usecols=use_cols, parse_dates=True, index_col=col_names[0])
    print(df_long.head())
    print(df_long.info())
    return df_long


def group_by_and_compute_mean(df_long):
    df = df_long.tail(48)
    print(df)
    return df


def preprocess_dataset(df):
    tf.random.set_seed(7)
    dataset = df.values
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


def split_dataset(dataset):
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))
    return train, test


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


def reshape_dataset(dataset):
    return np.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))


def create_lstm_model(look_back=1, units=16):
    model = Sequential()
    model.add(LSTM(units, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_and_train_lstm_model(train_x, train_y, look_back):
    model = create_lstm_model(look_back=look_back)
    model.fit(train_x, train_y, epochs=20, batch_size=1, verbose=2)
    return model


def make_predictions(model, train_x, test_x):
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    return train_predict, test_predict


def inverse_transform_predictions(predictions, true_values, scaler):
    predictions = scaler.inverse_transform(predictions)
    true_values = scaler.inverse_transform([true_values])
    return predictions, true_values


def calculate_rmse(true_values, predictions):
    return np.sqrt(mean_squared_error(true_values, predictions))


def prepare_plot_data(dataset, train_predict, test_predict, look_back):
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict

    return train_predict_plot, test_predict_plot


def plot_data(dataset, train_predict_plot, test_predict_plot):
    plt.figure(figsize=(6, 5))
    plt.plot(dataset)
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show(block=False)
