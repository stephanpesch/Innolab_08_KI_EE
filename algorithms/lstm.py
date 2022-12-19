def lstm_algorithm():

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    df_long = pd.read_csv("csv_files/lstm/electricity-consumption-raw.csv", header=0, usecols=["datetime","consumption"], parse_dates=["datetime"])
    df_long["month"] = df_long["datetime"].dt.month
    df_long["year"] = df_long["datetime"].dt.year
    df_long["day"] = df_long["datetime"].dt.day
    df = df_long.groupby(["year", "month", "day"]).mean()
    #df = df_long.head(1000)

    # ---------------------------------------------------------------------------------

    df_long = pd.read_csv("csv_files/lstm/electricity-consumption-raw.csv", header=0, usecols=["datetime","consumption"], parse_dates=["datetime"])
    df_long["month"] = df_long["datetime"].dt.month
    df_long["year"] = df_long["datetime"].dt.year
    df_long["day"] = df_long["datetime"].dt.day
    df = df_long.groupby(["year", "month", "day"]).mean()
    #df = df_long.head(1000)

    # ---------------------------------------------------------------------------------

    df.tail()

    # ---------------------------------------------------------------------------------

    df.info()

    # ---------------------------------------------------------------------------------

    # fix random seed for reproducibility
    tf.random.set_seed(7)

    # ---------------------------------------------------------------------------------

    dataset = df.values
    dataset = dataset.astype('float32')

    # ---------------------------------------------------------------------------------

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # ---------------------------------------------------------------------------------

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))

    # ---------------------------------------------------------------------------------

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # ---------------------------------------------------------------------------------

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # ---------------------------------------------------------------------------------

    trainY
    
    # ---------------------------------------------------------------------------------

    trainX

    # ---------------------------------------------------------------------------------

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # ---------------------------------------------------------------------------------

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # epochs changed from 100 to 15 only for test purposes 
    model.fit(trainX, trainY, epochs=15, batch_size=1, verbose=2)

    # ---------------------------------------------------------------------------------

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # ---------------------------------------------------------------------------------

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.figure(figsize=(6, 5))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show(block=False)