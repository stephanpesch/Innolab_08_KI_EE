def rnn_algorithm(df, checked_columns):
    import pandas as pd
    import numpy as np # linear algebra # not used
    import matplotlib.pyplot as plt
    import sklearn.preprocessing # not used
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn import tree # not used
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.metrics import r2_score
    from keras.models import Sequential
    from keras.layers import Dense,Dropout,SimpleRNN,LSTM
    from keras.preprocessing.sequence import TimeseriesGenerator # not used

    data = pd.read_csv('csv_files/rnn/energy_consumption_levels.csv')
    X = data.drop(columns=['consumption'])
    y= data['consumption']
    consumptions = []
    for x in y:
        addthis = str(x)
        consumptions.append(addthis)
    X_train,X_test,y_train,y_test = train_test_split(X,consumptions,test_size=0.2)

    consumption_classifier = DecisionTreeClassifier(max_leaf_nodes=20,random_state=0)
    consumption_classifier.fit(X_train,y_train)
    predictions = consumption_classifier.predict(X_test)

    score = accuracy_score(y_test, predictions)
    print(score)

    # ---------------------------------------------------------------------------------

    data = pd.read_csv('csv_files/rnn/energy_consumption_levels.csv')
    X = data.drop(columns=['consumption'])
    y= data['consumption']
    consumptions = []
    for x in y:
        addthis = str(x)
        consumptions.append(addthis)
    X_train,X_test,y_train,y_test = train_test_split(X,consumptions,test_size=0.2)
        
    clf=RandomForestClassifier(n_estimators=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, predictions)
    print(score)

    # ---------------------------------------------------------------------------------

    data = pd.read_csv('csv_files/rnn/energy_consumption_levels.csv')
    X = data.drop(columns=['consumption'])
    y= data['consumption']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # ---------------------------------------------------------------------------------

    rnn_model = Sequential()

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(Dense(1))

    rnn_model.summary()

    # ---------------------------------------------------------------------------------

    rnn_model.compile(optimizer="adam",loss="MSE")
    history= rnn_model.fit(X_train, y_train, epochs=17, validation_data=(X_test, y_test), batch_size=1000)

    # ---------------------------------------------------------------------------------

    # if plt.figure(...) is not specified, first graph gets skipped (???)
    plt.figure(figsize=(7,7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show(block=False)

    # ---------------------------------------------------------------------------------

    rnn_predictions = rnn_model.predict(X_test)

    rnn_score = r2_score(y_test,rnn_predictions)
    print("R^2 Score of RNN model = ",rnn_score)

    # ---------------------------------------------------------------------------------

    def plot_predictions(y_test, rnn_predictions, title):
        # if plt.figure(...) is not specified, first graph gets skipped (???)
        plt.figure(figsize=(13,4))
        plt.plot(y_test, color='blue',label='Actual power consumption data')
        plt.plot(rnn_predictions, alpha=0.7, color='orange',label='Predicted power consumption data')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Normalized power consumption scale')
        plt.legend()
        plt.show(block=False)
        
    plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model")