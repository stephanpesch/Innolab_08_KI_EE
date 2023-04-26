from datetime import datetime, timedelta

from matplotlib import pyplot as plt


def sarimax_predict(model, location):
    startPrediction = datetime.now()
    endPrediction = startPrediction + timedelta(hours=40)

    predictionFuture = model.predict(startPrediction, endPrediction).rename('Prediction')
    #ax = test_df[useColumns[1]].plot(legend=True, figsize=(16, 8))
    predictionFuture.plot(legend=True)
    # ---------------------------------------------------------------------------------

    print(predictionFuture)

    plt.show()
