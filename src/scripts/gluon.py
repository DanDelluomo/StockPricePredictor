#!/usr/bin/env python

# Python Standard Library Modules
import os
import pathlib
import sys
import warnings

# External Libraries
from gluonts.dataset import common
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model import deepar
from gluonts.mx.trainer import Trainer
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
import matplotlib
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
mx.random.seed(0)
np.random.seed(0)

prediction_length = 30
validation_length = 30
if validation_length:
    prediction_length = prediction_length + validation_length


def covert_yahoo_series_dir(path: str, prediction_length: int) -> list:
    """Clean and load all coin histories in the Yahoo Finance folder

    Params:
        path: folder full of historical crypto coins timeseries data
        prediction_length: length on which to predict
    Returns:
        List of Gluon-compatible dicts from the coin data
    """
    gluon_list = []
    for file in os.listdir(path):
        coin_gluon_dict = dict()
        file_path = path + file
        coin = pd.read_csv(file_path)
        coin["Date"] = pd.to_datetime(coin["Date"])
        coin.set_index("Date", inplace=True)
        coin.dropna(inplace=True)
        if len(coin) < 100:
            continue
        coin = coin.asfreq("D")

        # Get values for ListDatasets
        coin_closes = coin["Close"]
        coin_closes.index = pd.DatetimeIndex(coin_closes.index)
        coin_closes = coin_closes.asfreq("D")
        start = coin_closes.index[0]

        coin_gluon_dict["test"] = {
            "start": start,
            "target": coin_closes,
            "name": file,
        }

        coin_gluon_dict["validation"] = {
            "start": start,
            "target": coin_closes[:-30],
            "name": file,
        }

        coin_gluon_dict["train"] = {
            "start": start,
            "target": coin_closes[:-60],
            "name": file,
        }

        gluon_list.append(coin_gluon_dict)

    return gluon_list


gluon_list = covert_yahoo_series_dir("../data/historical_yahoo/", 30)


train = btc_history.iloc[:-prediction_length]
test = btc_history
train = train[["Close"]]
test = test[["Close"]]
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


test_data = common.ListDataset(
    [
        {
            "start": test.index[0],
            "target": train["Close"],
        },
    ],
    freq="D",
)

data = common.ListDataset(
    [
        {
            "start": train.index[0],
            "target": train["Close"],
        },
    ],
    freq="D",
)

validation_data = common.ListDataset(
    [
        {
            "start": test.index[0],
            "target": test["Close"][:-validation_length],
        }
    ],
    freq="D",
)

for coin_gluon_dict in gluon_list:
    data.list_data.append(coin_gluon_dict)
    new_dict = dict()
    new_dict["start"] = coin_gluon_dict["start"]
    new_dict["target"] = coin_gluon_dict["target"][:-30]
    test_data.list_data.append(new_dict)


trainer = Trainer(epochs=5, batch_size=256, learning_rate=0.01)
estimator = deepar.DeepAREstimator(
    freq="D",
    num_cells=100,
    num_layers=2,
    cell_type="lstm",
    #         num_layers=3,
    prediction_length=prediction_length,
    trainer=trainer,
    context_length=prediction_length,
    use_feat_dynamic_real=True,
)

predictor = estimator.train(
    training_data=data,
    #     validation_data=validation_data
)

mx.random.seed(0)
global_loss = 0
predictions = predictor.predict(train_data.list_data)
for index, value in enumerate(range(len(gluon_list))):
    prediction = next(predictions)
    name = test_data.list_data[index]['name']
    # Skip graphs with absurd loss, stablecoins etc.
    if name in {"USDT-USD.csv", "CCXX-USD.csv", "TUSD-USD.csv"}:
        continue
    print(name)
    full_actual = test_data.list_data[index]['target']
    actual = full_actual[-30:]
    preds = pd.Series(prediction.mean)
    preds.index = actual.index
    plt.figure()
    preds.plot(legend=True, label=f"{name} PREDICTED")
    actual.plot(legend=True, label=f"{name} ACTUAL")
    plt.show()

    # SCALING
    scaler = MinMaxScaler()
    scaled_actual = np.array(actual)
    scaler.fit([scaled_actual])
    scaled_actual = scaler.fit_transform(np.array(scaled_actual[:, np.newaxis]))
    scaled_preds = scaler.transform([preds])
    scaled_preds = scaled_preds.reshape(-1, 1)
    mse = mean_squared_error(scaled_actual, scaled_preds)
    print(f"mse: {mse}")
    global_loss += mse
    
print(f"global_loss is {global_loss}")
