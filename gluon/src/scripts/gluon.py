# Python Standard Library Modules
import os
import pathlib
import sys
import warnings

# External Libraries

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from gluonts.dataset import common
from gluonts.model import deepar
import mxnet as mx
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.mx.trainer import Trainer


warnings.filterwarnings("always")
mx.random.seed(0)
np.random.seed(0)

prediction_length = 30
validation_length = 30
if validation_length:
    prediction_length = prediction_length + validation_length


btc_history = pd.read_csv("Gemini_Bitcoin_Historical.csv")
btc_history = btc_history[::-1]
btc_history = btc_history.drop(["Unix Timestamp", "Symbol"], axis=1)
btc_history["Date"] = pd.to_datetime(btc_history["Date"])
btc_history.set_index("Date", inplace=True)
btc_history.index.freq = "D"
btc_history["Yesterday_Volume"] = btc_history["Volume"].shift(1)
btc_history["Yesterday_Volume"] = btc_history["Volume"].shift(1)
btc_history["Yesterday_Close"] = btc_history["Close"].shift(1)
btc_history["Yesterday_High"] = btc_history["High"].shift(1)
btc_history["Yesterday_Low"] = btc_history["Low"].shift(1)
btc_history["Yesterday_Spread"] = (
    btc_history["Yesterday_High"] - btc_history["Yesterday_Low"]
) / btc_history["Yesterday_Close"]
btc_history["dayofweek"] = btc_history.index.dayofweek
btc_history = btc_history[1:]
btc_history["TripleExp"] = (
    ExponentialSmoothing(
        btc_history["Yesterday_Close"], trend="mul", seasonal="mul", seasonal_periods=7
    )
    .fit()
    .fittedvalues
)

btc_closes = btc_history["Close"]


dyn_btc_history = btc_history[
    ["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]
]
dyn_btc_history = dyn_btc_history.iloc[:-prediction_length]
dyn_btc_history = dyn_btc_history.values

# For Gluon feat_dynamic_real features: test
test_adjust = btc_history[
    ["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]
]
test_adjust = test_adjust.values
test_adjust = test_adjust.T

# For Gluon feat_dynamic_real features: validation_data
val_adjust = btc_history[
    ["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]
][:-validation_length]
val_adjust = val_adjust.values
val_adjust = val_adjust.T

# For Gluon feat_dynamic_real features: train
adjust = btc_history[["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]][
    :-prediction_length
]
adjust = adjust.values
adjust = adjust.T


def covert_yahoo_series_dir(path: str, prediction_length: int) -> list:
    """Clean and load all the coins in the Yahoo Finance folder

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
        na_vals = coin.isna().sum()[0]
        if na_vals > 0:
            print(f"{file[:-4]} has {na_vals} NA rows")
        coin["Date"] = pd.to_datetime(coin["Date"])
        coin.set_index("Date", inplace=True)
        coin.dropna(inplace=True)
        if len(coin) == 0:
            continue
        coin.index = pd.DatetimeIndex(coin.index).to_period("D")
        coin["Yesterday_Close"] = coin["Close"].shift(1)
        coin["Yesterday_High"] = coin["High"].shift(1)
        coin["Yesterday_Low"] = coin["Low"].shift(1)
        coin["Yesterday_Spread"] = (
            coin["Yesterday_High"] - coin["Yesterday_Low"]
        ) / coin["Yesterday_Close"]
        coin["dayofweek"] = coin.index.dayofweek
        coin = coin[1:]
        coin["TripleExp"] = (
            ExponentialSmoothing(
                coin["Yesterday_Close"], trend="add", seasonal="add", seasonal_periods=7
            )
            .fit()
            .fittedvalues
        )
        coin = coin.asfreq("D")

        #         x = coin.values #returns a numpy array
        #         min_max_scaler = preprocessing.MinMaxScaler()
        #         x_scaled = min_max_scaler.fit_transform(x)
        #         new_columnless_frame = pd.DataFrame(x_scaled)
        #         new_columnless_frame.columns = coin.columns
        #         coin = new_columnless_frame

        coin_closes = coin["Close"]
        coin_closes = coin_closes.asfreq("D")
        coin_gluon_dict["target_test"] = coin_closes
        coin_closes = coin["Close"][:-prediction_length]
        coin_closes = coin_closes.asfreq("D")
        coin_gluon_dict["start"] = coin_closes.index[0].to_timestamp()
        coin_gluon_dict["target"] = coin_closes

        coin_features_train = coin[
            ["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]
        ]
        coin_features_train = coin_features_train[:-prediction_length]
        coin_features_train = coin_features_train.asfreq("D")
        coin_features_predict = coin[
            ["Yesterday_Close", "Yesterday_Spread", "dayofweek", "TripleExp"]
        ]
        coin_features_predict = coin_features_predict.asfreq("D")

        coin_features_train = coin_features_train.values
        coin_features_train = coin_features_train.T

        coin_features_predict = coin_features_predict.values
        coin_features_predict = coin_features_predict.T

        coin_gluon_dict["feat_dynamic_real"] = coin_features_train
        coin_gluon_dict["feat_dynamic_real_predict"] = coin_features_predict

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
            "feat_dynamic_real": test_adjust,
        },
    ],
    freq="D",
)

data = common.ListDataset(
    [
        {
            "start": train.index[0],
            "target": train["Close"],
            "feat_dynamic_real": adjust,
        },
    ],
    freq="D",
)

validation_data = common.ListDataset(
    [
        {
            "start": test.index[0],
            "target": test["Close"][:-validation_length],
            "feat_dynamic_real": val_adjust,
        }
    ],
    freq="D",
)

for coin_gluon_dict in gluon_list:
    data.list_data.append(coin_gluon_dict)
    new_dict = dict()
    new_dict["start"] = coin_gluon_dict["start"]
    new_dict["target"] = coin_gluon_dict["target"][:-30]
    new_dict["feat_dynamic_real"] = coin_gluon_dict["feat_dynamic_real_predict"]
    test_data.list_data.append(new_dict)

# Hyperopt :
#
# def objective123(params):
#     batch_size = params['batch_size']
#     learning_rate = params['learning_rate']
#     params = {k: v for k, v in params.items() if k not in ('batch_size', 'learning_rate')}
#     trainer = Trainer(epochs=1, batch_size=batch_size)
#     estimator = deepar.DeepAREstimator(
#         freq="1D",
#         prediction_length=prediction_length,
#         trainer=trainer,
#         **params
#     )
#     predictor = estimator.train(training_data=data)
#     prediction = next(predictor.predict(data))
#     accuracy = mean_squared_error(test[:prediction_length].Close, prediction.mean)
#     return {'loss': accuracy, 'status': STATUS_OK}


# search_space = {
#     'num_layers': scope.int(hp.quniform('num_layers', 1, 8, q=1)),
#     'num_cells': scope.int(hp.quniform('num_cells', 30, 100, q=1)),
#     'cell_type': hp.choice('cell_type', ['lstm', 'gru']),
#     'batch_size': scope.int(hp.quniform('batch_size', 16, 256, q=1)),
#     'learning_rate': hp.quniform('learning_rate', 1e-5, 1e-1, 0.00005),
#     'context_length': scope.int(hp.quniform('context_length', 1, 200, q=1)),
# }

# trials = Trials()
# best = fmin(
#     objective123,
#     space=search_space,
#     algo=tpe.suggest,
#     max_evals=10,
#     trials=trials,
# )


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

prediction = next(predictor.predict(test_data))
print(prediction.mean)


fig, ax = plt.subplots()
actual = test[-prediction_length:].Close
ax.plot(actual, label="Actual")
preds = pd.Series(prediction.mean)
preds.index = test[-prediction_length:].index
ax.plot(preds, label="Prediction")
leg = ax.legend()

mse = mean_squared_error(test[-prediction_length:].Close, prediction.mean)
print(f"Mean Squared Error of Model: {mse}")
