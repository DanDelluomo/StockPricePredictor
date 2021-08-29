#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Featurize data and train XGBoost Model
"""

# Python standard library packages
from datetime import datetime
import glob
import os.path
import time

# External packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

today_date = datetime.today().strftime('%m-%d-%Y')

latest_row_clean = None
def create_features(df) -> pd.DataFrame:
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    del df['date']
    return df

def featurize(
    coin: str,
    cleaned_data: pd.DataFrame,
    scale: bool=False,
) -> (pd.DataFrame, pd.DataFrame):
    """ Featurize data
    """
    latest_row_clean = cleaned_data.copy()
    latest_row_clean['Yesterday_Volume'] = cleaned_data['Volume']
    latest_row_clean['Yesterday_Price'] = cleaned_data['Price']
    latest_row_clean['Yesterday_Percent_Change'] = cleaned_data['Percent_Change']
    latest_row_clean['30_Day_Moving_Average'] = latest_row_clean.loc[:, "Yesterday_Price"][::-1].rolling(window=30).mean()
    latest_row_clean['15_Day_Moving_Average'] = latest_row_clean.loc[:, "Yesterday_Price"][::-1].rolling(window=15).mean()
    latest_row_clean['10_Day_Moving_Average'] = latest_row_clean.loc[:, "Yesterday_Price"][::-1].rolling(window=10).mean()
    latest_row_clean['5_Day_Moving_Average'] = latest_row_clean.loc[:, "Yesterday_Price"][::-1].rolling(window=5).mean()
    latest_row_clean['5_Day_Volume_Average'] = latest_row_clean.loc[:, "Yesterday_Volume"][::-1].rolling(window=5).mean()


    # Generate Features
    cleaned_data['Yesterday_Volume'] = cleaned_data['Volume'].shift(-1)
    cleaned_data['Yesterday_Price'] = cleaned_data['Price'].shift(-1)
    cleaned_data['Yesterday_Percent_Change'] = cleaned_data['Percent_Change'].shift(-1)
    cleaned_data['30_Day_Moving_Average'] = cleaned_data.loc[:, "Yesterday_Price"][::-1].rolling(window=30).mean()
    cleaned_data['15_Day_Moving_Average'] = cleaned_data.loc[:, "Yesterday_Price"][::-1].rolling(window=15).mean()
    cleaned_data['10_Day_Moving_Average'] = cleaned_data.loc[:, "Yesterday_Price"][::-1].rolling(window=10).mean()
    cleaned_data['5_Day_Moving_Average'] = cleaned_data.loc[:, "Yesterday_Price"][::-1].rolling(window=5).mean()
    cleaned_data['5_Day_Volume_Average'] = cleaned_data.loc[:, "Yesterday_Volume"][::-1].rolling(window=5).mean()

    cleaned_data = cleaned_data.set_index('Date')
    print(cleaned_data)
    cleaned_data = cleaned_data[today_date:'2021-01-01']

    cleaned_data = cleaned_data.astype({
        '30_Day_Moving_Average': 'float32',
        '15_Day_Moving_Average': 'float32',
        '10_Day_Moving_Average': 'float32',
        '5_Day_Moving_Average': 'float32',
        '5_Day_Volume_Average': 'float32', 
    })

    # if scale:
    #     total_len = len(cleaned_data)
    #     training_set = cleaned_data['Price'].values
    #     sc = MinMaxScaler()
    #     training_set = np.reshape(training_set, (total_len, 1))
    #     training_set = sc.fit_transform(training_set)
    #     cleaned_data['Price'] = training_set.reshape(total_len)


    train_data = cleaned_data[
        [
            'Price', 
            'Yesterday_Price', 
            'Yesterday_Percent_Change', 
            'Yesterday_Volume', 
            '30_Day_Moving_Average', 
            '15_Day_Moving_Average',
            '10_Day_Moving_Average', 
            '5_Day_Moving_Average', 
            '5_Day_Volume_Average',
        ]
    ]

    boost_train_data = create_features(train_data)
    return boost_train_data, latest_row_clean


def train(
    coin: str,
    boost_train_data: pd.DataFrame,
    job_time: str,
    scale: bool=False,
    horizon: int=60,
) -> xgb.XGBRegressor:
    print(f"SCALE IS {scale}")
    top_price = boost_train_data['Price'].max()
    print(f"...Running XGBoost on {coin} data")
    sc = None
    if scale:
        total_len = len(boost_train_data)
        print("HERE")
        training_set = boost_train_data['Price'].values
        sc = MinMaxScaler()
        training_set = np.reshape(training_set, (total_len, 1))
        training_set = sc.fit_transform(training_set)
        boost_train_data['Price'] = training_set.reshape(total_len)

    boost_train_data = boost_train_data.reindex(index=boost_train_data.index[::-1])
    boost_test_data = boost_train_data[-horizon:]
    boost_train_data = boost_train_data[:-horizon]
    y_train = boost_train_data['Price']
    X_train = boost_train_data.drop(['Price'], axis=1)
    y_test = boost_test_data['Price']
    X_test = boost_test_data.drop(['Price'], axis=1)

    parameters = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8],
        'gamma': [0, 0.3, 1],
        'random_state': [42]
    }

    # Initialize XGB and GridSearch
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
    grid = GridSearchCV(xgb_reg, parameters)
    print("Doing grid search")
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    model = xgb.XGBRegressor(**grid.best_params_, objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_series = pd.Series(preds)
    preds_series.index = y_test.index

    # Save model 
    print(f'mean_squared_error = {mean_squared_error(y_test, preds_series)}')
    model_name = f"{coin}_model_{today_date}.json"
    model.save_model(model_name)

    # See how good the model would have done in the last 30 days
    bankroll = 0
    total_coin = 0
    lowest_bankroll = 0
    preds = model.predict(X_test)
    if scale:
        y_test = y_test * top_price
    for i in range(0, horizon):
        model = model
        current_row = X_test.iloc[[i]]
        print(current_row.index)
        print(f"date: {X_test.index[i]}")
        current_price = y_test[i]
        print(f"current_price: {current_price}")
        pred_price = preds[i]
        if scale:
            print("SCALED")
            pred_price = sc.inverse_transform([[pred_price]])[0][0]
        print("pred_price ", str(pred_price))
        difference = pred_price - current_price
        # print(difference)
        # print(f"Bankroll at {bankroll}")
        if bankroll < lowest_bankroll:
            lowest_bankroll = bankroll
        if difference > 0:
            # Buy
            bankroll -= 125
            print(f"Investing in {coin}")
            total_coin += ( 125 / (current_price) )
            # print(f"total_bitcoin = {total_bitcoin}")
        elif difference < 0:
            # Sell
            print(f"Withdrawing from {coin}")
            # print(f"And total_bitcoin is {total_bitcoin}")
            # print(f"And total_bitcoin_price is {total_bitcoin * current_price}")
            bankroll += (total_coin * current_price)
            total_coin = 0
        print("-" * 30)

    total_coin_price = total_coin * current_price

    print(f"Results:\n\tModel would have performed this well in the last {horizon} days \
    : {bankroll + total_coin_price}\n\t...with largest investment of {lowest_bankroll} ")

    model.fit(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test])
    )
    return model, sc
