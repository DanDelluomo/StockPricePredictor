#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Orchestrate master pipeline
"""

# Python standard library packages
from datetime import datetime
import os
import os.path
import time
from time import sleep

# External packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Project packages
from clean_data import clean_data
from gemini import gemini_buy_function, gemini_sell_function
from get_data import download_coin_history, retrieve_and_archive
from predict import predict
from train_model import featurize, train


master_dict = {
    "btc": {
        "name": "Bitcoin",
        "data_address": "https://www.investing.com/indices/investing.com-btc-usd-historical-data",
        "gemini_url": "https://api.gemini.com/v1/pubticker/BTCUSD",
        "best_xgboost_params": {
            'gamma': 0.3, 
            'learning_rate': 0.1, 
            'max_depth': 6, 
            'n_estimators': 200, 
            'random_state': 42
        },
        "scale": False
    },

    "eth": {
        "name": "Ethereum",
        "data_address": "https://www.investing.com/indices/investing.com-eth-usd-historical-data",
        "gemini_url": "https://api.gemini.com/v1/pubticker/ETHUSD",
        "best_xgboost_params": {
            'gamma': 0.3, 
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'n_estimators': 200, 
            'random_state': 42
            },
        "scale": True,
    }
}

job_time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
today_date = datetime.today().strftime('%m-%d-%Y')

for coin in master_dict:
    if coin == 'btc':
        continue

    # print(f"...Downloading data for {coin}.")
    # download_coin_history(
    #     master_dict[coin]['name'],
    #     coin,
    #     master_dict[coin]['data_address'],
    # )

    # print(f"...Retrieving and archiving data for {coin}.")
    # coin_df = retrieve_and_archive(
    #     master_dict[coin]['name'],
    #     coin,
    #     job_time,
    # )
    coin_df = pd.read_csv('Ethereum_Historical_Data_2021-08-23_08:14:14.csv')

    

    print(f"...Cleaning data for {coin}.")
    cleaned_data = clean_data(
        master_dict[coin]['name'],
        coin_df,
    )

    print(f"...Generating features for {coin}.")
    featured_data, latest_row_clean = featurize(
        master_dict[coin]['name'],
        cleaned_data,
        master_dict[coin]['scale'],
    )
    print("LATEST ROW CLEAN", latest_row_clean)

    print(f"...Training XGBoost on {coin} data.")
    model, sc = train(
        master_dict[coin]['name'],
        featured_data,
        job_time,
        master_dict[coin]['scale'],
        horizon=60
    )

    print(f"...Generating tomorrow price prediction for {coin}.")
    prediction, today_price = predict(
        model,
        latest_row_clean,
        today_date,
        sc=sc,
    )
    print(f"price_today: {today_price}\nprediction tomorrow: {prediction}")
    difference = prediction - today_price
    print("...Executing Gemini code")
    if difference > 0:
        gemini_buy_function(coin)
    elif difference < 0:
        gemini_sell_function(coin)
    
    print("PIPELINE COMPLETE")