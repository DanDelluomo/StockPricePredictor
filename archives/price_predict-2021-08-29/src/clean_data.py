#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Clean historical coin csv data
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb

def clean_data(
    coin: str,
    df: pd.DataFrame,
) -> pd.DataFrame:

    """ 
    Clean historical coin csv data

    Arguments:
        df: pandas DataFrame from downloads folder
        coin_ticker: investing.com url coin abbreviation
        address: investing.com URL of historical data for coin
    Returns:
        pandas DataFrame (cleaned)
    """

    # Clean latest DataFrame
    print(f"...Cleaning latest {coin} data")
    clean_prices = df
    clean_prices.rename(columns={
        "Vol.": "Volume",
        "Change %": "Percent_Change",
    }, inplace=True)
    clean_prices['Price'] = clean_prices['Price'].replace(',','', regex=True)
    clean_prices['Price'] = pd.to_numeric(clean_prices['Price'])

    clean_prices['Percent_Change'] = clean_prices['Percent_Change'].apply(
        lambda x: np.float(x[:-1]) 
    )

    # Convert volume to integer
    # See all letters
    letters = []
    for i in clean_prices['Volume'].unique():
        last_letter = i[-1]
        if last_letter not in letters:
            letters.append(last_letter)

    clean_prices = clean_prices[~clean_prices["Volume"].str.contains('-', regex=False)]
    clean_prices['Volume'] = clean_prices['Volume'].apply(
        lambda x: 
            ( ( pd.to_numeric(x[:-1]) ) * 1000000 ) if ("M" in x) 
            else ( ( pd.to_numeric(x[:-1]) ) * 1000 ) 
    )

    clean_prices = clean_prices.astype({
        'Date': 'datetime64',
        'Price': 'float32', 
        'Volume': 'float32',
        'Percent_Change': 'float32', 
    })

    return clean_prices