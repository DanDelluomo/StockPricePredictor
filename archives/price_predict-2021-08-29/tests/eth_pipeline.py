from datetime import datetime
import glob
import os.path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb

from time import sleep
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os
path = '/Users/dan/Downloads/'


today_date = datetime.today().strftime('%m-%d-%Y')


print("...Cleaning latest Eth data")
# Clean latest data
clean_prices = pd.read_csv('Ethereum_Historical_Data_2021-08-22_20:53:58.csv')

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

# Make sure all the data will be available on the given new day

clean_prices['Yesterday_Volume'] = clean_prices['Volume'].shift(-1)
clean_prices['Yesterday_Price'] = clean_prices['Price'].shift(-1)
clean_prices['Yesterday_Percent_Change'] = clean_prices['Percent_Change'].shift(-1)
clean_prices['30_Day_Moving_Average'] = clean_prices.loc[:, "Yesterday_Price"][::-1].rolling(window=30).mean()
clean_prices['15_Day_Moving_Average'] = clean_prices.loc[:, "Yesterday_Price"][::-1].rolling(window=15).mean()
clean_prices['10_Day_Moving_Average'] = clean_prices.loc[:, "Yesterday_Price"][::-1].rolling(window=10).mean()
clean_prices['5_Day_Moving_Average'] = clean_prices.loc[:, "Yesterday_Price"][::-1].rolling(window=5).mean()
clean_prices['5_Day_Volume_Average'] = clean_prices.loc[:, "Yesterday_Volume"][::-1].rolling(window=5).mean()

clean_prices = clean_prices.set_index('Date')
clean_prices = clean_prices[today_date:'2021-05-01']

clean_prices = clean_prices.astype({
    '30_Day_Moving_Average': 'float32',
    '15_Day_Moving_Average': 'float32',
    '10_Day_Moving_Average': 'float32',
    '5_Day_Moving_Average': 'float32',
    '5_Day_Volume_Average': 'float32', 
})

top_price = clean_prices['Yesterday_Price'].max()

train_data = clean_prices[
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

def create_features(df):
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

print("...Running XGBoost on Eth data")
boost_train_data = create_features(train_data)
boost_train_data = boost_train_data.reindex(index=boost_train_data.index[::-1])
boost_test_data = boost_train_data[-125:]
boost_train_data = boost_train_data[:-125]
y_train = boost_train_data['Price']
X_train = boost_train_data.drop(['Price'], axis=1)
y_test = boost_test_data['Price']
X_test = boost_test_data.drop(['Price'], axis=1)

parameters = {
    'n_estimators': [200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10, 12, 14, 16, 18],
    'gamma': [0.01, 0.05, 0.1],
    'random_state': [42]
}

# Initialize XGB and GridSearch
xgb_reg = xgb.XGBRegressor(nthread=-1, objective='reg:squarederror')
grid = GridSearchCV(xgb_reg, parameters)
print("Doing grid search")
grid.fit(X_train, y_train)
print(grid.best_params_)
model = xgb.XGBRegressor(**grid.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds_series = pd.Series(preds)
preds_series.index = y_test.index
y_test.plot()
preds_series.plot()

# Save model 
print(f'mean_squared_error = {mean_squared_error(y_test, preds_series)}')

xgb.plot_importance(model)

bankroll = 0
total_coin = 0
lowest_bankroll = 0
for i in range(0, 125):
    current_row = X_test.iloc[[i]]
    current_price = current_row['Yesterday_Price'][0]
    # print(f"current_price: {current_price}")
    pred_price = model.predict(current_row)[0] * top_price
    # print(f"pred_price: {pred_price}")
    # print("-" * 30)
    difference = pred_price - current_price
    # print(difference)
    # print(f"Bankroll at {bankroll}")
    if bankroll < lowest_bankroll:
        lowest_bankroll = bankroll
    if difference > 0:
        # Buy
        bankroll -= 125
        # print("Investing in Bitcoin")
        total_coin += ( 125 / (current_price) )
        # print(f"total_bitcoin = {total_bitcoin}")
        next_price = y_test[i]
    elif difference < 0:
        # Sell
        # print("Withdrawing from Bitcoin")
        # print(f"And total_bitcoin is {total_bitcoin}")
        # print(f"And total_bitcoin_price is {total_bitcoin * current_price}")
        bankroll += (total_coin * current_price)
        total_coin = 0

# # print(f"Results:\n\tBankroll: {bankroll}\n\tMoney in Bitcoin: {money_in_bitcoin}")
# print(f"Results:\n\tModel would have performed this well in the last 125 days \
# : {bankroll + total_coin}\n\t...with largest investment of {lowest_bankroll} "
# )

# # Investment Decision
# today_price = y_test[[-1]][0]
# predicted_tomorrow_price = model.predict(X_test.iloc[[-1]])[0]
# print(f"Price today: {today_price}\n\
# Predicted price tomorrow: {predicted_tomorrow_price}")

# difference = predicted_tomorrow_price - today_price
# print(difference)

# def gemini_buy_function() -> None:
#     import requests
#     import pandas as pd
#     import hmac
#     import json
#     import base64
#     import hashlib
#     import datetime, time
#     gemini_api_key = "account-jQlGTGUecZeEXFVJVtRM"
#     gemini_api_secret = b"2Pg4uZeYLys77pjsH7CEBWZ45ZdS"

#     response = requests.get("https://api.gemini.com/v1/pubticker/BTCUSD").json()
#     bitcoin_price = float(response['ask'])
#     print(bitcoin_price)

#     base_url = "https://api.gemini.com/v1"
#     response = requests.get(base_url + "/book/btcusd")
#     btc_book = response.json()
#     btc_book
#     t = datetime.datetime.now()
#     payload_nonce =  str(int(time.mktime(t.timetuple())*1000))

#     amount = 125 / bitcoin_price
#     payload = {
#     "request": "/v1/order/new",
#         "nonce": payload_nonce,
#         "symbol": "btcusd",
#         "amount": amount,
#         "price": bitcoin_price,
#         "side": "buy",
#         "type": "exchange limit",
#     }

#     encoded_payload = json.dumps(payload).encode()
#     b64 = base64.b64encode(encoded_payload)
#     signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

#     base_url = "https://api.gemini.com"
#     endpoint = "/v1/order/new"
#     url = base_url + endpoint


#     def Auth(payload):   
#         encoded_payload = json.dumps(payload).encode()
#         b64 = base64.b64encode(encoded_payload)
#         signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

#         request_headers = { 
#             'Content-Type': "text/plain",
#             'Content-Length': "0",
#             'X-GEMINI-APIKEY': gemini_api_key,
#             'X-GEMINI-PAYLOAD': b64,
#             'X-GEMINI-SIGNATURE': signature,
#             'Cache-Control': "no-cache"
#         }
        
#         return(request_headers)


#     try:
#         new_order = requests.post(url, data=None, headers=Auth(payload)).json()
#         print(new_order)
#     except Exception as e:
#         print(f'Error placing buy order: {e}')



# def gemini_sell_function() -> None:
#     # Get Amount of Bitcoin in account

#     import requests
#     import pandas as pd
#     import hmac
#     import json
#     import base64
#     import hashlib
#     import datetime, time
#     gemini_api_key = "account-jQlGTGUecZeEXFVJVtRM"
#     gemini_api_secret = b"2Pg4uZeYLys77pjsH7CEBWZ45ZdS"

#     t = datetime.datetime.now()
#     payload_nonce =  str(int(time.mktime(t.timetuple())*1000))
#     def Auth(payload):   
#         encoded_payload = json.dumps(payload).encode()
#         b64 = base64.b64encode(encoded_payload)
#         signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

#         request_headers = { 
#             'Content-Type': "text/plain",
#             'Content-Length': "0",
#             'X-GEMINI-APIKEY': gemini_api_key,
#             'X-GEMINI-PAYLOAD': b64,
#             'X-GEMINI-SIGNATURE': signature,
#             'Cache-Control': "no-cache" 
#         }
#         return(request_headers)


#     payload = {
#         "nonce": payload_nonce,
#         "request": "/v1/balances",
#     }

#     url = "https://api.gemini.com/v1/balances"
#     response = requests.post(
#         url,
#         data=None,
#         headers=Auth(payload),
#     ).json()

#     bitcoin_holdings = 0
#     for dictionary in response:
#         if dictionary['currency'] == 'BTC':
#             bitcoin_holdings = float(dictionary['availableForWithdrawal'])

#     if bitcoin_holdings > 0:
#         time.sleep(1)
#         t = datetime.datetime.now()
#         payload_nonce =  str(int(time.mktime(t.timetuple())*1000))
#         response = requests.get("https://api.gemini.com/v1/pubticker/BTCUSD").json()
#         bitcoin_price = float(response['ask'])
#         payload = {
#             "request": "/v1/order/new",
#             "nonce": payload_nonce,
#             "symbol": "btcusd",
#             "amount": bitcoin_holdings,
#             "price": bitcoin_price,
#             "side": "sell",
#     #         "quantity": str(bitcoin_holdings),
#             "type": "exchange limit",
#         }
        
#         base_url = "https://api.gemini.com"
#         endpoint = "/v1/order/new"
#         url = base_url + endpoint
#         try:
#             new_order = requests.post(
#                 url,
#                 data=None,
#                 headers=Auth(payload)
#             ).json()
#             print(new_order)

#         except Exception as e:
#             print(f'Error placing sell order: {e}')
#     else:
#         print("No Bitcoin to sell. How can you short it?")
    

# if difference > 0:
#     gemini_buy_function()

# elif difference < 0:
#     gemini_sell_function()