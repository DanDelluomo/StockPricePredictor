#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Gemini buy and sell functions.
"""

# Python Standard Library
import requests
import pandas as pd
import hmac
import json
import base64
import hashlib
import datetime, time

# Project modules
from credentials import credentials


def gemini_buy_function(
    coin: str,
) -> None:
    print(f"...Starting Gemini Buy function for {coin}")
    gemini_url = master_dict['coin']['gemini_url']
    response = requests.get(gemini_url).json()
    coin_price = float(response['ask'])
    print(bitcoin_price)

    base_url = "https://api.gemini.com/v1"
    response = requests.get(base_url + f"/book/{coin}usd")
    btc_book = response.json()
    btc_book
    t = datetime.datetime.now()
    payload_nonce =  str(int(time.mktime(t.timetuple())*1000))

    amount = 125 / bitcoin_price
    payload = {
    "request": "/v1/order/new",
        "nonce": payload_nonce,
        "symbol": f"{coin}usd",
        "amount": amount,
        "price": coin_price,
        "side": "buy",
        "type": "exchange limit",
    }

    encoded_payload = json.dumps(payload).encode()
    b64 = base64.b64encode(encoded_payload)
    signature = hmac.new(
        credentials['gemini_api_secret'], b64, hashlib.sha384
        ).hexdigest()

    base_url = "https://api.gemini.com"
    endpoint = "/v1/order/new"
    url = base_url + endpoint

    def Auth(payload):   
        encoded_payload = json.dumps(payload).encode()
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(
            credentials['gemini_api_secret'], 
            b64, 
            hashlib.sha384
        ).hexdigest()

        request_headers = { 
            'Content-Type': "text/plain",
            'Content-Length': "0",
            'X-GEMINI-APIKEY': credentials['gemini_api_key'],
            'X-GEMINI-PAYLOAD': b64,
            'X-GEMINI-SIGNATURE': signature,
            'Cache-Control': "no-cache"
        }
        
        return(request_headers)


    try:
        new_order = requests.post(url, data=None, headers=Auth(payload)).json()
        print(new_order)
    except Exception as e:
        print(f'Error placing buy order: {e}')



def gemini_sell_function(
    coin:str,
) -> None:
    print(f"...Starting Gemini sell function for {coin}")
    # Get Amount of Bitcoin in account
    t = datetime.datetime.now()
    payload_nonce =  str(int(time.mktime(t.timetuple())*1000))
    def Auth(payload):   
        encoded_payload = json.dumps(payload).encode()
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

        request_headers = { 
            'Content-Type': "text/plain",
            'Content-Length': "0",
            'X-GEMINI-APIKEY': credentials['gemini_api_key'],
            'X-GEMINI-PAYLOAD': b64,
            'X-GEMINI-SIGNATURE': signature,
            'Cache-Control': "no-cache" 
        }
        return(request_headers)


    payload = {
        "nonce": payload_nonce,
        "request": "/v1/balances",
    }

    url = "https://api.gemini.com/v1/balances"
    response = requests.post(
        url,
        data=None,
        headers=Auth(payload),
    ).json()

    coin_holdings = 0
    for dictionary in response:
        if dictionary['currency'] == coin.upper():
            coin_holdings = float(dictionary['availableForWithdrawal'])

    if coin_holdings > 0:
        time.sleep(1)
        t = datetime.datetime.now()
        payload_nonce =  str(int(time.mktime(t.timetuple())*1000))
        response = requests.get(f"https://api.gemini.com/v1/pubticker/{coin.upper()}USD").json()
        coin_price = float(response['ask'])
        payload = {
            "request": "/v1/order/new",
            "nonce": payload_nonce,
            "symbol": "btcusd",
            "amount": coin_holdings,
            "price": coin_price,
            "side": "sell",
            "type": "exchange limit",
        }
        
        base_url = "https://api.gemini.com"
        endpoint = "/v1/order/new"
        url = base_url + endpoint
        try:
            new_order = requests.post(
                url,
                data=None,
                headers=Auth(payload)
            ).json()
            print(new_order)

        except Exception as e:
            print(f'Error placing sell order: {e}')
    else:
        print(f"No {coin} to sell. How can you short it?")
    

