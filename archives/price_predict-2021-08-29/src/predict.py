import datetime
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from train_model import create_features

def predict(
    model: xgb.XGBRegressor,
    latest_row_clean: pd.DataFrame,
    today_date: str,
    sc=None,
) -> (float, float):

    """ 
    Predict the price tomorrow based on tomorrow's DataFrame row.

    Arguments:
        model: XGBoost Regressor model fit on train + test data
    Returns:
        float: model's prediction for the coin's price tomorrow
    """
    today_price = latest_row_clean['Price'][0]
    # print(today_price)
    date_tomorrow =  pd.to_datetime(today_date) + pd.DateOffset(1)
    index = pd.date_range(date_tomorrow-datetime.timedelta(1), periods=1, freq='D')
    tomorrow = pd.DataFrame(index=index)
    # tomorrow = df_.fillna(0) # with 0s rather than NaNs
    date_tomorrow =  pd.to_datetime(today_date) + pd.DateOffset(1)
    tomorrow['Date'] = date_tomorrow
    tomorrow = tomorrow.set_index('Date')
    tomorrow = create_features(tomorrow)
    tomorrow['Yesterday_Price'] = today_price
    tomorrow['Yesterday_Volume'] = latest_row_clean.iloc[[0]]['Volume'][0]
    tomorrow['Yesterday_Percent_Change'] = latest_row_clean.iloc[[0]]['Percent_Change'][0]

    tomorrow['30_Day_Moving_Average'] = latest_row_clean.iloc[[0]]['30_Day_Moving_Average'][0]
    tomorrow['15_Day_Moving_Average'] = latest_row_clean.iloc[[0]]['15_Day_Moving_Average'][0]
    tomorrow['10_Day_Moving_Average'] = latest_row_clean.iloc[[0]]['10_Day_Moving_Average'][0]
    tomorrow['5_Day_Moving_Average'] = latest_row_clean.iloc[[0]]['5_Day_Moving_Average'][0]
    tomorrow['5_Day_Volume_Average'] = latest_row_clean.iloc[[0]]['15_Day_Moving_Average'][0]

    tomorrow = tomorrow [
        [
            'Yesterday_Price', 
            'Yesterday_Percent_Change', 
            'Yesterday_Volume',
            '30_Day_Moving_Average', 
            '15_Day_Moving_Average',
            '10_Day_Moving_Average', 
            '5_Day_Moving_Average', 
            '5_Day_Volume_Average',
            'dayofweek', 
            'quarter', 
            'month', 
            'year', 
            'dayofyear', 
            'dayofmonth',
            'weekofyear',
        ]
    ]

    # print(tomorrow)
    pred_price = model.predict(tomorrow)[0]
    print("PRED_PRICE ORIG", pred_price)
    # if sc:
    #     pred_price = sc.inverse_transform([[pred_price]])[0][0]
    print("PRED_PRICE ", pred_price)
    return pred_price, today_price

