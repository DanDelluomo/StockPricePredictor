#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download historical coin csv data from investing.com using Selenium.
"""

# Python standard library packages
from datetime import datetime
import glob
import os
import os.path
import time
from time import sleep

# External packages
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Project modules
from credentials import credentials

def download_coin_history(
    coin: str,
    coin_ticker: str,
    address: str,
) -> None:

    """ 
    Download historical coin csv data from investing.com using Selenium.

    Arguments:
        coin: full, formal name of cryptocurrency
        coin_ticker: investing.com url coin abbreviation
        address: investing.com URL of historical data for coin
    Returns:
        None
    """

    path = '/Users/dan/Downloads/'
    # Grab the latest Ethereum historical price CSV from investing.com
    print("...Grabbing {coin} data from investing.com")
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-notifications")
    
    # The next two options were attempts to do it in EC2
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(800, 800))  
    display.start()

    driver = webdriver.Chrome(ChromeDriverManager().install(),
                    chrome_options=chrome_options)
    driver.get(f"https://www.investing.com/indices/investing.com-{coin_ticker}-usd-historical-data")
    popup = WebDriverWait(driver, 5).until(lambda x: x.find_element(By.ID, "PromoteSignUpPopUp"))
    driver.execute_script("""
    let element = document.querySelector('#PromoteSignUpPopUp');
    if (element){
        element.parentNode.removeChild(element);
    }
    """, popup)

    all_iframes = driver.find_elements_by_tag_name("iframe")
    if len(all_iframes) > 0:
        print("Ad Found\n")
        driver.execute_script("""
            var elems = document.getElementsByTagName("iframe"); 
            for(var i = 0, max = elems.length; i < max; i++)
                {
                    elems[i].hidden=true;
                }
                            """)
        print('Total Ads: ' + str(len(all_iframes)))
    else:
        print('No frames found')
        
    driver.execute_script("""
        document.querySelector('.js-general-overlay').style.display = "none"
    """)

    button = driver.find_element_by_id('widgetFieldDateRange')
    button.click()

    start_date = driver.find_element_by_css_selector(".ui-datepicker-group input#startDate")
    start_date.clear()
    # Get all historical data until the current day
    start_date.send_keys("01/01/2000")
    apply_dates_button = driver.find_element_by_css_selector('#applyBtn')
    apply_dates_button.click()
    download_button = driver.find_element_by_css_selector('.downloadDataWrap a.downloadBlueIcon')
    download_button.click()


    username_field = driver.find_element_by_css_selector("#loginFormUser_email")
    password_field = driver.find_element_by_css_selector("#loginForm_password")
    username_field.send_keys(credentials['username'])
    password_field.send_keys(credentials['password'])
    sign_in = driver.find_element_by_css_selector("#signup a.newButton.orange")
    sign_in.click()
    time.sleep(3)
    button = driver.find_element_by_id('widgetFieldDateRange')
    button.click()
    time.sleep(3)
    start_date = driver.find_element_by_css_selector(".ui-datepicker-group input#startDate")
    start_date.clear()
    # driver.find_element_by_css_selector(".ui-datepicker-group input#endDate")
    start_date.send_keys("01/01/2000")
    # start_date.send_keys(Keys.RETURN)
    time.sleep(3)
    apply_dates_button = driver.find_element_by_css_selector('#applyBtn')
    apply_dates_button.click()
    download_button = WebDriverWait(driver, 5).until(lambda x: x.find_element(By.CSS_SELECTOR, ".downloadDataWrap a.downloadBlueIcon"))
    time.sleep(3)
    download_button.click()
    time.sleep(3)
    driver.close()

def retrieve_and_archive(
    coin: str,
    coin_ticker: str,
    job_time: str,
) -> pd.DataFrame:
    """ 
        Download historical coin csv data from investing.com using Selenium.

        Arguments:
            coin: full, formal name of cryptocurrency
            coin_ticker: investing.com url coin abbreviation
            job_time: timestamp of current pipeline
        Returns:
            pandas DataFrame
        """
    

    # Archive yesterday's historical coin csv data
    print(f"...Archiving yesterday's {coin} csv data")
    archive_path = f"/Users/dan/projects/price_predict/data/cryptos/historical_csvs/{coin}/"
    folder_path = '/Users/dan/projects/price_predict/data/cryptos/'
    file_type = '*.csv'
    files = glob.glob(folder_path + file_type)
    files = [i for i in files if coin in i]
    for i in files:
        name = i.split('/')[-1]
        archived_path = archive_path + name
        os.rename(i, archived_path)
    
    # Archive yesterday's historical coin json data
    print(f"...Archiving yesterday's {coin} json data")
    archive_path = f"/Users/dan/projects/price_predict/data/cryptos/historical_models/{coin}/"
    folder_path = '/Users/dan/projects/price_predict/data/cryptos/'
    file_type = '*.json'
    files = glob.glob(folder_path + file_type)
    files = [i for i in files if coin in i]
    for i in files:
        name = i.split('/')[-1]
        archived_path = archive_path + name
        os.rename(i, archived_path)

    # Retrieve latest csv download from Downloads folder
    folder_path = '/Users/dan/Downloads/'
    file_type = '*.csv'
    files = glob.glob(folder_path + file_type)
    latest_csv_download = max(files, key=os.path.getctime)
    assert len(pd.read_csv(latest_csv_download)) > 200
    assert coin in latest_csv_download
    new_path = f"/Users/dan/projects/price_predict/data/cryptos/{coin}_Historical_Data_{job_time}.csv"
    os.rename(latest_csv_download, new_path)
    return pd.read_csv(new_path)