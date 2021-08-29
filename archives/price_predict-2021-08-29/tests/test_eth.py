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
print(f"...Running Ethereum Gemini pipeline at {today_date}")

print("...Grabbing latest Ethereum data from investing.com")
# Grab the latest Ethereum historical price CSV from investing.com
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-notifications")
driver = webdriver.Chrome(ChromeDriverManager().install(),
                 chrome_options=chrome_options)

driver.get("https://www.investing.com/indices/investing.com-eth-usd-historical-data")
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
start_date.send_keys("01/01/2021")
apply_dates_button = driver.find_element_by_css_selector('#applyBtn')
apply_dates_button.click()
download_button = driver.find_element_by_css_selector('.downloadDataWrap a.downloadBlueIcon')
download_button.click()

username = "dandelluomo@gmail.com"
password = "Math9147"
username_field = driver.find_element_by_css_selector("#loginFormUser_email")
password_field = driver.find_element_by_css_selector("#loginForm_password")
username_field.send_keys(username)
password_field.send_keys(password)
sign_in = driver.find_element_by_css_selector("#signup a.newButton.orange")
sign_in.click()
time.sleep(3)
button = driver.find_element_by_id('widgetFieldDateRange')
button.click()
time.sleep(3)
start_date = driver.find_element_by_css_selector(".ui-datepicker-group input#startDate")
start_date.clear()
# driver.find_element_by_css_selector(".ui-datepicker-group input#endDate")
start_date.send_keys("01/01/2021")
# start_date.send_keys(Keys.RETURN)
time.sleep(3)
apply_dates_button = driver.find_element_by_css_selector('#applyBtn')
apply_dates_button.click()
download_button = WebDriverWait(driver, 5).until(lambda x: x.find_element(By.CSS_SELECTOR, ".downloadDataWrap a.downloadBlueIcon"))
time.sleep(3)
download_button.click()
time.sleep(3)
driver.close()

