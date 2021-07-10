"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency
"""

# Downloading necessary files
import numpy as np
# from numpy import log
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import mplfinance as mpf 
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib import pyplot
import datetime
from datetime import datetime

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
import plotly.graph_objects as go

import statsmodels.api as sm
from pylab import rcParams

# importing functions from the file
from HDIP_Project_Functions import *


# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'


# =============================================================================
# Scraping the Top 25 Cryptocurrencies off Yahoo Finance
# =============================================================================

#from HDIP_Project_Scaping_JSON import *
#
## Run live Website
#def main():
#    global df_cryptolist
#    contents = get_page_contents()
#    soup = convert_to_soup(contents)
#    get_title(soup)
#    get_pattern(soup)
#    print(df_cryptolist.head(1).transpose())
#    
#if __name__ == '__main__':
#    main()   

# =============================================================================
# creating a list from the crypto-table
# =============================================================================      
get_crypto_df(df_cryptolist)
     
# ============================================================================
# Asking the user for an input   
# ============================================================================
please_choose_crypto()

from HDIP_Project_Functions import crypto_name, insert

# =============================================================================
# Collecting info from Yahoo Finance and creating a dataset for that cryptocurrency
# =============================================================================
create_df_for_crypto(insert)

from HDIP_Project_Functions import *

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
create_graphs()

create_candlestick()

# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================
#create_hist_and_box(y['close_pct_change'])

#create_hist_and_box_pct_change()
#
#logged_create_hist_and_box_pct_change()
#
#
#
## =============================================================================
## Decomposition
## =============================================================================
##decomposition_plot()
#
## =============================================================================
## Dickey-Fuller Test
## =============================================================================
##adfuller_test(df['Close'])
#
## =============================================================================
## Creating a plot with analysis and rolling mean and standard deviation
## =============================================================================
#test_stationarity(y['Close'])
#
#
#test_stationarity(y['Close Percentage Change'])
#
##test_stationarity(y['diff'])
#
##test_stationarity(y['log_Close'])
##
##test_stationarity(y['sqrt_Close'])
#
## =============================================================================
## ACF and PACF plots
## =============================================================================
#
#
## =============================================================================
## Exploring the difference
## =============================================================================    
##plotting the graph for ['diff'] variable
#diff_plot(y['diff'])
#
#create_diff_volume(y['diff'])
#
## checking the adfuller of the ['diff'] variable
##adfuller_test(y['diff'])
#
## =============================================================================
## Exploring the logged data
## =============================================================================
## creating a graph with histogram and lineplot of logged data
#log_create_hist_and_line(y['log_Close'])
#
## creating an analysis and adfuller-dickey test
##test_stationarity(y['log_Close'])
#
#
## =============================================================================
## Splitting the data in Training and Test Data
## =============================================================================
## splitting the data 
#create_train_and_test()
#
## creating a plot for the training and test set
#training_and_test_plot()
