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

# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'

# =============================================================================
# Read in file for ease of use
# =============================================================================

# read the CSV file
df_cryptolist = pd.read_csv('df_cryptolist.csv')

# read the CSV file
df = pd.read_csv('df.csv')

# read the CSV file
y = pd.read_csv('y.csv')

crypto_name = 'Bitcoin'

insert = 'BTC-USD'



# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
# =============================================================================
from HDIP_Project_Functions import *

test_stationarity(y['Close'])

test_stationarity(y['Close Percentage Change'])

test_stationarity(y['diff'])

test_stationarity(y['log_Close'])

test_stationarity(y['sqrt_Close'])


        
adfuller_test(y['Close'])

adfuller_test(y['Close Percentage Change'])

adfuller_test(y['diff'])

adfuller_test(y['log_Close'])

adfuller_test(y['sqrt_Close'])


# =============================================================================
# 
# =============================================================================

from statsmodels.tsa.arima_model import ARMA


# A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
df.index = pd.DatetimeIndex(df.index).to_period('D')
y.index = pd.DatetimeIndex(y.index).to_period('D')

# Instantiate the model
model = ARMA(y['diff'], order = (2,1))

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

## Generate predictions
#one_step_forecast = results.get_prediction(start=-30)
#
## Extract prediction mean
#mean_forecast = one_step_forecast.predicted_mean
#
## Get confidence intervals of  predictions
#confidence_intervals = one_step_forecast.conf_int()
#
## Select lower and upper confidence limits
#lower_limits = confidence_intervals.loc[:,'lower close']
#upper_limits = confidence_intervals.loc[:,'upper close']
#
## Print best estimate  predictions
#print(mean_forecast)
