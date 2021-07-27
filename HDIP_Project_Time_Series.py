"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency
"""

# Downloading necessary Packages
import numpy as np
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


df.index = pd.to_datetime(df['Date'])
y.index = pd.to_datetime(y['Date'])


df.index = pd.to_datetime(df.index)
y.index = pd.to_datetime(y.index)
df = df.asfreq('D')
y = y.asfreq('D')


# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
# =============================================================================
from HDIP_Project_Functions import *

test_stationarity(y['Close'])

test_stationarity(y['Close Percentage Change'])

test_stationarity(y['diff'])

test_stationarity(y['log_Close'])

test_stationarity(y['sqrt_Close'])




KPSS_test(y['Close'])

KPSS_test(y['Close Percentage Change'])

KPSS_test(y['diff'])

KPSS_test(y['log_Close'])

KPSS_test(y['sqrt_Close'])


        
adfuller_test(y['Close'])

adfuller_test(y['Close Percentage Change'])

adfuller_test(y['diff'])

adfuller_test(y['log_Close'])

adfuller_test(y['sqrt_Close'])


# =============================================================================
# 
# =============================================================================
def dealing_with_NAN(column):
    # checking for missing data


    print('Nan in each columns' , column.isna().sum(), sep='\n')
    # Backward Fill 
    df_bfill = column.bfill()

    print('Nan in each columns after Backfill ' , df_bfill.isna().sum(), sep='\n')
    
#    # Forward Fill    
#    df_ffill = y['Close'].ffill()

dealing_with_NAN(df['Close'])
# =============================================================================
# Detrending the Time Series
# =============================================================================

# Using scipy: Subtract the line of best fit
from scipy import signal
detrended = signal.detrend(df_bfill.values)
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the least squares fit', fontsize=16)
plt.show()


# Using statmodels: Subtracting the Trend Component.
from statsmodels.tsa.seasonal import seasonal_decompose
result_mul = seasonal_decompose(df_bfill, model='multiplicative', extrapolate_trend='freq')
detrended = df_bfill.values - result_mul.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
plt.show()






# =============================================================================
# Decomposition
# =============================================================================
def decomposition_plot():    
    # Viewing the seasonal decompose of the target variable
    rcParams['figure.figsize'] = 8, 8
    decomposition = sm.tsa.seasonal_decompose(df_bfill.asfreq('D'), )
    fig = decomposition.plot()
    plt.show()    

decomposition_plot()




############# Trying get PLOTLY #############
# https://github.com/Pierian-Data/AutoArima-Time-Series-Blog/blob/master/Forecasting%20a%20Time%20Series%20in%20Python.ipynb
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(y['Close'], model='multiplicative')
fig = result.plot()
plot_mpl(fig)








from pyramid.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)



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
