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
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf, adfuller, kpss
from statsmodels.tsa.arima_model import ARMA
import plotly.graph_objects as go
import statsmodels.api as sm
from pylab import rcParams
import statsmodels.api as sm


from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# display plotly in browser when run in Spyder
pio.renderers.default = 'browser'

# =============================================================================
# Read in file for ease of use
# =============================================================================

# read the CSV file
df_cryptolist = pd.read_csv('df_cryptolist.csv')

# read the CSV file
df = pd.read_csv('df.csv', parse_dates=['Date'], index_col='Date')

# read the CSV file
y = pd.read_csv('y.csv', parse_dates=['Date'], index_col='Date')

crypto_name = 'Bitcoin'

insert = 'BTC-USD'

# =============================================================================
# Fixing issues with frequency
# =============================================================================

df = df.asfreq('D')
print('Nan in each columns' , df.isna().sum())
df = df.ffill()
print('Nan in each columns' , df.isna().sum())


y = y.asfreq('D')
print('Nan in each columns' , y.isna().sum())
y = y.ffill()
print('Nan in each columns' , y.isna().sum())

# =============================================================================
# New Variables
# =============================================================================
# Create daily_return
df['daily_return'] = df['Close'].pct_change(periods=1)

# Create monthly_return
df['monthly_return'] = df['Close'].pct_change(periods=30)

# Create annual_return
df['annual_return'] = df['Close'].pct_change(periods=365)

# Create 'lagged' and 'shifted'
df['lagged'] =  df['Close'].shift(periods=-90)
df['shifted'] = df['Close'].shift(periods=90)

# Create Month and Year variable from index
df['month_year'] = pd.to_datetime(df.index).to_period('M')


# =============================================================================
# Examing returns with boxplots
# =============================================================================

def boxplots(x, y):
    fig, ax = plt.subplots(figsize=(15,6))
    sns.boxplot(x, y, ax=ax)
    ax.set_xticklabels(x.unique(),  rotation = 45, fontsize = 12)
    plt.show()

# Examining daily returns in each year
boxplots(df.daily_return.index.year, df.daily_return)

# Examining daily returns in each month year
boxplots(df['month_year'], df['daily_return'])

# Examining monthly returns in each month year
boxplots(df['month_year'], df['monthly_return'])

# =============================================================================
# creating important functions
# =============================================================================
def normalise():
    # Select first prices
    first_price = df['Close'].iloc[0]
    # Create normalized
    normalized = df['Close'].div(first_price)
    # Plot normalized
    normalized.plot()
    plt.show()


# Dickey Fuller Test
def adfuller_test(data):
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print('============================================================')
    print('Results of Dickey-Fuller Test for {}:'.format(crypto_name))
    print('============================================================')
    print (dfoutput)
    if dftest[1]>0.05:
        print('Conclude not stationary')
    else:
        print('Conclude stationary')
    
# KPSS Test
def KPSS_test(data):
    result = kpss(data.values, regression='c', lags='auto')
    print('============================================================')
    print('Results of KPSS Test for {}:'.format(crypto_name))
    print('============================================================')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

# seasonal decomposition
def simple_seasonal_decompose(data,freq):
    rcParams['figure.figsize'] = 10, 8
    decomposition = seasonal_decompose(data, model='additive', freq=freq)
    decomposition.plot()
    plt.show()
    

def simple_plot_acf(data):
    fig = tsaplots.plot_acf(data, lags=40)
    plt.show()
    
def simple_plot_pacf(data):
    fig = tsaplots.plot_pacf(data, lags=40)
    plt.show()

def rolling_mean_std(timeseries, freq):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=freq).mean()
    rolstd = timeseries.rolling(window=freq).std()
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# =============================================================================
# Examining CLOSE
# =============================================================================

simple_seasonal_decompose(y['Close'], 365)
simple_plot_pacf(y['Close'])
simple_plot_acf(y['Close'])
KPSS_test(y['Close'])
adfuller_test(y['Close'])
rolling_mean_std(y['Close'], 365)

# =============================================================================
# Examining LOG CLOSE
# =============================================================================

simple_seasonal_decompose(y['log_Close'], 365)
simple_plot_pacf(y['log_Close'])
simple_plot_acf(y['log_Close'])
KPSS_test(y['log_Close'])
adfuller_test(y['log_Close'])
rolling_mean_std(y['log_Close'], 365)

# =============================================================================
# Examining DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['diff'], 365)
simple_plot_pacf(y['diff'])
simple_plot_acf(y['diff'])
KPSS_test(y['diff'])
adfuller_test(y['diff'])
rolling_mean_std(y['diff'], 365)

# =============================================================================
# Examining LOG CLOSE DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['log_Close_diff'], 365)
simple_plot_pacf(y['log_Close_diff'])
simple_plot_acf(y['log_Close_diff'])
KPSS_test(y['log_Close_diff'])
adfuller_test(y['log_Close_diff'])
rolling_mean_std(y['log_Close_diff'], 365)

# =============================================================================
# Monthly Data - 2511 observations to 82 - Not good
# =============================================================================
# RESAMPLING DATA INTO MONTHL1Y
monthly_y = y.copy()
monthly_y.resample('M').mean().head()
monthly_y = monthly_y.asfreq('M')
#monthly_y.resample('M').median().head()


# DIFF - STATIONARY
simple_seasonal_decompose(monthly_y['diff'], 12)
simple_plot_pacf(monthly_y['diff'])
simple_plot_acf(monthly_y['diff'])
KPSS_test(monthly_y['diff'])
adfuller_test(monthly_y['diff'])
rolling_mean_std(monthly_y['diff'], 365)


# LOGGED CLOSE DIFF - STATIONARY
simple_seasonal_decompose(monthly_y['log_Close_diff'], 12)
simple_plot_pacf(monthly_y['log_Close_diff'])
simple_plot_acf(monthly_y['log_Close_diff'])
KPSS_test(monthly_y['log_Close_diff'])
adfuller_test(monthly_y['log_Close_diff'])
rolling_mean_std(monthly_y['log_Close_diff'], 365)




# =============================================================================
# 
# =============================================================================
y['moving_avg']= y['log_Close'].rolling(window=365).mean()
plt.plot(y['log_Close'])
plt.plot(y['moving_avg'], color='red')
plt.show()


y['ts_log_moving_avg_diff'] = y['log_Close'] - y['moving_avg']

y['ts_log_moving_avg_diff'].dropna(inplace=True)
y['ts_log_moving_avg_diff'].head(15)
test_stationarity(y['ts_log_moving_avg_diff'], 365)

















# =============================================================================
# 
# =============================================================================

from statsmodels.tsa.arima_model import ARMA


# Instantiate the model
model = ARMA(y['Close'], order = (2,1))

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

# Generate predictions
forecast = results.get_prediction(start=-25)

# forecast mean
mean_forecast = forecast.predicted_mean

# =============================================================================
# 
# =============================================================================


import statsmodels.api as sm

new_y = y[['diff']]
new_y.index


# Construct the model
mod = sm.tsa.SARIMAX(new_y, order=(1, 0, 0), trend='c')
# Estimate the parameters
res = mod.fit()

print(res.summary())


# The default is to get a one-step-ahead forecast:
print(res.forecast())


# Here we construct a more complete results object.
fcast_res1 = res.get_forecast()

# Most results are collected in the `summary_frame` attribute.
# Here we specify that we want a confidence level of 90%
print(fcast_res1.summary_frame(alpha=0.10))


print(res.forecast(steps=2))
fcast_res2 = res.get_forecast(steps=2)
# Note: since we did not specify the alpha parameter, the
# confidence level is at the default, 95%
print(fcast_res2.summary_frame())



fig, ax = plt.subplots(figsize=(15, 5))

# Plot the data (here we are subsetting it to get a better look at the forecasts)
new_y.iloc[0:].plot(ax=ax)

# Construct the forecasts
fcast = res.get_forecast('2021-08-30').summary_frame()
fcast['mean'].plot(ax=ax, style='k--')
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.show()




































# =============================================================================
# Boxplots of Returns with PLOTLY
# =============================================================================
def box_year():
    fig = go.Figure()

    
    fig.add_trace(go.Box(x = df.index.year, y = df['daily_return'],
                         customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Daily Return: %{y:.0%}<br>"+
                                    "<extra></extra>"))

    fig.update_layout(
        title = 'Daily Returns of {}'.format(crypto_name),
        yaxis_title = '% Change',
    yaxis_tickformat = ',.0%')
    fig.show()

box_year()


## DOES NOT WORK!!!!!!!
#def box_month_year():
#    fig = go.Figure()
#
#    
#    fig.add_trace(go.Box(x = df['month_year'], y = df['monthly_return'],
#                         customdata = df['Name'],
#                            hovertemplate="<b>%{customdata}</b><br><br>" +
#                                    "Date: %{x|%d %b %Y} <br>" +
#                                    "Monthly Return: %{y:.0%}<br>"+
#                                    "<extra></extra>"))
#
#    fig.update_layout(
#        title = 'Monthly Returns of {}'.format(crypto_name),
#        yaxis_title = '% Change',
#    yaxis_tickformat = ',.0%')
#    fig.show()
#
#box_month_year()
# =============================================================================
# Decomposition with PLOTLY PACKAGE!
# =============================================================================

def decomposition(data, freq):
    
    decomposition = sm.tsa.seasonal_decompose(data, model='additive', freq = freq)
    
    #seasonality
    decomp_seasonal = decomposition.seasonal

    #trend
    decomp_trend = decomposition.trend

    #residual
    decomp_resid = decomposition.resid

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[
            'Price  of {}'.format(str(crypto_name)),
            'Trend values of {}'.format(str(crypto_name)),
            'Seasonal values of {}'.format(str(crypto_name)),
            'Residual values of {}'.format(str(crypto_name))])


    fig.add_trace(go.Scatter(x = df.index,
                            y = data,
                            name = crypto_name, 
                            mode='lines'),row = 1, col = 1)


    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_trend,
                            name = 'Trend', 
                            mode='lines'),row = 2, col = 1)


    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_seasonal,
                            name = 'Seasonality', 
                            mode='lines'),row = 3, col = 1)

    fig.add_trace(go.Scatter(x = df.index,
                            y = decomp_resid,
                            name = 'Residual', 
                            mode='lines'),row = 4, col = 1)

    # Add titles
    fig.update_layout( 
            title = 'Decomposition of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Trend'
    fig['layout']['yaxis3']['title']='Seasonality'
    fig['layout']['yaxis4']['title']='Residual'

    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    fig.update_yaxes(tickformat = ',.', row = 2, col = 1)
    fig.update_yaxes(tickformat = ',.', row = 3, col = 1)
    fig.update_yaxes(tickformat = ',.', row = 4, col = 1)

    fig.show()



decomposition(df['Close'], 365)