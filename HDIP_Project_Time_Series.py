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
# 
# =============================================================================

start = "1980-01-01"
end = dt.datetime.now()
short_sma = 50
long_sma = 200
symbol = 'MSFT'

# creating a dataset for selected cryptocurrency 
df = yf.download(symbol, start, end,interval = '1d')
df = pd.DataFrame(df.dropna(), columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume'])

# Create short SMA
df['short_SMA'] = df.iloc[:,1].rolling(window = short_sma).mean()

# Create Long SMA
df['long_SMA'] = df.iloc[:,1].rolling(window = long_sma).mean()

# Create daily_return
df['daily_return'] = df['Close'].pct_change(periods=1).mul(100)

# Create monthly_return
df['monthly_return'] = df['Close'].pct_change(periods=30).mul(100)

# Create annual_return
df['annual_return'] = df['Close'].pct_change(periods=365).mul(100)
df['Name'] = symbol
print('============================================================')
print(symbol, '- Full Dataset')
print('------------------------------------------------------------')
print(df.head())
print('------------------------------------------------------------')
print(symbol, 'Full Dataset - Column Names')
print(df.columns)
print('============================================================')

# preparing data from time series analysis
# eliminating any NAs - in most cryptocurrencies there are 4 days missing
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')
print('Nan in each columns' , df.isna().sum())
df = df.ffill()
print('Nan in each columns' , df.isna().sum())


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
# Lagged and Shifted
# =============================================================================


# Create 'lagged' and 'shifted'
df['lagged'] =  df['Close'].shift(periods=-90)
df['shifted'] = df['Close'].shift(periods=90)

# Plot the google price series
df[['Close', 'lagged', 'shifted']].plot(subplots=True)
plt.show()




## =============================================================================
## 
## =============================================================================
#
## Select first prices
#first_price = df['Close'].iloc[0]
#print('Fist Price', first_price)
#
## Create normalized
#normalized = df['Close'].div(first_price).mul(100)
#print('Normalised Price', normalized)
#
## Plot normalized
#normalized.plot()
#plt.show()
#






# Create daily_return
df['daily_return'] = df['Close'].pct_change(periods=1).mul(100)

# Create monthly_return
df['monthly_return'] = df['Close'].pct_change(periods=30).mul(100)

# Create annual_return
df['annual_return'] = df['Close'].pct_change(periods=365).mul(100)
df['Name'] = crypto_name


import seaborn as sns
fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(df.daily_return.index.year, df.daily_return, ax=ax)
plt.show()






df['month_year'] = pd.to_datetime(df.index).to_period('M')

fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(df['month_year'], df['daily_return'], ax=ax)
ax.set_xticklabels(df['month_year'].unique(),  rotation = 45, fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(df['month_year'], df['monthly_return'], ax=ax)
ax.set_xticklabels(df['month_year'].unique(),  rotation = 45, fontsize = 12)
plt.show()



#### NOT WORKING!!!!!!
def box_year():
    fig = go.Figure()

    
    fig.add_trace(go.Box(x = df.index, y = df['daily_return']))

#    fig.update_layout(
#        title = 'Price of {}'.format(crypto_name),
#        yaxis_title = 'US Dollars',
#        yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()

box_year()

## =============================================================================
## Adfuller Test
## =============================================================================
# creating a function to run an adfuller-dickey test on target data
def adfuller_test(data):
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print('============================================================')
    print('Results of Dickey-Fuller Test for {}:'.format(crypto_name))
    print('============================================================')
    print (dfoutput)
    

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
    
KPSS_test(y['log_Close'])

adfuller_test(y['log_Close'])

# =============================================================================
# ACF and PACF charts
# =============================================================================


fig = tsaplots.plot_acf(y['log_Close'], lags=40)
plt.show()


fig = tsaplots.plot_pacf(y['log_Close'], lags=40)
plt.show()




# =============================================================================
# 
# =============================================================================


# Get month for each dates from the index of airline
index_month = df.index.month

# Compute the mean number of passengers for each month of the year
df_close_by_month = df['Close'].groupby(index_month).mean()

# Plot the mean number of passengers for each month of the year
df_close_by_month.plot()
plt.legend(fontsize=20)
plt.show()

# =============================================================================
# Decomposition
# =============================================================================


rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df['Close'])
fig_decomposition = decomposition.plot()
plt.show()




#seasonality - WTF???
decomp_seasonal = decomposition.seasonal
ax = decomp_seasonal.plot(figsize=(14, 2))
ax.set_xlabel('Date')
ax.set_ylabel('Seasonality of time series')
ax.set_title('Seasonal values of the time series')
plt.show()

#trend
decomp_trend = decomposition.trend
ax = decomp_trend.plot(figsize=(14, 2))
ax.set_xlabel('Date')
ax.set_ylabel('Trend of time series')
ax.set_title('Trend values of the time series')
plt.show()

#residual
decomp_resid = decomposition.resid
ax = decomp_resid.plot(figsize=(14, 2))
ax.set_xlabel('Date')
ax.set_ylabel('Residual of time series')
ax.set_title('Residual values of the time series')
plt.show()

#
#decomp_seasonal = pd.DataFrame(decomp_seasonal)
#decomp_trend= pd.DataFrame(decomp_trend)
#decomp_resid= pd.DataFrame(decomp_resid)
#
#
#decomposed = pd.concat(decomp_seasonal,decomp_trend, decomp_resid)
#
## Print the first 5 rows of airline_decomposed
#print(decomposed)
#
## Plot the values of the airline_decomposed DataFrame
#ax = decomposed.plot(figsize=(12, 6), fontsize=15)
#
## Specify axis labels
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(fontsize=15)
#plt.show()

# =============================================================================
# Decomposition
# =============================================================================

############# Trying get PLOTLY #############
# https://github.com/Pierian-Data/AutoArima-Time-Series-Blog/blob/master/Forecasting%20a%20Time%20Series%20in%20Python.ipynb
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(y['Close'], model='multiplicative')
fig = result.plot()
plot_mpl(fig)




# =============================================================================
# 
# =============================================================================

from statsmodels.tsa.arima_model import ARMA
import statsmodels

# A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.

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

# Ge
# =============================================================================
# 
# =============================================================================


import statsmodels.api as sm

df = df[['Close']]



# Construct the model
mod = sm.tsa.SARIMAX(df['Close'], order=(1, 0, 0), trend='c')
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
df.iloc[0:].plot(ax=ax)

# Construct the forecasts
fcast = res.get_forecast('2021-08-30').summary_frame()
fcast['mean'].plot(ax=ax, style='k--')
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.show()
