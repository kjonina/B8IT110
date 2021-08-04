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

crypto_name = 'Ethereum'

insert = 'ETH-USD'


# Fixing issues with frequency

df = df.asfreq('D')
print('Nan in each columns' , df.isna().sum())


y = y.asfreq('D')
print('Nan in each columns' , y.isna().sum())


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
def simple_seasonal_decompose(data,number):
    rcParams['figure.figsize'] = 10, 8
    decomposition = seasonal_decompose(data, model='additive', period=number)
    decomposition.plot()
    plt.show()
    

def simple_plot_acf(data, no_lags):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,5))
    ax1.plot(data)
    ax1.set_title('Original')
    plot_pacf(data, lags=no_lags, ax=ax2);
    plt.show()

    
def simple_plot_pacf(data, no_lags):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,5))
    ax1.plot(data)
    ax1.set_title('Original')
    plot_acf(data, lags=no_lags, ax=ax2);
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

def create_train_and_test():
    global df_train 
    global df_test
    # Train data - 80%
    df_train = y[:int(0.90*(len(y)))]
    
    print('============================================================')
    print('{} Training Set'.format(crypto_name))
    print('============================================================')
    print(df_train.head())
    print('Training set has {} rows and {} columns.'.format(*df_train.shape))
    # Test data - 20%
    df_test = y[int(0.90*(len(y))):]
    print('============================================================')
    print('{} Test Set'.format(crypto_name))
    print('============================================================')
    print(df_test.head())
    print('Test set has {} rows and {} columns.'.format(*df_test.shape))
    

def training_and_test_plot(): 
    # creating a plotly graph for training and test set
    df_train['Close'].plot()
    df_test['Close'].plot()
    plt.show()

create_train_and_test()

training_and_test_plot()

# =============================================================================
# Examining CLOSE
# =============================================================================

simple_seasonal_decompose(y['Close'], 365)
simple_plot_pacf(y['Close'], 60)
simple_plot_acf(y['Close'], 60)
KPSS_test(y['Close'])
adfuller_test(y['Close'])
rolling_mean_std(y['Close'], 365)

# =============================================================================
# Examining LOG CLOSE
# =============================================================================

simple_seasonal_decompose(y['log_Close'], 365)
simple_plot_pacf(y['log_Close'], 60)
simple_plot_acf(y['log_Close'], 60)
KPSS_test(y['log_Close'])
adfuller_test(y['log_Close'])
rolling_mean_std(y['log_Close'], 365)

# =============================================================================
# Examining DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['diff'], 365)
simple_plot_pacf(y['diff'], 60)
simple_plot_acf(y['diff'], 60)
KPSS_test(y['diff'])
adfuller_test(y['diff'])
rolling_mean_std(y['diff'], 365)

# =============================================================================
# Examining LOG CLOSE DIFF - STATIONARY
# =============================================================================

simple_seasonal_decompose(y['log_Close_diff'], 365)
simple_plot_pacf(y['log_Close_diff'],40)
simple_plot_acf(y['log_Close_diff'],40)
KPSS_test(y['log_Close_diff'])
adfuller_test(y['log_Close_diff'])
rolling_mean_std(y['log_Close_diff'], 365)


## =============================================================================
## Monthly Data - 2511 observations to 82 - Not good
## =============================================================================
## RESAMPLING DATA INTO MONTHL1Y
#monthly_y = y.copy()
#monthly_y.resample('M').mean().head()
#monthly_y = monthly_y.asfreq('M')
##monthly_y.resample('M').median().head()
#
#
## DIFF - STATIONARY
#simple_seasonal_decompose(monthly_y['diff'], 12)
#simple_plot_pacf(monthly_y['diff'])
#simple_plot_acf(monthly_y['diff'])
#KPSS_test(monthly_y['diff'])
#adfuller_test(monthly_y['diff'])
#rolling_mean_std(monthly_y['diff'], 365)
#
#
## LOGGED CLOSE DIFF - STATIONARY
#simple_seasonal_decompose(monthly_y['log_Close_diff'], 12)
#simple_plot_pacf(monthly_y['log_Close_diff'])
#simple_plot_acf(monthly_y['log_Close_diff'])
#KPSS_test(monthly_y['log_Close_diff'])
#adfuller_test(monthly_y['log_Close_diff'])
#rolling_mean_std(monthly_y['log_Close_diff'], 365)

# =============================================================================
# AR MODEL
# =============================================================================

from statsmodels.tsa.ar_model import AR
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

def run_AR_model(data):
    # AR fit model
    model = AR(data)
    results = model.fit()

    print (results.summary())

    
    plt.plot(data)
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% np.nansum((results.fittedvalues-data)**2))
    plt.show()
    
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    print('============================================================')
    print('predictions_ARIMA_diff')
    print('============================================================')
    print (predictions_ARIMA_diff.tail())
    
    
    # Cumulative Sum to reverse differencing:
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print('============================================================')
    print('predictions_ARIMA_diff_cumsum')
    print('============================================================')
    
    print (predictions_ARIMA_diff_cumsum.head())
    
    predictions_ARIMA_log = pd.Series(y['log_Close'].iloc[0], index=y['log_Close'].index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    print('============================================================')
    print('predictions_ARIMA_log')
    print('============================================================')
    print (predictions_ARIMA_log.tail())
    
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('============================================================')
    print('predictions_ARIMA')
    print('============================================================')
    print (predictions_ARIMA.tail())



    fig, ax = plt.subplots(figsize=(15, 5), sharey=False) 
    y['Close'].plot(ax=ax)
    predictions_ARIMA.plot(ax=ax)
    plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA-y.Close)**2)/len(y.Close)))
    plt.show();
    



def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y,pred):
    results = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['msle'] = mean_squared_log_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results



run_AR_model(y['log_Close_diff'])


evaluate_forecast(y.Close, predictions_ARIMA).transpose()


# =============================================================================
# EXACTLY LIKE ABOVE BUT DOES NOT DO ANYTHING
# ARIMA MODEL - NOT WORKING
# =============================================================================
# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
    
def run_ARIMA_model():
    # fit model
    model = ARIMA(y['log_Close_diff'], order=(4, 1, 4))
    results = model.fit()

    print (results.summary())

    
    plt.plot(y['log_Close_diff'])
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% np.nansum((results.fittedvalues-y['log_Close_diff'])**2))
    plt.show()
    
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    print('============================================================')
    print('predictions_ARIMA_diff')
    print('============================================================')
    print (predictions_ARIMA_diff.tail())
    
    
    # Cumulative Sum to reverse differencing:
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print('============================================================')
    print('predictions_ARIMA_diff_cumsum')
    print('============================================================')
    
    print (predictions_ARIMA_diff_cumsum.head())
    
    predictions_ARIMA_log = pd.Series(y['log_Close'].iloc[0], index=y['log_Close'].index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    print('============================================================')
    print('predictions_ARIMA_log')
    print('============================================================')
    print (predictions_ARIMA_log.tail())
    
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('============================================================')
    print('predictions_ARIMA')
    print('============================================================')
    print (predictions_ARIMA.tail())
        
    plt.plot(y.Close)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA - y['log_Close_diff'])**2)/len(y['log_Close_diff'])))
    plt.show();
    
    results.plot_diagnostics()
    plt.show()


run_ARIMA_model()





# =============================================================================
# ARIMA MODEL
# =============================================================================

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
    
def ARIMA_forecasting_dftest_with_log_Close_diff():
    # fit model
    model = ARIMA(df_train['log_Close_diff'], order=(6, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    
    
    start_index = df_test.index.min()
    end_index = df_test.index.max()
    
    

    
    #Predictions
    predictions = model_fit.predict(steps=219, dynamic = True)
        
#    # Confidence level of 90%
#    predictions = forecast.summary_frame(alpha=0.10) 
#    print('============================================================')
#    print('Forecast')
#    print('============================================================')
#    print(predictions.tail())    


    predictions_ARIMA_diff = pd.Series(predictions, copy=True)
    print('============================================================')
    print('predictions_ARIMA_diff')
    print('============================================================')
    print (predictions_ARIMA_diff.head())
    
    
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print (predictions_ARIMA_diff_cumsum.head())
    print('============================================================')
    print('predictions_ARIMA_diff_cumsum')
    print('============================================================')
    print (predictions_ARIMA_diff_cumsum.tail())    
    
    
    
    predictions_ARIMA_log = pd.Series(df_test.Close.iloc[0], index=df_test.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA_log.head()
    print('============================================================')
    print('predictions_ARIMA_log')
    print('============================================================')
    print (predictions_ARIMA_log.tail())
    
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('============================================================')
    print('predictions_ARIMA')
    print('============================================================')
    print (predictions_ARIMA.tail())
        
    plt.plot(y.Close)
    plt.plot(predictions_ARIMA_log)
    plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA_log - y['log_Close_diff'])**2)/len(y['log_Close_diff'])))
    plt.show();
    
    model_fit.plot_diagnostics()
    plt.show()


ARIMA_forecasting_dftest_with_log_Close_diff()
# =============================================================================
#  DATACAMP CODE
# =============================================================================


from statsmodels.tsa.arima.model import ARIMA


shape = pd.DataFrame(df_test.shape)
size = shape.loc[0,0]
print(size)

def ARIMA_forecasting_dftest_with_Close():
    # Instantiate the model
    model =  ARIMA(df_train['Close'], order=(6,1,3))
    
    # Fit the model
    results = model.fit()
    
    # Print summary
    print(results.summary())
    
#    start_index = df_test.index.min()
#    end_index = df_test.index.max()

    
    #Predictions
    forecast = results.get_forecast(steps=219, dynamic = True)
        
    # Confidence level of 90%
    fcast = forecast.summary_frame(alpha=0.10) 
    print('============================================================')
    print('Forecast')
    print('============================================================')
    print(fcast.tail())
    
    
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # plot the data
    plt.plot(y.index, y['Close'], label='observed')
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
    
    # set labels, legends and show plot
    plt.xlabel('Date')
    plt.ylabel('{} Stock Price - Close USD'.format(str(crypto_name)))
    plt.legend()
    plt.show()


ARIMA_forecasting_dftest_with_Close()


"""
Maybr Try SAMIRAX with  exog = df['Volume']

"""












# =============================================================================
# FORECASTING!
# =============================================================================
#https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html

import statsmodels.api as sm


def ARIMA_forecasting_with_Close():
    
    # Construct the model
#    mod = sm.tsa.SARIMAX(y[['Close']], order=(1, 0, 0), trend='c')
    
    mod =ARIMA(y['Close'], order=(6,1,3))
    # Estimate the parameters
    res = mod.fit()
    print(res.summary())
    
    
    # Forecasting out-of-sample
    forecast = res.get_forecast(steps=120, dynamic = True)
    
    # Confidence level of 90%
    print('============================================================')
    print('Forecast')
    print('============================================================')
    print(forecast.summary_frame(alpha=0.10).tail())
    
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the data (here we are subsetting it to get a better look at the forecasts)
    y[['Close']].iloc[0:].plot(ax=ax)
    
    # Construct the forecasts
    fcast = res.get_forecast('2021-12-30').summary_frame()
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
    plt.show()

    print('============================================================')
    print('Plot Diagnostics')
    print('============================================================')
    res.plot_diagnostics()
    plt.show()



ARIMA_forecasting_with_Close()

""" NOTES:
Does not actually run on the log_Close_diff 
So it does not have to be reconstructed after differencing and exponentional from log
"""

































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

    
    decomposition = sm.tsa.seasonal_decompose(data, model='additive', period = freq)
    
    #seasonality
    decomp_seasonal = decomposition.seasonal

    #trend
    decomp_trend = decomposition.trend

    #residual
    decomp_resid = decomposition.resid

    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, subplot_titles=[
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


# =============================================================================
# AR model with PLOTLY
# =============================================================================


from statsmodels.tsa.ar_model import AR
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

def run_AR_model_with_PLOTLY():
    # AR fit model
    model = AR(y['log_Close_diff'])
    results = model.fit()

    print (results.summary())

    plt.plot(y['log_Close_diff'])
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% np.nansum((results.fittedvalues-y['log_Close_diff'])**2))
    plt.show()
    
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    print('============================================================')
    print('predictions_ARIMA_diff')
    print('============================================================')
    print (predictions_ARIMA_diff.tail())
    
    # Cumulative Sum to reverse differencing:
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print('============================================================')
    print('predictions_ARIMA_diff_cumsum')
    print('============================================================')
    print (predictions_ARIMA_diff_cumsum.head())
    
    predictions_ARIMA_log = pd.Series(y['log_Close'].iloc[0], index=y['log_Close'].index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    print('============================================================')
    print('predictions_ARIMA_log')
    print('============================================================')
    print (predictions_ARIMA_log.tail())
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('============================================================')
    print('predictions_ARIMA')
    print('============================================================')
    print (predictions_ARIMA.tail())
    # creating a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df.index,
        y = df['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Observed Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Observed')
    
    trace2 = go.Scatter(
        x = predictions_ARIMA.index,
        y = predictions_ARIMA,
        name = 'Predicted',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    fig.update_layout({'title': {'text':'ARIMA Forecasting of {}'.format(str(crypto_name))}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
    
    
    
run_AR_model_with_PLOTLY()





from statsmodels.tsa.arima.model import ARIMA

def ARIMA_forecasting_with_Close_PLOTLY():
    
    # Construct the model
    #    mod = sm.tsa.SARIMAX(y[['Close']], order=(1, 0, 0), trend='c')
    
    mod =ARIMA(y['Close'], order=(6,1,3))
    # Estimate the parameters
    res = mod.fit()
    print(res.summary())
    
    
    # Forecasting out-of-sample
    forecast = res.get_forecast(steps=120, dynamic = True)
    
    # Confidence level of 90%
    print('============================================================')
    print('Forecast')
    print('============================================================')
    print(forecast.summary_frame(alpha=0.10).tail())
    
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the data (here we are subsetting it to get a better look at the forecasts)
    y[['Close']].iloc[0:].plot(ax=ax)
    
    # Construct the forecasts
    fcast = res.get_forecast('2021-12-30').summary_frame()
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
    plt.show()
    
    
    
    # a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df.index,
        y = df['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")
    

    
    print(fcast.index)
    y_upper = fcast['mean_ci_upper']
    y_lower = fcast['mean_ci_lower']
    
    trace2 = go.Scatter(
        x=y_upper.index,
        y=y_upper, 
        line = dict(color='green'),
#        name = 'Predicted2',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")
    

    trace3 = go.Scatter(
        x=y_upper.index,
        y= y_lower,
        line = dict(color='green'),
#        name = 'Predicted Lower Confidence ',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        fill='tonexty'
        )


    trace4 = go.Scatter(
        x=fcast['mean_ci_upper'].index,
        y=fcast['mean'],
#        name = 'Predicted', 
        line = dict(color='firebrick', width=4, dash='dot'),
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")
    
    
    
    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data = data)
    fig.update_layout(showlegend=False)
    
    fig.update_layout({'title': {'text':'ARIMA Forecasting of {}'.format(str(crypto_name))}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
    
    
    
#    print('============================================================')
#    print('Plot Diagnostics')
#    print('============================================================')
#    res.plot_diagnostics()
#    plt.show()




ARIMA_forecasting_with_Close_PLOTLY()







def ARIMA_forecasting_dftest_with_Close_PLOTLY():
    # Instantiate the model
    model =  ARIMA(df_train['Close'], order=(6,1,3))
    
    # Fit the model
    results = model.fit()
    
    # Print summary
    print(results.summary())
    
#    start_index = df_test.index.min()
#    end_index = df_test.index.max()

    
    #Predictions
    forecast = results.get_forecast(steps=219, dynamic = True)
        
    # Confidence level of 90%
    fcast = forecast.summary_frame(alpha=0.10) 
    print('============================================================')
    print('Forecast')
    print('============================================================')
    print(fcast.tail())
    
    
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(y.index, y['Close'], label='observed')
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
    plt.xlabel('Date')
    plt.ylabel('{} Stock Price - Close USD'.format(str(crypto_name)))
    plt.legend()
    plt.show()

    
    
    
    # a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df.index,
        y = df['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")

    trace5 = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")

    
    print(fcast.index)
    y_upper = fcast['mean_ci_upper']
    y_lower = fcast['mean_ci_lower']
    
    trace2 = go.Scatter(
        x=y_upper.index,
        y=y_upper, 
        line = dict(color='green'),
#        name = 'Predicted2',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")
    

    trace3 = go.Scatter(
        x=y_upper.index,
        y= y_lower,
        line = dict(color='green'),
#        name = 'Predicted Lower Confidence ',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        fill='tonexty'
        )


    trace4 = go.Scatter(
        x=fcast['mean_ci_upper'].index,
        y=fcast['mean'],
#        name = 'Predicted', 
        line = dict(color='firebrick', width=4, dash='dot'),
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Predicted Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>")
    
    
    
    data = [trace1, trace2, trace3, trace4, trace5]
    fig = go.Figure(data = data)
    fig.update_layout(showlegend=False)
    
    fig.update_layout({'title': {'text':'ARIMA Forecasting of {}'.format(str(crypto_name))}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()


ARIMA_forecasting_dftest_with_Close_PLOTLY()





