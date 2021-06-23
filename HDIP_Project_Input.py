

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
# Downloading the top 10 cryptocurrencies based on market cap.
# =============================================================================
cryptolist = ['BTC-USD','ETH-USD', 'USDT-USD','BNB-USD', 'ADA-USD', 'DOGE-USD','XRP-USD','USDC-USD','DOT1-USD']
start = "2009-01-01"
end = dt.datetime.now()
short_sma = 20
long_sma = 50

# creating a dataset for each cryptocurrency 
df = {}
for crypto in cryptolist :
    data = yf.download(crypto, start, end,interval = '1d')
    df[crypto] = pd.DataFrame(data.dropna(), columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume'])
    df[crypto]['short_SMA'] = df[crypto].iloc[:,1].rolling(window = short_sma).mean()
    df[crypto]['long_SMA'] = df[crypto].iloc[:,1].rolling(window = long_sma).mean()


keys = df.keys()
keys = pd.DataFrame(keys)
print(keys)

# assigning a value to input
insert = str(input('What cryptocurrency would you like to try out?')).upper()
  
# =============================================================================
# Trying to create an error message    
# ============================================================================
def select_ticket():
    print(keys)
    while True:
        try:
            insert = str(input('What cryptocurrency would you like to try out?')).upper()
            
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
    
        if not insert in cryptolist:
            print('Sorry. You did not select an available ticket or you misspelled the ticket')
            
        else:
            print(insert)
            break

''' #only global ''' 
select_ticket()


    
# assigning a value to input to SMA and LMA

while True:
    try:
        insert_SMA = int(input("Please write down the length of time you would like to SMA: "))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    if insert_SMA < 0:
        print("Sorry, your response must not be negative.")
        continue
    else:
        #age was successfully parsed, and we're happy with its value.
        #we're ready to exit the loop.
        break
if insert_SMA >= 18: 
    print("You are able to vote in the United States!")
else:
    print("You are not able to vote in the United States.")

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
'''https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=6'''

def create_graphs(data):

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,  subplot_titles=[
            'Price and Moving Averages of {}'.format(str(data)),
            'Candlesticks for {}'.format(str(data)),
            'Volume of {}'.format(str(data))])
    # Lineplots of price and moving averages
    fig.add_trace(go.Scatter(
                            x = df[data].index,
                            y = df[data]['Close'],
                            name = data, 
                            mode='lines',
                            line = dict(color="black")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df[data].index,
                             y = df[data]['short_SMA'],
                             name = 'Short SMA',
                             mode = 'lines',
                             line = dict(color="red")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df[data].index,
                             y = df[data]['long_SMA'],
                             name = 'Long SMA',
                             mode = 'lines',
                             line = dict(color="green")), row = 1, col = 1)
    # Candlestick
    fig.add_trace(go.Candlestick(x = df[data].index,
                    open = df[data]['Open'],
                    high = df[data]['High'],
                    low = df[data]['Low'],
                    close = df[data]['Close'],
                    name = 'market data'), row = 2, col = 1)
    # Barplot of volume 
    fig.add_trace(go.Bar(x = df[data].index,
                    y = df[data]['Volume'],
                    name = 'Volume',
                    marker = dict(color="black", opacity = True)), row = 3, col = 1)
    # Add titles
    fig.update_layout(
        title = 'Price of {}'.format(str(data)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='US Dollars'
    fig['layout']['yaxis3']['title']='Volume'
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count = 7, label = "1W", step = "day", stepmode = "backward"),
                dict(count = 28, label = "1M", step = "day", stepmode = "backward"),
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5Y", step = "year", stepmode = "backward"),
                dict(step = "all")])))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider= {'visible':False}, row=2, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=2, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=3, col=1)
    #Show
    fig.show()


# making it easier to call the GRAPHS
create_graphs(insert)
"""
TO FIX
- ticks
- create a slider for short and long SMA
"""


# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================
df_new = df[insert]

df_new = pd.DataFrame(df_new, columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume', 'short_SMA', 'long_SMA'])

# ensu
df_new.sort_index(inplace = True)

# dropping hh:mm:ss from index
df_new.index = pd.to_datetime(df_new.index)
print(df_new)



def create_hist_and_box(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Histogram of {} price'.format(insert),
                                        'Box plot of {} price'.format(insert)],
                        x_title = 'US Dollars')
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = round(len(df_new) / 20)), row=1, col=1)
    fig.add_trace(go.Box(x = data, name = 'Boxplot'), row=2, col=1)
    fig.update_layout({'title': {'text':'Plots of {} price'.format(insert)}})
    fig.show()

create_hist_and_box(df_new['Close'])


"""
TO FIX
- ticks
- create a slider for time period?
"""
# =============================================================================
# Assigning the target variable
# =============================================================================

# assigning the target variable
y = pd.DataFrame(df_new['Close'], columns = ['Close'])
y.sort_index(inplace = True)

print(y.head())

# =============================================================================
# Decomposition
# =============================================================================
# Viewing the seasonal decompose of the target variable
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y.asfreq('MS'), model='multiplicative')
fig = decomposition.plot()
plt.show()

# =============================================================================
# Adfuller Test
# =============================================================================
# creating a function to run an adfuller-dickey test on target data
def adfuller_test(data):
    print ('Results of Dickey-Fuller Test for {}:'.format(insert))
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

# running an adfuller dickey test on the target 
adfuller_test(y['Close'])

# 0th element is test statistic (-1.34)
# More negative means more likely to be stationary
# 1st element is p-value: (0.60)
# If p-value is small → reject null hypothesis. Reject non-stationary.
# 4th element is the critical test statistics

# =============================================================================
# 
# =============================================================================
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()

    #Plot rolling statistics:   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = timeseries.index,
                                y = timeseries,
                                name = 'Original', 
                                mode='lines',
                                line = dict(color="blue")))
    fig.add_trace(go.Scatter(x = timeseries.index,
                                y = rolmean,
                                name = 'Rolling Mean', 
                                mode='lines',
                                line = dict(color="red")))
    fig.add_trace(go.Scatter(x = y.index,
                                y = rolstd,
                                name = 'Rolling Std', 
                                mode='lines',
                                line = dict(color="black")))
    # Add titles
    fig.update_layout(
            title = 'Rolling Mean & Standard Deviation of {}'.format(insert),
            yaxis_title = 'US Dollars')
    #Show
    fig.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test of {}: '.format(insert))
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


test_stationarity(y['Close'])


# =============================================================================
# ACF and PACF plots
# =============================================================================

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
# Make ACF plot
plot_acf(y['Close'], lags=30, zero=False, ax=ax1)
# Make PACF plot
plot_pacf(y['Close'], lags=30, zero=False, ax=ax2)
plt.show()



# df['sum'] is my time series where i want the pacf of.
df_pacf = pacf(y['Close'], nlags=300)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = np.arange(len(df_pacf)),
    y = df_pacf,
    name = 'PACF',
    ))
fig.update_xaxes(rangeslider_visible = True)
fig.update_layout(
    title = "Partial Autocorrelation",
    xaxis_title = "Lag",
    yaxis_title = "Partial Autocorrelation",
    #     autosize = False,
    #     width = 500,
         height = 500,
    )
fig.show()

from scipy.signal import detrend
from statsmodels.graphics.tsaplots import plot_acf
stat_ts = pd.Series(detrend(np.log(y.Close)), index=y.index)
plot_acf(stat_ts)

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(stat_ts)
# =============================================================================
# Making the dataset stationary
# =============================================================================
# Creating a new variable, examining the difference for each observation
y['diff'] = y['Close'].diff()
# dropping the first na (because there is no difference)
y = y.dropna()

# creating the plot to examine the difference
def diff_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index,
                            y = data['diff'],
                            name = str(insert), 
                            mode='lines'))
    # Add titles
    fig.update_layout(
        title = 'Price of {}'.format(insert),
        yaxis_title = 'US Dollars')
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count = 7, label = "1W", step = "day", stepmode = "backward"),
                dict(count = 28, label = "1M", step = "day", stepmode = "backward"),
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5Y", step = "year", stepmode = "backward"),
                dict(step = "all")])))
    #Show
    fig.show()
    
#plotting the graph for ['diff'] variable
diff_plot(y)

#checking the adfuller of the ['diff'] variable
adfuller_test(y['diff'])

# =============================================================================
# Log the data
# =============================================================================
# logging the target varialbe due to great variance
y['log_Close'] = np.log(y['Close'])

# examining the columns
y.describe()

# function to create a histogram and lineplot of logged data
def log_create_hist_and_line(data):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Histogram of {} logged price'.format(insert),
                                        'Line plot of {} logged price'.format(insert)])
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = 100), row=1, col=1)
    fig.add_trace(go.Scatter(x = data.index , y = data, mode = 'lines', name = 'lineplot'), row=2, col=1)
    fig.update_layout({'title': {'text':'Plots of {} logged price'.format(insert)}})
    fig.show()

# producing the graph
log_create_hist_and_line(y['log_Close'])

test_stationarity(y['log_Close'])


# =============================================================================
# 
# =============================================================================
y.index = y.index.to_period('D')

model = ARMA(y['diff'], order=(1,1))
results = model.fit()

results.summary()


# =============================================================================
# Splitting the data in Training and Test Data
# =============================================================================
# Train data - 780%
df_train = df_new[:int(0.80*(len(df_new)))]
# Test data - 20%
df_test = df_new[int(0.80*(len(df_new))):]

def training_and_test_plot(): 
    # creating a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        name = 'Training Set')
    
    trace2 = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    fig.update_layout({'title': {'text':'Training and Test Set Plot'}})
    fig.show()
    
training_and_test_plot()

"""
TO FIX
- ticks
- create a slider for time period?
"""