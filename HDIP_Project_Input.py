

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
# Scraping the Top 20 Cryptocurrencies off Yahoo Finance
# =============================================================================
# read the CSV file
df_cryptolist = pd.read_csv('cryptolist.csv')

# dropping unnecessary columns 
df_cryptolist = df_cryptolist[['Symbol','Name','Market Cap']]

# =============================================================================
# getting a list from the table
# =============================================================================
cryptolist = [] 
def get_crypto(data):
    index = 0
        
    while index < len(data.iloc[:,0]):
        try:
            for crypto in data.iloc[:,0]:
                cryptolist.append(str(crypto))
                index += 1
                
        except:
            index = len(data.iloc[:,0])
            break
    return cryptolist
        

get_crypto(df_cryptolist)
    
     
# ============================================================================
# Trying to create an error message    
# ============================================================================
def insert():   
    global crypto_name
    global insert
    while True:
        print('--------------------------------------------')
        print('Top 10 Cryptocurrencies')
        print('--------------------------------------------')
        print(df_cryptolist.head(len(df_cryptolist)))
        try:
            insert = str(input('What cryptocurrency would you like to try out? Please select a symbol: ')).upper()
            #found = df_cryptolist[df_cryptolist['Symbol'].str.contains(insert)]
            crypto_name = str(df_cryptolist[df_cryptolist['Symbol'].str.contains(insert)].iloc[:,1]).split(' ')[4]
            
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        
        if not insert in cryptolist:
            print('Sorry. You did not select an available symbol or you misspelled the symbol')
                
        else:
            print('--------------------------------------------')
            print('You have selected: ', insert)
            print('The name of this cryptocurrency is: ', crypto_name)
            print('--------------------------------------------')
            break

insert()

# =============================================================================
# Collecting info from Yahoo Finance and creating a dataset for that cryptocurrency
# =============================================================================
def create_df(x):

    # =============================================================================
    # Creating a new dataset
    # =============================================================================
    
    global df
    
    start = "2009-01-01"
    end = dt.datetime.now()
    short_sma = 20
    long_sma = 50
    
    # creating a dataset for selected cryptocurrency 
    df = yf.download(x, start, end,interval = '1d')
    df = pd.DataFrame(df.dropna(), columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume'])
    df['short_SMA'] = df.iloc[:,1].rolling(window = short_sma).mean()
    df['long_SMA'] = df.iloc[:,1].rolling(window = long_sma).mean()
    df['Name'] = crypto_name
    print('--------------------------------------------')
    print(crypto_name, '- Full Database')
    print('--------------------------------------------')
    print(df.head())
    print('--------------------------------------------')
    
    # =============================================================================
    # Assigning the target variable
    # =============================================================================
    global y
    
    y = pd.DataFrame(df['Close'], columns = ['Close'])
    y.sort_index(inplace = True)
        # Creating a new variable, examining the difference for each observation
    y['diff'] = y['Close'].diff()
    # dropping the first na (because there is no difference)
    y = y.dropna()
    # logging the target varialbe due to great variance
    y['log_Close'] = np.log(y['Close'])

    # logging the target varialbe due to great variance
    y['sqrt_Close'] = np.sqrt(y['Close'])
    print('--------------------------------------------')
    print(crypto_name, '- Target Variable')
    print('--------------------------------------------')
    print(y.head())
    print('--------------------------------------------')

  
create_df(insert)

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================

def create_graphs(data):

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,  subplot_titles=[
            'Price and Moving Averages of {}'.format(str(crypto_name)),
            'Candlesticks for {}'.format(str(crypto_name)),
            'Volume of {}'.format(str(crypto_name))])
    # Lineplots of price and moving averages
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['Close'],
                            name = data, 
                            mode='lines',
                            customdata = df['Name'], 
                            # corrects hovertemplate labels!
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Closing Price: %{y:$,.2f}<br>" +
                                            "<extra></extra>",
                            line = dict(color="black")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df.index,
                             y = df['short_SMA'],
                             name = 'Short SMA',
                             mode = 'lines', customdata = df['Name'], 
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Short Moving Average Price: %{y:$,.2f}<br>" +
                                            "<extra></extra>",
                             line = dict(color="red")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df.index,
                             y = df['long_SMA'],
                             name = 'Long SMA',
                             mode = 'lines',customdata = df['Name'], 
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Long Moving Average Price: %{y:$,.2f}<br>"+
                                            "<extra></extra>",
                             line = dict(color="green")), row = 1, col = 1)
    # Candlestick
    fig.add_trace(go.Candlestick(x = df.index,
                    open = df['Open'],
                    high = df['High'],
                    low = df['Low'],
                    close = df['Close'],
#                    customdata = df['Name'], 
#                    hovertemplate="<b>%{customdata}</b><br><br>" +
#                                    "Date: %{x|%d %b %Y} <br>" +
#                                     "Open Price: %{open:$,.2f}<br>" +
#                                     "High Price: %{high:$,.2f}<br>" +
#                                     "Low Price: %{low:$,.2f}<br>"  +
#                                     "Close Price: %{close:$,.2f}<br>" ,
                    name = 'market data'), row = 2, col = 1)
    # Barplot of volume 
    fig.add_trace(go.Bar(x = df.index,
                    y = df['Volume'],
                    name = 'Volume',
                    # corrects hovertemplate labels!
                    customdata = df['Name'],  
                    hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Volume: %{y:,.}<br>" +
                                    "<extra></extra>",
                    marker = dict(color="black", opacity = True)), row = 3, col = 1)
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)),
            #changes the size of the plots
            autosize=False, width = 1800, height = 2000
            )
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
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    fig.update_yaxes(tickprefix = '$', tickformat = ',.',row = 2, col = 1)
    fig.update_xaxes(rangeslider= {'visible': False}, row = 2, col = 1)
    fig.update_xaxes(rangeselector= {'visible' :False}, row = 2, col = 1)
    fig.update_xaxes(rangeselector= {'visible': False}, row = 3, col = 1)
    #Show
    fig.show()


# calling the graph
create_graphs(insert)



"""
TO FIX
- Candlestick
- create a slider for short and long SMA
"""


# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================

def create_hist_and_box(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Histogram of {} price'.format(crypto_name),
                                        'Box plot of {} price'.format(crypto_name)],
                        x_title = 'US Dollars')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = round(len(df) / 20),
#                               customdata = df['Name'],
#                               hovertemplate="<b>%{customdata}</b>"
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = data, name = 'Boxplot',
                         customdata = df['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Closing Price: %{x:$,.2f}<br>"+
                                    "<extra></extra>"), row=2, col=1)

    fig.update_layout(title = 'Plots of {} price'.format(crypto_name))
    fig.update_xaxes(tickprefix = '$', tickformat = ',.')
    fig.show()

# calling the graph
create_hist_and_box(df['Close'])


"""
TO FIX
- ticks in boxplot and histogram

"""


# =============================================================================
# Decomposition
# =============================================================================
## Viewing the seasonal decompose of the target variable
#rcParams['figure.figsize'] = 18, 8
#decomposition = sm.tsa.seasonal_decompose(y.asfreq('MS'), model='multiplicative')
#fig = decomposition.plot()
#plt.show()

## =============================================================================
## Adfuller Test
## =============================================================================
# creating a function to run an adfuller-dickey test on target data
def adfuller_test(data):
    print ('Results of Dickey-Fuller Test for {}:'.format(insert))
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adfuller_test(y['diff'])

# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
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
                                customdata = df['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Closing Price: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="blue")))
    fig.add_trace(go.Scatter(x = timeseries.index,
                                y = rolmean,
                                name = 'Rolling Mean', 
                                mode='lines',
                                customdata = df['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Rolling Mean Price: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="red")))
    fig.add_trace(go.Scatter(x = y.index,
                                y = rolstd,
                                name = 'Rolling Std', 
                                mode='lines',
                                customdata = df['Name'],
                                hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Rolling Std: %{y:$,.2f}<br>" +
                                    "<extra></extra>",
                                line = dict(color="black")))
    # Add titles
    fig.update_layout(
            title = 'Rolling Mean & Standard Deviation of {}'.format(crypto_name),
            yaxis_title = 'US Dollars',
            yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    #Show
    fig.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test of {}: '.format(crypto_name))
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


test_stationarity(y['Close'])

#test_stationarity(y['log_Close'])
#
#test_stationarity(y['sqrt_Close'])


# 0th element is test statistic (-1.34)
# More negative means more likely to be stationary
# 1st element is p-value: (0.60)
# If p-value is small → reject null hypothesis. Reject non-stationary.
# 4th element is the critical test statistics



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
# Exploring the difference
# =============================================================================
# creating the plot to examine the difference
def diff_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index,
                            y = data['diff'],
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"))
    # Add titles
    fig.update_layout(
        title = 'Price of {}'.format(insert),
        yaxis_title = 'US Dollars',
        yaxis_tickprefix = '$', yaxis_tickformat = ',.')
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
# Exploring the data
# =============================================================================

# function to create a histogram and lineplot of logged data
def log_create_hist_and_line(data):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Histogram of {} logged price'.format(crypto_name),
                                        'Line plot of {} logged price'.format(crypto_name)])
    # 1. Histogram
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = 100), row=1, col=1)
    # 1. Boxplot 
    fig.add_trace(go.Scatter(x = data.index , y = data, mode = 'lines', name = 'lineplot',
                             customdata = df['Name'],
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Logged Closing Price: %{y:$,.0f}<br>"+
                                            "<extra></extra>"), row=2, col=1)
    fig.update_layout({'title': {'text':'Plots of {} logged price'.format(crypto_name)}})

    fig.show()

# producing the graph
log_create_hist_and_line(y['log_Close'])

test_stationarity(y['log_Close'])



# =============================================================================
# ARIMA Models
# =============================================================================



model = ARMA(y['diff'], order=(1,1))
results = model.fit()

results.summary()


# =============================================================================
# Splitting the data in Training and Test Data
# =============================================================================
def create_train_and_test(x):
    global df_train 
    global df_test
    # Train data - 80%
    df_train = x[:int(0.80*(len(df)))]
    print('============================================================')
    print('Training Set')
    print('============================================================')
    print(df_train.head())
    # Test data - 20%
    df_test = x[int(0.80*(len(df))):]
    print('============================================================')
    print('Test Set')
    print('============================================================')
    print(df_test.head())

    
create_train_and_test(df)

def training_and_test_plot(): 
    # creating a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df_train.index,
        y = df_train['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Training Set')
    
    trace2 = go.Scatter(
        x = df_test.index,
        y = df_test['Close'],
        name = 'Test Set',
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        yaxis="y1")
    
    data = [trace1, trace2]
    fig = go.Figure(data = data)
    
    fig.update_layout({'title': {'text':'Training and Test Set Plot'}},
                      yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.show()
    
training_and_test_plot()

"""
TO FIX
- ticks
- create a slider for time period?
"""