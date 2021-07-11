

"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Task:           Time Series Forecasting of Cryptocurrency
File:           This file is for functions to run the 
"""

# Downloading necessary files
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import mplfinance as mpf 
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
# Reading in files for ease of use
# =============================================================================
# read the CSV file
df_cryptolist = pd.read_csv('df_cryptolist.csv')

# =============================================================================
# getting a list from the table
# =============================================================================
cryptolist = [] 
def get_crypto_df(data):
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
          
# ============================================================================
# Trying to create an error message    
# ============================================================================
def please_choose_crypto():   
    global crypto_name
    global insert
    
    while True:
        print('============================================================')
        print('Top', len(df_cryptolist), 'Cryptocurrencies')
        print('============================================================')
        print(df_cryptolist[['Symbol','Name','Market Cap']].head(len(df_cryptolist)))
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
            print('============================================================')
            print('You have selected: ', insert)
            df_new = df_cryptolist.copy()
            df_new.set_index("Symbol", inplace=True)
            df_new.head()
            print('============================================================')
            print(df_new.loc[insert])
            print('============================================================')
            break

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
    short_sma = 50
    long_sma = 200
    
    # creating a dataset for selected cryptocurrency 
    df = yf.download(x, start, end,interval = '1d')
    df = pd.DataFrame(df.dropna(), columns = ['Open', 'High','Low','Close', 'Adj Close', 'Volume'])
    df['short_SMA'] = df.iloc[:,1].rolling(window = short_sma).mean()
    df['long_SMA'] = df.iloc[:,1].rolling(window = long_sma).mean()
    df['Name'] = crypto_name
    print('============================================================')
    print(crypto_name, '- Full Dataset')
    print('------------------------------------------------------------')
    print(df.head())
    print('------------------------------------------------------------')
    print(crypto_name, 'Full Dataset - Column Names')
    print(df.columns)
    print('============================================================')


#    # write to csv
    df.to_csv(r"df.csv", index =  True)
    
    # =============================================================================
    # Assigning the target variable
    # =============================================================================
    
    
def create_y(x):
    
    global y
    
    
    y = pd.DataFrame(df['Close'], columns = ['Close'])
    y.sort_index(inplace = True)
    
    # examining the pct_change
    y['Close Percentage Change'] = y['Close'].pct_change(1)
    
    # Creating a new variable, examining the difference for each observation
    y['diff'] = y['Close'].diff()

    # logging the target varialbe due to great variance
    y['log_Close'] = np.log(y['Close'])
    
    # Creating a new variable, examining the difference for each observation
    y['log_Close_diff'] = y['log_Close'].diff()
    
    y['Logged Close Percentage Change'] = y['log_Close'].pct_change(1)

    # logging the target varialbe due to great variance
    y['sqrt_Close'] = np.sqrt(y['Close'])
    
    y['Square Root Close Percentage Change'] = y['sqrt_Close'].pct_change(1)
    
    # Creating a new variable, examining the difference for each observation
    y['sqrt_Close_diff'] = y['sqrt_Close'].diff()
    
    # dropping the first na (because there is no difference)
    y = y.dropna()



#    # write to csv
#    y.to_csv(r"y.csv", index =  False)

    print('============================================================')
    print(crypto_name, '- Target Variable')
    print('------------------------------------------------------------')
    print(y.head())
    print('------------------------------------------------------------')
    print('Column Names')
    print(y.columns)
    print('============================================================')

# =============================================================================
# Creating a graph examining the price and moving averages
# =============================================================================
def price_sma_volume_chart():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
            'Price and Moving Averages of {}'.format(str(crypto_name)),
            'Volume of {}'.format(str(crypto_name))])
    # Lineplots of price and moving averages
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['Close'],
                            name = crypto_name, 
                            mode='lines',
                            
                            # corrects hovertemplate labels!
                            customdata = df['Name'], 
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Closing Price: %{y:$,.2f}<br>" +
                                            "<extra></extra>",
                            line = dict(color="black")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df.index,
                             y = df['short_SMA'],
                             name = 'Short SMA',
                             mode = 'lines', 
                             
                             # corrects hovertemplate labels!
                             customdata = df['Name'], 
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Short Moving Average Price: %{y:$,.2f}<br>" +
                                            "<extra></extra>",
                             line = dict(color="red")), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df.index,
                             y = df['long_SMA'],
                             name = 'Long SMA',
                             mode = 'lines',
                             
                             # corrects hovertemplate labels!
                             customdata = df['Name'], 
                             hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Long Moving Average Price: %{y:$,.2f}<br>"+
                                            "<extra></extra>",
                             line = dict(color="green")), row = 1, col = 1)
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
                    marker = dict(color="black", opacity = True)), row = 2, col = 1)
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Volume'
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
    #time buttons 
    fig.update_xaxes(rangeselector= {'visible' :False}, row = 2, col = 1)

    #Show
    fig.show()
#    #writing the graph to html 
#    fig.write_html('first_figure.html', auto_open=True, include_plotlyjs = 'cdn')



def create_candlestick():
    fig = go.Figure()
    
#    hovertext=[]
#    for i in range(len(df.Open)):
#        hovertext.append('Date: '+str(df.index[i])+
#                        '<br>Open: '+str(df.Open[i])+
#                        '<br>High: '+str(df.High[i])+
#                        '<br>Low: '+str(df.Open[i])+
#                         '<br>Close: '+str(df.Close[i]))


    # Candlestick
    fig.add_trace(go.Candlestick(x = df.index,
                    open = df['Open'],
                    high = df['High'],
                    low = df['Low'],
                    close = df['Close'],
#                    text=hovertext,
#                    hoverinfo='text',
                    name = 'market data'))

    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis']['title']='US Dollars'
    
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
    fig.update_yaxes(tickprefix = '$', tickformat = ',.')
    #Show
    fig.show()
#    #writing the graph to html 
#    fig.write_html('first_figure.html', auto_open=True, include_plotlyjs = 'cdn')

"""
TO FIX
- fix hovertemplate for Candlestick
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
    

# creating graph for Close Percentage Change
def create_hist_and_box_pct_change():
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Histogram of {} 1-Day Close Percentage Change'.format(crypto_name),
                                        'Box plot of {} 1-Day Close Percentage Change'.format(crypto_name)],
                        x_title = '1-Day Close Percentage Change')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = y['Close Percentage Change'], name = 'Histogram', nbinsx = round(len(df) / 20),
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = y['Close Percentage Change'], name = 'Boxplot',
                         customdata = df['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "1-Day Percentage Change: %{x:.0%}<br>"+
                                    "<extra></extra>"), row=2, col=1)

    fig.update_layout(title = 'Plots of 1-Day Close Percentage Change for {}'.format(crypto_name))
    fig['layout']['yaxis1']['title'] = '# of Observations'
    fig.update_xaxes(tickformat = '.0%', row = 1, col = 1)
    fig.update_xaxes(tickformat = '.0%', row = 2, col = 1)
    fig.show()
    
    

def logged_create_hist_and_box_pct_change():
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Logged Closing Price - Histogram of {} 1-Day Close Percentage Change'.format(crypto_name),
                                        'Logged Closing Price - Box plot of {} 1-Day Close Percentage Change'.format(crypto_name)],
                        x_title = 'Loogged Price -  1-Day Close Percentage Change')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = y['Logged Close Percentage Change'], name = 'Histogram', nbinsx = round(len(df) / 20),
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = y['Logged Close Percentage Change'], name = 'Boxplot',
                         customdata = df['Name'],
                         hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "1-Day Percentage Change: %{x:.0%}<br>"+
                                    "<extra></extra>"), row=2, col=1)

    fig.update_layout(title = 'Loogged Closing Price - Plots of 1-Day Close Percentage Change for {}'.format(crypto_name))
    fig['layout']['yaxis1']['title'] = '# of Observations'
    fig.update_xaxes(tickformat = '.0%', row = 1, col = 1)
    fig.update_xaxes(tickformat = '.0%', row = 2, col = 1)
    fig.show() 


"""
TO FIX
- hovertemplate in boxplot and histogram
"""

# =============================================================================
# Decomposition
# =============================================================================
def decomposition_plot(y):
    # Viewing the seasonal decompose of the target variable
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y.asfreq('MS'), model='multiplicative')
    fig = decomposition.plot()
    plt.show()

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
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print('============================================================')
    print ('Results of Dickey-Fuller Test for {}: '.format(crypto_name))
    print('============================================================')
    print (dfoutput)

# =============================================================================
# ACF and PACF plots
# =============================================================================

## Create figure
#fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
## Make ACF plot
#plot_acf(y['Close'], lags=30, zero=False, ax=ax1)
## Make PACF plot
#plot_pacf(y['Close'], lags=30, zero=False, ax=ax2)
#plt.show()
#
#
#
## df['sum'] is my time series where i want the pacf of.
#df_pacf = pacf(y['Close'], nlags=300)
#fig = go.Figure()
#fig.add_trace(go.Scatter(
#    x = np.arange(len(df_pacf)),
#    y = df_pacf,
#    name = 'PACF',
#    ))
#fig.update_xaxes(rangeslider_visible = True)
#fig.update_layout(
#    title = "Partial Autocorrelation",
#    xaxis_title = "Lag",
#    yaxis_title = "Partial Autocorrelation",
#    #     autosize = False,
#    #     width = 500,
#         height = 500,
#    )
#fig.show()
#
#from scipy.signal import detrend
#from statsmodels.graphics.tsaplots import plot_acf
#stat_ts = pd.Series(detrend(np.log(y.Close)), index=y.index)
#plot_acf(stat_ts)
#
#from statsmodels.graphics.tsaplots import plot_pacf
#plot_pacf(stat_ts)

# =============================================================================
# Exploring the difference
# =============================================================================
# creating the plot to examine the difference
def diff_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index,
                            y = data,
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"))
    # Add titles
    fig.update_layout(
        title = 'Price of {}'.format(crypto_name),
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
    
# =============================================================================
# Diff and volume plot
# =============================================================================

def create_diff_volume(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=['Differnce of {} price'.format(crypto_name),
                                        'Volume of {}'.format(crypto_name)])
    # 1.Difference
    fig.add_trace(go.Scatter(x = data.index,
                            y = data,
                            name = str(crypto_name), 
                            mode='lines',
                            customdata = df['Name'],
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                    "Date: %{x|%d %b %Y} <br>" +
                                    "Price Volatility: %{y:$,.2f}<br>"+
                                    "<extra></extra>"), row = 1, col =1)
    #2. Volume
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
                    marker = dict(color="black", opacity = True)), row = 2, col = 1)
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='Volume'
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
    #time buttons 
    fig.update_xaxes(rangeselector= {'visible' :False}, row = 2, col = 1)

    #Show
    fig.show()


"""
TO FIX
- hovertemplate in boxplot and histogram
"""


# =============================================================================
# Exploring the logged data
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

# =============================================================================
# ARIMA Models
# =============================================================================



#model = ARMA(y['diff'], order=(1,1))
#results = model.fit()
#
#results.summary()


# =============================================================================
# Splitting the data in Training and Test Data
# =============================================================================
def create_train_and_test():
    global df_train 
    global df_test
    # Train data - 80%
    df_train = y[:int(0.80*(len(df)))]
    
    print('============================================================')
    print('{} Training Set'.format(crypto_name))
    print('============================================================')
    print(df_train.head())
    print('Training set has {} rows and {} columns.'.format(*df_train.shape))
    # Test data - 20%
    df_test = y[int(0.80*(len(df))):]
    print('============================================================')
    print('{} Test Set'.format(crypto_name))
    print('============================================================')
    print(df_test.head())
    print('Test set has {} rows and {} columns.'.format(*df_test.shape))
    

def training_and_test_plot(): 
    # creating a plotly graph for training and test set
    trace1 = go.Scatter(
        x = df_train['Date'],
        y = df_train['Close'],
        customdata = df['Name'],
        hovertemplate="<b>%{customdata}</b><br><br>" +
        "Date: %{x|%d %b %Y} <br>" +
        "Closing Price: %{y:$,.2f}<br>"+
        "<extra></extra>",
        name = 'Training Set')
    
    trace2 = go.Scatter(
        x = df_test['Date'],
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