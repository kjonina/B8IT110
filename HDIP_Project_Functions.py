

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
from statsmodels.tsa.stattools import kpss, adfuller
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
    df['Name'] = crypto_name
#    print('============================================================')
#    print(crypto_name, '- Full Dataset')
#    print('------------------------------------------------------------')
#    print(df.head())
#    print('------------------------------------------------------------')
#    print(crypto_name, 'Full Dataset - Column Names')
#    print(df.columns)
    print('============================================================')
    
    # preparing data from time series analysis
    # eliminating any NAs - in most cryptocurrencies there are 4 days missing
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    print(crypto_name)
    print('Nan in each columns' , df.isna().sum())
    df = df.bfill()
    print('Nan in each columns' , df.isna().sum())
    df = df.dropna()

#    # write to csv
#    df.to_csv(r"df.csv", index =  True)
    
    # =============================================================================
    # Assigning the target variable
    # =============================================================================

    
def create_y(x):
    
    global y
    
    y = pd.DataFrame(df['Close'], columns = ['Close'])
    y.sort_index(inplace = True)
    y['Name'] = crypto_name
    
    # examining the pct_change
    y['Close Percentage Change'] = y['Close'].pct_change(1)
    
    # Creating a new variable, examining the difference for each observation
    y['diff'] = y['Close'].diff()

    # logging the target varialbe due to great variance
    y['log_Close'] = np.log(y['Close'])
    
    # Creating a new variable, examining the difference for each observation
    y['log_Close_diff'] = y['log_Close'].diff()
    
    y['Logged Close Percentage Change'] = y['log_Close'].pct_change(1)
    
    # dropping the first na (because there is no difference)
    y = y.dropna()
    

#    # write to csv
#    y.to_csv(r"y.csv", index =  True)

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



def candlestick_moving_average():

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
            'Price and Moving Averages of {}'.format(str(crypto_name)),
            'Volume of {}'.format(str(crypto_name))])

    trace1 = go.Candlestick(
        x = df.index,
        open = df["Open"],
        high = df["High"],
        low = df["Low"],
        close = df["Close"],
        name = crypto_name)

    data = [trace1]

    for i in range(5, 201, 5):

        sma = go.Scatter(
            x = df.index,
            y = df["Close"].rolling(i).mean(), # Pandas SMA
            name = "SMA" + str(i),
            line = dict(color = "#3E86AB"),
            customdata = df['Name'],
            hovertemplate="<b>%{customdata}</b><br><br>" +
                        "Date: %{x|%d %b %Y} <br>" +
                        "Simple Moving Average Price: %{y:$,.2f}<br>",
            opacity = 0.7,
            visible = False,
        )

        data.append(sma)

    sliders = dict(

        # GENERAL
        steps = [],
        currentvalue = dict(
            font = dict(size = 16),
            prefix = "SMA: ",
            xanchor = "left",
        ),

        x = 0.15,
        y = 0,
        len = 0.85,
        pad = dict(t = 0, b = 0),
        yanchor = "bottom",
        xanchor = "left",
    )

    for i in range((200 // 5) + 1):

        step = dict(
            method = "restyle",
            label = str(i * 5),
            value = str(i * 5),
            args = ["visible", [False] * ((200 // 5) + 1)],
        )

        step['args'][1][0] = True
        step['args'][1][i] = True
        sliders["steps"].append(step)



    layout = dict(

        title = 'Price of {}'.format(str(crypto_name)),

        # ANIMATIONS
        sliders = [sliders],
        xaxis = dict(

            rangeselector = dict(
                activecolor = "#888888",
                bgcolor = "#DDDDDD",
                buttons = [
                            dict(count = 7, step = "day", stepmode = "backward", label = "1W"),
                            dict(count = 1, step = "month", stepmode = "backward", label = "1M"),
                            dict(count = 3, step = "month", stepmode = "backward", label = "3M"),
                            dict(count = 6, step = "month", stepmode = "backward", label = "6M"),
                            dict(count = 1, step = "year", stepmode = "backward", label = "1Y"),
                            dict(count = 2, step = "year", stepmode = "backward", label = "2Y"),
                            dict(count = 5, step = "year", stepmode = "backward", label = "5Y"),
                            dict(count = 1, step = "all", stepmode = "backward", label = "MAX"),
                            dict(count = 1, step = "year", stepmode = "todate", label = "YTD"),
                ]
            ),

        ),
        yaxis = dict(
            tickprefix = "$",
            type = "linear",
            domain = [0.25, 1],
        ),

    )



    fig = go.Figure(data = data, layout = layout)
    
    #Show
    fig.show()


# =============================================================================
# Analysing the Histogram and Boxplot for crypto
# =============================================================================

def create_hist_and_box(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Histogram of {} price'.format(crypto_name),
                                        'Box plot of {} price'.format(crypto_name)],
                        x_title = 'US Dollars')
    # 1.Histogram
    fig.add_trace(go.Histogram(x = data, name = 'Histogram', nbinsx = round(len(data) / 20),
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
    fig.add_trace(go.Histogram(x = y['Close Percentage Change'], name = 'Histogram', nbinsx = round(len(y) / 20),
                               ), row=1, col=1)
    
    #2. Boxplot 
    fig.add_trace(go.Box(x = y['Close Percentage Change'], name = 'Boxplot',
                         customdata = y['Name'],
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

# =============================================================================
# Creating a plot with analysis and rolling mean and standard deviation
# =============================================================================
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 365).mean()
    rolstd = timeseries.rolling(window = 365).std()

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
#  daily, monthly, annual returns
# =============================================================================

def returns():
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[
            'Closing Price of {}'.format(str(crypto_name)),
            'Daily Return of {}'.format(str(crypto_name)),
            'Monthly Return of {}'.format(str(crypto_name)),
            'Annual Return of {}'.format(str(crypto_name))])
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['Close'],
                            mode='lines',
                            customdata = df['Name'], name = 'Closing Price',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Closing Price: %{y:$,.2f}<br>"+
                                            "<extra></extra>"), row = 1, col = 1)

    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['daily_return'], 
                            mode='lines',
                            customdata = df['Name'], name = 'Daily Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Daily Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 2, col = 1)

    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['monthly_return'],
                            mode='lines',
                            customdata = df['Name'], name = 'Monthly Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Monthly Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 3, col = 1)
    
    fig.add_trace(go.Scatter(
                            x = df.index,
                            y = df['annual_return'],
                            mode='lines',
                            customdata = df['Name'], name = 'Annual Return',
                            hovertemplate="<b>%{customdata}</b><br><br>" +
                                            "Date: %{x|%d %b %Y} <br>" +
                                            "Annual Return: %{y:,.0%}<br>"+
                                            "<extra></extra>"), row = 4, col = 1)
    
    # Add titles
    fig.update_layout( 
            title = 'Price of {}'.format(str(crypto_name)))
    fig['layout']['yaxis1']['title']='US Dollars'
    fig['layout']['yaxis2']['title']='US Dollars'
    fig['layout']['yaxis3']['title']='US Dollars'
    fig['layout']['yaxis4']['title']='US Dollars'
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
    fig.update_xaxes(rangeslider= {'visible':False}, row=3, col=1)
    fig.update_xaxes(rangeslider= {'visible':False}, row=4, col=1)

    fig.update_xaxes(rangeselector= {'visible':False}, row=2, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=3, col=1)
    fig.update_xaxes(rangeselector= {'visible':False}, row=4, col=1)    

    fig.update_yaxes(tickprefix = '$', tickformat = ',.', row = 1, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 2, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 3, col = 1)
    fig.update_yaxes(tickformat = ',.0%', row = 4, col = 1)

    #Show
    fig.show()

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
# Splitting the data in Training and Test Data
# =============================================================================
def create_train_and_test():
    global df_train 
    global df_test
    # Train data - 80%
    df_train = y[:int(0.80*(len(y)))]
    
    print('============================================================')
    print('{} Training Set'.format(crypto_name))
    print('============================================================')
    print(df_train.head())
    print('Training set has {} rows and {} columns.'.format(*df_train.shape))
    # Test data - 20%
    df_test = y[int(0.80*(len(y))):]
    print('============================================================')
    print('{} Test Set'.format(crypto_name))
    print('============================================================')
    print(df_test.head())
    print('Test set has {} rows and {} columns.'.format(*df_test.shape))
    

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


# =============================================================================
# creating important functions for Time Series Analysis 
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