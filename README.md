# B8IT110
Learning Journal for my HDIP project 

# Learning Outcomes of the HDIP project
I aim to:
- [ ] download Yahoo Finance data
- [ ] use Plotly to design an interactive visualisations such as price chart, apply functions such as slider input, tabs, and display images with ‘plotly’ packages. 
- [ ] build a strong foundation for Machine Learning models such as Time Series (ARIMA and Prophet) to forecast the price of cryptocurrencies
- [ ] build and design an interactive web application with Django in Python language.

## 20th May 2021
Revised Time Series in DataCamp
![Tutorial _Time_Series_Analysis_in_Python.png](https://github.com/kjonina/B8IT110/blob/main/Tutorial%20_Time_Series_Analysis_in_Python.png)

## 17th June 2021
Revised Time Series in DataCamp
![Tutorial_ARIMA_Models_in_Python.png](https://github.com/kjonina/B8IT110/blob/main/Tutorial%20%20-%20ARIMA%20Models%20in%20Python.png)

Had meeting with supervisor

## 18h June 
##### Graphs
- Created several complex grapsh

![sub_plots_line_candle_volume.png](https://github.com/kjonina/B8IT110/blob/main/sub_plots_line_candle_volume.PNG)

![sub_plots_hist_and_box.png](https://github.com/kjonina/B8IT110/blob/main/sub_plots_hist_and_box.PNG)

![training_and_test_graph.png](https://github.com/kjonina/B8IT110/blob/main/training_and_test_graph.PNG)

##### Selection and Functions 
- Created an input variable where the user is asked to type in their own chosen crypto
- Created several functions to build graphs
- Created a sophisticated split to insure that all cryptocurrencies will be split on 80/20 rather than year. 

## 23rd  June 
##### Beautiful Soup

Working with Beautiful Soup to scrape the Yahoo Finance to present the user with a list of possible cryptocurrencies to pick.

Created unittests for it too.

FURTHER: need to extract the info 

##### If not in the list

Also tried to create a condition that is the written ticket is not in the cryptolist, then the user has to resubmit the ticket. 

FURTHER: the insert is saved locally and not globally. Need to get that input out the function. 

##### Django Tutorial

Half-way through a tutorial, need to fix environment on the computer and in the project. 

##### Removal of datasets and freeing RAM space

In my previous code, I had a loop that stated:
- for every crypto in cryptolist, create a df 

From there, the code worked to select the correct Dataframe and create graphs.
To use Adfuller-Dickey test, I created a dataframe with just that one dataframe.

However this meant that there were too many dataframes in the RAM and if the datasets had more data, it could clog up the computer.


## 26th June 
##### Fixing Graphs
Spent some time fixing the graphs so that the y-axis has $, the ticks have better labels, also extract the crypto name from the list 'Bitcoin' and not use the ticket in the graphs. 

Not finished yet.

![training_and_test_graph.png](https://github.com/kjonina/B8IT110/blob/main/training_and_test_graph.PNG)

##### Django Tutorial
Finished huge chunk and can use this for the thesis. 