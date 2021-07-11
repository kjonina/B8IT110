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

## 29th June 

##### Fixing Graphs
Continuing to put finishing touches on the plotly graphs

##### Django Tutorial
Created a new page on the website for thesis.
Also created a droplet in DigitalOcean to host the site. 
Used Github to download the Django code onto the DigitalOcean server.

## 30th June 
##### Beautiful Soup
Huge progress on Beautiful Soup, managed to parse JSON file, rename column names but I was stuck on how to deal with 'raw' and the 'fmt' in numeric cases.

##### Meeting with Supervisor
Went well. 
- Gave a brief presentation of work so far. 
- Gave feedback on boxplots! Need to change those to make more sense.
- Gave me a small tip on my JSON file problem: pd.json_normalise to get the 'raw' and the 'fmt'. The missing piece of the puzzle.
- Happy to use Github as version control for my project.


## 7th July
##### Sidebar
Got the yahoo Finance to display in the sidebar after placing it in the views and making a table out of it
Also created a save button for the code

Took the whole day to fix these issues so I take this as a win!

Needs to be done:
- error if the ticket is not on the list
- get the input to be saved and converted (?) to python so that it can be used to download the chosen cryptocurrency
- create first graph in the dashboard

## 10th July
##### Button and Dataset
- Got the button to save the response 
- downloaded the dataset from Yahoo Finance.

Needs to be done:
- error if the ticket is not on the list
- create first graph in the dashboard


## 11th July
##### First Graph online
- Got the graphs displayed in Dashboard!!!!!

![price_sma_volume_chart.png](https://github.com/kjonina/B8IT110/blob/main/price_sma_volume_chart.PNG)

-	The function uses ‘df’ and will automatically update. However, at the moment, the users input and the function to create ‘df’ are not linked. Also update hover templates. 



