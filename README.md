# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency trading strategies using machine and deep learning on price, hash rates, google trends and scraped forum sentiment data (using natural language processing). Based on Sklearn and Tensorflow.
The script is inspired by both the pytrader project https://github.com/owocki/pytrader, and the auto-sklearn project https://automl.github.io/auto-sklearn/stable/. 

## Disclaimer
The information in this repository is provided for information purposes only. The Information is not intended to be and does not constitute financial advice or any other advice, is general in nature and not specific to you.

## Settings
The script settings are controlled by the strategy dictionary variable, for each fitting you need to supply:  
'trading_currencies' - the cryptocurrencies for the trader to trade between;  
'ticker_#' - the poloniex tickers to pull data from;  
'scraper_currency_#' - the currency with which to scrape forums for sentiment analysis;  
'candle_size' - the candlestick chart candle size to use for input data;  
'n_days' - total number of days of data to use as input;  
'offset' - offset from end of data to pull data from;  
'bid_ask_spread' - trading currency bid-ask spread;  
'transaction_fee' - transaction fee per trade;  
'train_test_validation_ratios' - fraction of data to use for training, testing and validation;  
'output_flag' - supply text output for every machine learning model that is fit;  
'plot_flag' - supply graphical output for every machine learnign model that is fit;  
'ml_iterations' - number of random search iterations for hyperparameter fitting;  
'target_score' - select scoring system to generate targets for fitting;  
''windows' - windows to fit exponential moving averages over;  
'web_flag' - pull price data from web of csv;  
'filename_#' - file to use for candlestick csv if required;  
'scraper_page_limit' - number of web pages to scrape per currency per web source;  

## Input Data
Minor changes were made to the Poloniex API python wrapper which is included in the repository https://github.com/s4w3d0ff/python-poloniex. Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format along with volume data.
A series of non-price data are also provided, hash rates, google trends data are pulled using the pytrends pseudo API and the web scraping and Reddit and bitcointalk forum data is supplied using the natural language processing sentiment analysis package from here https://github.com/llens/CryptocurrencyWebScrapingAndSentimentAnalysis. 

Alternatively, price and volume data can be supplied in the form of .csv files by including them in the working directory, setting web_flag as false and supplying the filenames as filename1 and filename2, (filename1 will be the currency pair used for trading).


### Technical Indicators - Training Inputs
A series of technical indicators are calculated and provided as inputs to the machine learning optimisation, exponential moving averages, exponential moving volatilities and exponential moving volumes over a series of windows. A Kalman filter is also provided as an input.


### Training Targets - Strategy Score
An ideal trading strategy is generated based on past data, every candlestick is given a score which represents the potential profit or loss before the next price reversal exceeding the combined transaction fee and bid ask spread. This minimum price reversal is represented by Δp in the diagram below.
![Alt text](strategyscore.jpg?raw=true "Optional Title")

### Strategy Generation
A buy threshold and sell threshold are selected which maximise profit based on the score returned for the training data, where a sell or buy signal is generated if the respective threshold is crossed.

## Machine Learning Meta-fitting and Hyper Parameter Optimisation
The machine learning optimisation is based on a two layer random search, as outlined in the diagram below. The meta-fitting selects a machine learning and preprocessing pair, the selected machine learning model is then optimised using a second random grid search to fit the hyperparameters for that particular machine learning model. One option (solid box) is selected for each parameter (inner dashed box) for each fitting stage (outer dashed box). (Without GPU support the tensorflow fitting may take a long time!)
![Alt text](ML_Flowchart.png?raw=true "Optional Title")

## Example results
With none of the different automated machine learning optimisation strategies was I able to get a set of fitting parameters which was consistently profitable at multiple offsets. Some of the offsets would be profitable an example is included below.
![Alt text](Fitting_example.png?raw=true "Optional Title")

## Validation
In order to estimate the amount of overfitting, a series of offset hyperparameter fittings are performed. If the trading strategy is not overfitted, fitting should be approximately consistent across at all offsets in terms of profit fraction and fitting error.

## To Do
With none of the different automated machine learning optimisation strategies was I able to get a set of fitting parameters which was consistently profitable at multiple offsets.
* Add none price data.

# Setup
To run as a standalone script add a file named API_settings.py containing your poloniex API, google and reddit API login:
poloniex_API_key = ""  
poloniex_API_secret = ""  
google_username = ""  
google_password = ""  
client_id = ""  
client_secret = ""  
user_agent = 'Python Scraping App'  

## Python 2.7 + Tensorflow + MiniConda
https://conda.io/docs/installation.html    
https://conda.io/docs/_downloads/conda-cheatsheet.pdf   
## OSX / linux   
conda create -n tensorflow-p2 python=2.7   
source activate tensorflow-p2    
conda install numpy pandas matplotlib tensorflow jupyter notebook scipy scikit-learn nb_conda     
conda install -c auto multiprocessing statsmodels arch   
pip install arch polyaxon   

## Windows
conda create -n tensorflow-p2 python=2.7   
activate tensorflow-p2   
conda install numpy pandas matplotlib tensorflow jupyter notebook scipy scikit-learn nb_conda    
conda install -c auto multiprocessing statsmodels arch    
pip install arch polyaxon   


