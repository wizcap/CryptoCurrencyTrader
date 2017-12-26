# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency trading strategies using machine learning.
The script is inspired by both the pytrader project https://github.com/owocki/pytrader, and the auto-sklearn project https://automl.github.io/auto-sklearn/stable/. 

# Disclaimer
The information in this repository is provided for information purposes only. The Information is not intended to be and does not constitute financial advice or any other advice, is general in nature and not specific to you.

## Project Status
The project is still a work in progress, I found a bug in the iteration of the project that was indicating profitability when it wasn't present. I am now getting stable if unprofitable performance.

![Alt text](Fitting_example.png?raw=true "Optional Title") 

The results show profitability in some situations and not others, I believe this is a result of overfitting combined with an uncertain detection of market regime. I detect this overfitting by running the same fitting heuristic with paramteters profitable at one time at other offset times, I am yet to arrive at a consistently profitable incarnation.

### Future
I am continuing to improve the model, to produce a system which is profitable after fees without overfitting.
I am planning and beginning to implement a deep reinforcement learning system to overhaul the tensorflow component.


## Input Data
Minor changes were made to the Poloniex API python wrapper which is inluded in the repository https://github.com/s4w3d0ff/python-poloniex. Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format along with volume data.
A series of non price data are also provided, google trends data are pulled using the pytrends psuedo API and the web scraping and reddit and bitcointalk forum data is supplied using the natural language processing sentiment analysis package from here https://github.com/llens/CryptocurrencyWebScrapingAndSentimentAnalysis.

Alternatively price and volume data can be supplied in the form of .csv files by including them in the working directory, setting web_flag as false and supplying the filenames as filename1 and filename2, (filename1 will be the currency pair used for trading).


### Technical Indicators - Training Inputs
A series of technical indicators are calculated and provided as inputs to the machine learning optimisation, exponential moving averages, exponential moving volatilities and exponential moving volumes over a series of windows. A kalman filter is also provided as an input.

## Machine Learning Meta-fitting and Hyper Parameter Optimisation
The machine learning optimisation is based on a random search as outlined in the diagram below, with a bayesian optimization based hyperparameter fitting for each algorithm. The meta-fitting selects a machine learning and preprocessing pair, the selected machine learning model is then optimised using a second random grid search to fit the hyperparameters for that particular machine learning model. (Without GPU support the tensorflow fitting may take a long time!)
![Alt text](ML_Flowchart.png?raw=true "Optional Title")

## Validation
In order to estimate the amount of overfitting, a series of offset hyperparameter fittings are performed. If the trading strategy is not overfit, fitting should be approximately consistent across at all offsets in terms of profit fraction and fitting error.

#Setup
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


