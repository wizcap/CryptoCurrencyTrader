# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency trading strategies using machine learning.
The script is inspired by both the pytrader project https://github.com/owocki/pytrader, and the auto-sklearn project https://automl.github.io/auto-sklearn/stable/. 

# Disclaimer
The information in this repository is provided for information purposes only. The Information is not intended to be and does not constitute financial advice or any other advice, is general in nature and not specific to you.

## Project Status
From a machine learning perspective the project has been a success, if exchange fees are ignored it is able to profitably predict price movements, with the sklearn evaluation script. An example trading over approximately a week after training and testing on 43 days of data for XMR-DASH is shown below.

![Alt text](no_fees_fitting_example.png?raw=true "Optional Title") 

These results show profit at multiple time offsets using the same fitting heuristic indicating that the system is not overfitting.

###Future
I am continuing to improve the model, to produce a system which is profitable after fees without overfitting.
I am planning and beginning to implement a deep reinforcement learning system to overhaul the tensorflow component. I will also be using an existing unsupervised hidden markov model as an input to both the tensorflow and sklearn system.


## Input Data
Minor changes were made to the Poloniex API python wrapper which is inluded in the repository https://github.com/s4w3d0ff/python-poloniex. Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format along with volume data.
A series of non price data are also provided, google trends data are pulled using the pytrends psuedo API and the web scraping and reddit and bitcointalk forum data is supplied using the natural language processing sentiment analysis package from here https://github.com/llens/CryptocurrencyWebScrapingAndSentimentAnalysis.

Alternatively price and volume data can be supplied in the form of .csv files by including them in the working directory, setting web_flag as false and supplying the filenames as filename1 and filename2, (filename1 will be the currency pair used for trading).


### Technical Indicators - Training Inputs
A series of technical indicators are calculated and provided as inputs to the machine learning optimisation, exponential moving averages, exponential moving volatilities and exponential moving volumes over a series of windows. A kalman filter is also provided as an input.


### Training Targets - Strategy Score
An ideal trading strategy is generated based on past data, every candlestick is given a score which represent the potential profit or loss before the next price reversal exceeding the combined transaction fee and bid ask spread. This minimum price reversal is represented by Δp in the diagram below.
![Alt text](strategyscore.jpg?raw=true "Optional Title")


## Machine Learning Meta-fitting and Hyper Parameter Optimisation
The machine learning optimisation is based on a random search as outlined in the diagram below, with a bayesian optimization based hyperparameter fitting for each algorithm. The meta-fitting selects a machine learning and preprocessing pair, the selected machine learning model is then optimised using a second random grid search to fit the hyperparameters for that particular machine learning model. (Without GPU support the tensorflow fitting may take a long time!)
![Alt text](ML_Flowchart.png?raw=true "Optional Title")

## Example results
With none of the different automated machine learning optimisation strategies was I able to get a set of fitting parameters which was consistently profitable at multiple offsets. Some of the offsets would be profitable an example is included below.
![Alt text](Fitting_example.png?raw=true "Optional Title")

## Validation
In order to estimate the amount of overfitting, a series of offset hyperparameter fittings are performed. If the trading strategy is not overfit, fitting should be approximately consistent across at all offsets in terms of profit fraction and fitting error.

## To Do
Due to reducing the number of data inputs and allowing a the strategy to continuously change the amount of currency in position, as opposed to simply being in or out, performance is a lot more stable although still not profitable compared to buy and hold (particularly with the excellent performance of bitcoin recently.)

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


