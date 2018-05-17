# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency portfolio allocations using machine learning.
I was inspired to create this script by both the pytrader project https://github.com/owocki/pytrader, the following paper https://arxiv.org/abs/1706.10059. This implementation differs from the paper itself in that I use the keras/tensorflow backend to implement the portfolio vector memory entirely within a loss function, reducing the complexity of the trained network, by reducing the number of independent input variables.

# Disclaimer
I provide the information in this repository for information purposes only. The Information is not intended to be and does not constitute financial advice or any other advice, is general and not specific to you.

## Project Status
This implementation of the CNN in Keras for the portfolio management problem from this paper https://arxiv.org/abs/1706.10059, shows profitability up until around February 2018. I believe this is a similar issue as it experienced by the original authors of the paper https://github.com/ZhengyaoJiang/PGPortfolio/issues/63, arising due to competition with other deep learning systems and market conditions. I cannot extend this repo beyond the public domain content of the paper due to being bound by non-disclosure agreements.


![Alt text](FittingExample1.png?raw=true "Optional Title")
![Alt text](FittingExample2.png?raw=true "Optional Title")


## Input Data
Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format along with volume data.

### Training Inputs
The Poloniex API provides Open, high, low close prices at a series of prior times to the solver.

## Validation
A series of offset training runs estimate the amount of overfitting. If there is no overfitting, fitting should be approximately consistent across at all offsets when measured by profit fraction and fitting error.

#Setup
To run as a standalone script add a file named API_settings.py containing your poloniex API:
poloniex_API_key = ""
poloniex_API_secret = ""

#TO DO
Requirements and setup instructions.

