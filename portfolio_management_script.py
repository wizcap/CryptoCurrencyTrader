import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pickle

from data_input_processing import Data
from porftfolio_functions import build_price_arrays, calculate_portfolio_value
from machine_learning import tensorflow_cnn_fitting

if __name__ == "__main__":
    TIME_LAG = 31
    TIME_LENGTH = 30 * 24 * 60 * 60
    OFFSET = 0
    current_time = time.time()

    running_portfolio_value = 1

    strategy_dictionary = {}

    input_data = []

    ticker_list = [
        "USDT_BTC",
        "USDT_ETH",
        "USDT_XRP",
        "USDT_LTC",
        "USDT_XMR",
        "USDT_STR",
        "USDT_NXT",
        "USDT_DASH"]

    for ticker in ticker_list:
        print('Downloading ', ticker, ' data')
        input_data.append(
            Data(ticker, 1800, True, start=current_time - TIME_LENGTH - OFFSET, end=current_time - OFFSET))
        print('Complete')

    price_array, price_array_training, liquidation_factor = build_price_arrays(
        input_data,
        time_lag=TIME_LAG)

    ticker_list.append('USDT')

    initial_idx = int(len(price_array) / 2)

    idx = initial_idx

    run_length = len(price_array) - idx

    cnn_kwargs = {}

    plot_time = []
    plot_prices = []
    plot_portfolio_value = []
    other_prices = []

    f, ax = plt.subplots(1, 1)
    f2, ax2 = plt.subplots(1, 1)

    while idx < len(price_array):
        train_indices = np.linspace(
            0,
            idx - 1,
            idx - 1,
            dtype=int)
        test_indices = [idx]

        if idx == int(len(price_array) / 2):

            cnn_kwargs = {
                'retrain': 80,
            }
        else:
            cnn_kwargs = {
                'retrain': 5,
                'load_net': 'model.h5'
            }

        fitting_dictionary = tensorflow_cnn_fitting(
            train_indices,
            test_indices,
            [],
            price_array_training,
            price_array,
            **cnn_kwargs)

        test_prices = price_array[test_indices, :]

        portfolio_value, cum_log_return = calculate_portfolio_value(
            fitting_dictionary['fitted_strategy_score'],
            test_prices,
            liquidation_factor[test_indices, :])

        final_portfolio_value = portfolio_value[-1]
        running_portfolio_value *= final_portfolio_value

        print('Backtest step ', idx - initial_idx, ' of ', run_length)
        print('Run profit fraction: ', final_portfolio_value - 1)
        print('Offset tests cumulative profit fraction:', running_portfolio_value - 1)
        print()

        if plot_time:
            plot_time.append(plot_time[-1] + 1/48.0)
        else:
            plot_time.append(1/48.0)

        plot_prices = np.cumprod(price_array[initial_idx:(idx+1), 0], axis=0)

        other_prices = np.cumprod(price_array[initial_idx:(idx+1), :], axis=0)

        plot_portfolio_value.append(running_portfolio_value)

        ax.clear()
        ax.plot(plot_time, plot_portfolio_value, label='Portfolio value')
        ax.plot(plot_time, plot_prices, label='Reference Asset Price')
        ax.set(xlabel='Time (days)', ylabel='Fractional Value')
        ax.set_title('Fractional Portfolio Value')
        ax.legend(handles=ax.lines, labels=["Portfolio Value", "Reference Asset Price"])

        ax2.clear()
        for price, label in zip(other_prices.T, ticker_list):
            ax2.plot(plot_time, price, label=label)
        ax2.set(xlabel='Time (days)', ylabel='Fractional Value')
        ax.set_title('Reference Asset Prices')
        ax2.legend()

        plt.pause(0.05)

        idx += 1

pickle.dump({'portfolio_value': plot_portfolio_value, 'prices': plot_prices}, open("backtest_data.pkl", "wb"))
plt.show()




