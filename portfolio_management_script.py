import time
from porftfolio_functions import build_price_arrays, calculate_portfolio_value
from machine_learning import tensorflow_cnn_fitting
from data_input_processing import Data, train_validation_test_indices
import numpy as np


if __name__ == "__main__":
    TIME_LAG = 50
    TIME_LENGTH = 1 * 365 * 24 * 60 * 60
    OFFSET = 365 * 60 * 60
    current_time = time.time()

    running_portfolio_value = 1
    for OFFSET in np.linspace(0, 1, 2, 3) * 30 * 24 * 60 * 60:

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
            temp_data = Data(ticker, 1800, True, start=current_time - TIME_LENGTH - OFFSET, end=current_time - OFFSET)
            input_data.append(Data(ticker, 1800, True, start=current_time - TIME_LENGTH - OFFSET, end=current_time - OFFSET))

        price_array, price_array_training = build_price_arrays(
            input_data,
            time_lag=TIME_LAG)

        #train_indices, validation_indices, test_indices = train_validation_test_indices(
        #    price_array_training,
        #    [0.9, 0.05, 0.0001])

        train_indices = np.linspace(0, len(price_array_training)-100, len(price_array_training)-100, dtype=int)
        test_indices = np.linspace(len(price_array_training) - 100, len(price_array_training) - 5, 95, dtype=int)
        validation_indices = np.linspace(len(price_array_training) -5, len(price_array_training)-1, 5, dtype=int)

        fitting_dictionary = tensorflow_cnn_fitting(
            train_indices,
            test_indices,
            validation_indices,
            price_array_training)

        portfolio_value, cum_log_return = calculate_portfolio_value(
            fitting_dictionary['fitted_strategy_score'],
            price_array[test_indices, :])

        final_portfolio_value = portfolio_value[-1]
        running_portfolio_value *= final_portfolio_value

        print('Run profit fraction: ', final_portfolio_value - 1)
        print('Offset tests cumulative profit fraction:', running_portfolio_value - 1)

        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(portfolio_value)
        plt.xlabel('Time (30 minute steps)')
        plt.ylabel('Fractional Portfolio Value')
        plt.show()

