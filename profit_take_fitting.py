from trading_strategy_fitting import import_data
from matplotlib import pyplot as plt
import numpy as np
import random


def random_search(data, strategy_dictionary_local, n_iterations):
    n = 0
    portfolio_value_best = np.zeros(len(data.close))
    cash_value_best = np.zeros(len(data.close))

    while n < n_iterations:
        strategy_dictionary_local['profit_take_up']\
            = random_from_series_range(strategy_dictionary_local['profit_takes_up'])

        portfolio_value, cash_value, num_trades = scalper_returns(data, strategy_dictionary_local)

        n += 1

        if portfolio_value[-1] * cash_value[-1] > portfolio_value_best[-1] * cash_value_best[-1]:
            portfolio_value_best = portfolio_value
            profit_take_best_up = strategy_dictionary_local['profit_take_up']
            num_trades_best = num_trades
            cash_value_best = cash_value

    strategy_dictionary_local['profit_take_up'] = profit_take_best_up
    strategy_dictionary_local['n_trades'] = num_trades_best

    return strategy_dictionary_local


def offset_validation(strategy_dictionary_local):
    print "Final Portfolio Value    Cash Value    Number of Trades    Cumulative"
    cum_val = 1

    for offset in strategy_dictionary_local['offsets']:
        strategy_dictionary_local['offset'] = offset + strategy_dictionary['training_days']

        strategy_dictionary_local['n_days'] = strategy_dictionary['training_days']
        data, data_2 = import_data(strategy_dictionary_local)
        strategy_dictionary_local = random_search(data, strategy_dictionary_local, n_iterations=1e3)

        strategy_dictionary_local['offset'] -= strategy_dictionary['training_days']
        strategy_dictionary_local['n_days'] = strategy_dictionary['validation_days']
        data, data_2 = import_data(strategy_dictionary_local)
        portfolio_value, cash_value, num_trades = scalper_returns(data, strategy_dictionary_local)

        cum_val *= portfolio_value[-1]
        print portfolio_value[-1], '   ', cash_value[-1], '   ', num_trades, '    ', cum_val
        print strategy_dictionary
        plt.figure(1)
        plt.plot(portfolio_value * data.close[0])
        plt.plot(data.close)
        plt.show()


def random_from_series_range(series):
    return random.uniform(np.max(series), np.min(series))


def scalper_returns(data, strategy_dictionary_local):
    portfolio_value_local = np.zeros(len(data.close))
    portfolio_value_local[0] = 1
    cash_value_local = np.zeros(len(data.close))
    cash_value_local[0] = 0
    num_trades_local = 0
    change_position = True

    for i in range(len(data.close)):
        if change_position:
            up_limit = (1 + strategy_dictionary_local['profit_take_up']) * data.close[i]

            change_position = False

        if i > 0:
            portfolio_value_local[i] = portfolio_value_local[i - 1]
            if data.high[i] > up_limit:
                change_position = True
                num_trades_local = num_trades_local + 1
                portfolio_value_local[i] = portfolio_value_local[i - 1]\
                                           * (1 + 0.5 * strategy_dictionary_local['profit_take_up']
                                              - strategy_dictionary_local['bid_ask_spread']
                                              - strategy_dictionary_local['transaction_fee'])

                cash_value_local[i] = cash_value_local[i - 1] + 0.5 * strategy_dictionary_local['profit_take_up']

            else:
                cash_value_local[i] = cash_value_local[i - 1]
                change_position = False

    return portfolio_value_local, cash_value_local, num_trades_local


def currency_value_final(data):
    return data.close[-1] / data.close[0]


if __name__ == '__main__':
    strategy_dictionary = {
        'scraper_currency_1': 'BTC',
        'scraper_currency_2': 'BTC',
        'trading_currencies': ['USDT', 'BTC'], #['USDT', 'BTC'],
        'ticker_1': 'USDT_BTC', #'USDT_BTC',
        'ticker_2': 'BTC_ETH',
        'candle_size': 300,
        'n_days': 1000,
        'offset': 0,
        'offsets': np.linspace(120, 1000, 100),
        'training_days': 360,
        'validation_days': 360,
        'bid_ask_spread': 0.004,
        'transaction_fee': 0.0025,
        'web_flag': 'True',
        'filename1': "USDT_BTC.csv",
        'filename2': "BTC_ETH.csv",
        'profit_takes_up': np.linspace(0, 0.25, 25),
        'scraper_page_limit': 0,
    }

    offset_validation(strategy_dictionary)
