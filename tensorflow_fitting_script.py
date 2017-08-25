import numpy as np
import random
from trading_strategy_fitting import tic, tensorflow_offset_scan_validation, fit_tensorflow,\
    underlined_output, import_data, input_processing
from strategy_evaluation import output_strategy_results
from data_input_processing import preprocessing_inputs


def random_search(strategy_dictionary_local, n_iterations):
    toc = tic()
    data_local, data_2 = import_data(strategy_dictionary_local)
    fitting_inputs_local, continuous_targets, classification_targets = input_processing(
        data_local, data_2, strategy_dictionary)

    counter = 0
    error = 1e5
    fitting_dictionary_optimum = []
    strategy_dictionary_optimum = []
    fitting_targets_local = []
    while counter < n_iterations:
        counter += 1

        strategy_dictionary['sequence_flag'] = np.random.choice([True, False])

        if strategy_dictionary['sequence_flag']:
            strategy_dictionary_local = randomise_sequence_dictionary_inputs(strategy_dictionary_local)
        else:
            strategy_dictionary_local = randomise_dictionary_inputs(strategy_dictionary_local)


        if strategy_dictionary_local['regression_mode'] == 'classification':
            fitting_targets_local = classification_targets
        elif strategy_dictionary_local['regression_mode'] == 'regression':
            fitting_targets_local = continuous_targets

        fitting_inputs_local = preprocessing_inputs(strategy_dictionary_local, fitting_inputs_local)

        fitting_dictionary, error_loop, profit_factor = fit_tensorflow(strategy_dictionary_local, data_local,
                                                                       fitting_inputs_local, fitting_targets_local)

        if error_loop < error:
            error = error_loop
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary

    underlined_output('Best strategy fit')
    output_strategy_results(strategy_dictionary_optimum, fitting_dictionary_optimum, data_local, toc)

    return strategy_dictionary_optimum, data_local, fitting_inputs_local, fitting_targets_local


def randomise_dictionary_inputs(strategy_dictionary_local):
    strategy_dictionary_local['learning_rate'] = 10 ** np.random.uniform(-5, -1)
    strategy_dictionary_local['keep_prob'] = np.random.uniform(0.2, 0.8)
    return strategy_dictionary_local


def randomise_sequence_dictionary_inputs(strategy_dictionary_local):
    strategy_dictionary_local['learning_rate'] = 10 ** np.random.uniform(-5, -1)
    strategy_dictionary_local['num_layers'] = random.randint(1, 100)
    strategy_dictionary_local['num_units'] = random.randint(5, 100)
    return strategy_dictionary_local


if __name__ == '__main__':
    strategy_dictionary = {
        'trading_currencies': ['USDT', 'BTC'],
        'ticker_1': 'USDT_BTC',
        'ticker_2': 'BTC_ETH',
        'scraper_currency_1': 'BTC',
        'scraper_currency_2': 'ETH',
        'candle_size': 1800,
        'n_days': 40,
        'offset': 0,
        'bid_ask_spread': 0.004,
        'transaction_fee': 0.0025,
        'train_test_validation_ratios': [0.5, 0.25, 0.25],
        'output_flag': True,
        'plot_flag': False,
        'target_score': 'idealstrategy',
        'windows': [10, 50, 100],
        'regression_mode': 'regression',
        'preprocessing': 'None',
        'ml_mode': 'tensorflow',
        'sequence_flag': False,
        'output_units': 1,
        'web_flag': True,
        'filename1': "USDT_BTC.csv",
        'filename2': "BTC_ETH.csv",
        'scraper_page_limit': 10,
    }

    search_iterations = 5

    strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets = random_search(
        strategy_dictionary, search_iterations)

    underlined_output('Offset validation')
    offsets = np.linspace(0, 100, 5)

    tensorflow_offset_scan_validation(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets, offsets)

    print strategy_dictionary
