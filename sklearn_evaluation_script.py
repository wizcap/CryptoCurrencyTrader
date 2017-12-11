import numpy as np
from random import choice, randint
from trading_strategy_fitting import fit_strategy, offset_scan_validation, tic, underlined_output, import_data,\
    input_processing
from data_input_processing import preprocessing_inputs
from strategy_evaluation import output_strategy_results, simple_momentum_comparison


def random_search(strategy_dictionary_local, n_iterations, toc):

    """random search to find optimum machine learning algorithm and preprocessing"""

    data_local = import_data(strategy_dictionary_local)
    fitting_inputs_local, continuous_targets, classification_targets = input_processing(
        data_local,
        strategy_dictionary_local)

    counter = 0
    error = 1e5
    fitting_targets_local = []
    fitting_dictionary_optimum = []
    strategy_dictionary_optimum = []
    while counter < n_iterations:
        counter += 1
        strategy_dictionary_local = randomise_dictionary_inputs(strategy_dictionary_local)

        if strategy_dictionary_local['regression_mode'] == 'classification':
            fitting_targets_local = classification_targets.astype(int)
        elif strategy_dictionary_local['regression_mode'] == 'regression':
            fitting_targets_local = continuous_targets

        fitting_inputs_local, strategy_dictionary = preprocessing_inputs(
            strategy_dictionary_local,
            fitting_inputs_local)

        fitting_dictionary, profit_factor, strategy_dictionary_local = fit_strategy(
            strategy_dictionary,
            data_local,
            fitting_inputs_local,
            fitting_targets_local)

        error_loop = fitting_dictionary['error']

        if error_loop < error and fitting_dictionary['n_trades'] != 0:
            error = error_loop
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary

    profit, test_profit = output_strategy_results(
        strategy_dictionary_optimum,
        fitting_dictionary_optimum,
        data_local,
        toc)

    return strategy_dictionary_optimum,\
        fitting_dictionary_optimum,\
        fitting_inputs_local,\
        fitting_targets_local,\
        data_local,\
        test_profit


def randomise_dictionary_inputs(strategy_dictionary_local):

    """ generate parameters for next step of random search """

    strategy_dictionary_local['ml_mode'] = choice([
        'svm',
        'randomforest',
        #'adaboost',
        'gradientboosting',
        'extratreesfitting'
    ])

    strategy_dictionary_local['preprocessing'] = choice(['PCA', 'FastICA', 'None'])
    return strategy_dictionary_local


def randomise_time_inputs(strategy_dictionary_local):

    """ generate time parameters for next step of random search """

    window = 1000

    strategy_dictionary_local['windows'] = randint(1, window / 10)

    strategy_dictionary_local['target_step'] = randint(1, window)

    return strategy_dictionary_local


def fit_time_scale(strategy_dictionary_input, search_iterations_local, time_iterations):

    """ fit timescale variables"""

    toc = tic()
    counter = 0
    strategy_dictionary_optimum = []
    optimum_profit = -2

    while counter < time_iterations:

        strategy_dictionary_input = randomise_time_inputs(strategy_dictionary_input)

        strategy_dictionary_local,\
            fitting_dictionary_local,\
            fitting_inputs_local,\
            fitting_targets_local,\
            data_local,\
            test_profit\
            = random_search(
                strategy_dictionary_input,
                search_iterations_local,
                toc)

        if test_profit > optimum_profit:
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary_local

        counter += 1

    underlined_output('Best strategy fit')

    if strategy_dictionary['plot_last']:
        strategy_dictionary['plot_flag'] = True

    output_strategy_results(
        strategy_dictionary_optimum,
        fitting_dictionary_optimum,
        data_local,
        toc,
        momentum_dict=simple_momentum_comparison(data_local, strategy_dictionary_optimum, fitting_dictionary_optimum))

    return strategy_dictionary_optimum


if __name__ == '__main__':
    strategy_dictionary = {
        'trading_currencies': ['USDT', 'BTC'],
        'ticker_1': 'USDT_BTC',
        'scraper_currency_1': 'BTC',
        'candle_size': 300,
        'n_days': 10,
        'offset': 0,
        'bid_ask_spread': 0.0007,
        'transaction_fee': 0.0025,
        'train_test_validation_ratios': [0.5, 0.2, 0.3],
        'output_flag': True,
        'plot_flag': False,
        'plot_last': True,
        'ml_iterations': 10,
        'target_score': 'n_steps',
        'web_flag': True,
        'filename1': "USDT_BTC.csv",
        'regression_mode': 'regression',
        'momentum_compare': True,
    }

    search_iterations = 10
    time_iterations = 10

    strategy_dictionary, fitting_inputs, fitting_targets, data_to_predict = fit_time_scale(
        strategy_dictionary,
        search_iterations,
        time_iterations)

    underlined_output('Offset validation')
    offsets = np.linspace(0, 300, 5)

    offset_scan_validation(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets, offsets)

    print strategy_dictionary
