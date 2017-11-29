import numpy as np
from matplotlib import pyplot as plt


def strategy_profit(strategy_score, fractional_price, strategy_dictionary):
    buy_sell_length = len(strategy_score)
    portfolio_value = np.ones(buy_sell_length)
    cash_value = np.zeros(buy_sell_length)
    crypto_value = np.zeros(buy_sell_length)
    n_trades = 0
    cash_value[0] = 1

    for index in range(1, buy_sell_length-1):

        effective_fee_factor = (strategy_dictionary['transaction_fee'] + strategy_dictionary['bid_ask_spread'])

        crypto_value[index] = crypto_value[index - 1] * fractional_price[index - 1]
        cash_value[index] = cash_value[index - 1]
        portfolio_value[index] = crypto_value[index] + cash_value[index]

        score_step = (strategy_score[index] - strategy_score[index-1]) * portfolio_value[index]

        if score_step > 0:

            if score_step > cash_value[index]:
                score_step = cash_value[index]

            effective_fee = effective_fee_factor * score_step

            if score_step > effective_fee:

                cash_value[index] = cash_value[index] - score_step
                crypto_value[index] = crypto_value[index] + score_step - effective_fee

                n_trades += 1

        else:

            score_step = abs(score_step)

            if score_step > crypto_value[index]:
                score_step = crypto_value[index]

            effective_fee = effective_fee_factor * score_step

            if score_step > effective_fee:

                cash_value[index] = cash_value[index] + score_step - effective_fee
                crypto_value[index] = crypto_value[index] - score_step

                n_trades += 1

    return portfolio_value, n_trades


def post_process_training_results(strategy_dictionary, fitting_dictionary, data):
    fitting_dictionary['portfolio_value'], fitting_dictionary['n_trades'] = strategy_profit(
        fitting_dictionary['fitted_strategy_score'],
        data.fractional_close[fitting_dictionary['test_indices']],
        strategy_dictionary)

    return fitting_dictionary


def strategy_profit_score(strategy_profit_local, number_of_trades):
    profit_fraction = strategy_profit_local[-1] / np.min(strategy_profit_local)
    if number_of_trades == 0:
        profit_fraction = -profit_fraction
    return profit_fraction


def draw_down(strategy_profit_local):
    draw_down_temp = np.diff(strategy_profit_local)
    draw_down_temp[draw_down_temp > 0] = 0
    return np.mean(draw_down_temp)


def output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc):
    prediction_data = data_to_predict.close[fitting_dictionary['validation_indices']]

    profit_factor = []
    if strategy_dictionary['output_flag']:
        print "Fitting time: ", toc()
        
        profit_factor = fitting_dictionary['portfolio_value'][-1] * prediction_data[0]\
                        / (fitting_dictionary['portfolio_value'][0] * prediction_data[-1]) - 1

        print "Fractional profit compared to buy and hold: ", profit_factor
        print "Mean squared error: ", fitting_dictionary['error']
        print "Number of days: ", strategy_dictionary['n_days']
        print "Candle time period:", strategy_dictionary['candle_size']
        print "Fitting model: ", strategy_dictionary['ml_mode']
        print "Regression/classification: ", strategy_dictionary['regression_mode']
        print "Number of trades: ", fitting_dictionary['n_trades']
        print "Offset: ", strategy_dictionary['offset']
        print "\n"

    if strategy_dictionary['plot_flag']:
        plt.figure(1)
        close_price = plt.plot(prediction_data)
        portfolio_value = plt.plot(prediction_data[0] * fitting_dictionary['portfolio_value'])
        plt.legend([close_price, portfolio_value], ['Close Price', 'Portfolio Value'])
        plt.xlabel('Candle number')
        plt.ylabel('Exchange rate')
        plt.show()
    
    return profit_factor
