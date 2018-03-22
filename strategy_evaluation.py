import numpy as np


def strategy_profit(
        strategy_score,
        close_price,
        bid_ask_spread,
        transaction_fee=0.0025,
        stop_loss=1,
        cash_reserve=0.1):

    """calculate net profit of trading strategy """

    buy_sell_length = len(strategy_score)
    portfolio_value = np.ones(buy_sell_length)
    cash_value = np.zeros(buy_sell_length)
    crypto_value = np.zeros(buy_sell_length)
    effective_fee_factor = (transaction_fee + 0.5 * bid_ask_spread)

    fractional_price = fractional_change(close_price)

    n_trades = 0
    curr_val = 1
    stop_val = 1

    cash_value[0] = 0.5
    crypto_value[0] = 0.5

    for index in range(1, buy_sell_length):
        crypto_value[index] = crypto_value[index - 1] * fractional_price[index - 1]
        cash_value[index] = cash_value[index - 1]
        portfolio_value[index] = crypto_value[index] + cash_value[index]

        curr_val *= fractional_price[index - 1]

        score_step = (strategy_score[index] - strategy_score[index - 1])

        score_step = portfolio_value[index] * score_step

        #Stoploss
        stop_val = max(stop_val, curr_val)

        if 0 < (stop_val - curr_val) / stop_val < (1 - stop_loss):
            score_step = - 10

            if np.prod(fractional_price[index-1]) > 1:
                stop_val = curr_val

        if score_step > 0:

            if score_step > cash_value[index]:
                score_step = cash_value[index]

            effective_fee = effective_fee_factor * score_step

            cash_value[index] = cash_value[index] - score_step
            crypto_value[index] = crypto_value[index] + score_step - effective_fee

            n_trades += 1

        elif score_step < 0:

            score_step = abs(score_step)

            if score_step > crypto_value[index]:
                score_step = crypto_value[index]

            effective_fee = effective_fee_factor * score_step

            cash_value[index] = cash_value[index] + score_step - effective_fee
            crypto_value[index] = crypto_value[index] - score_step

            n_trades += 1

    if cash_value[index] / (cash_value[index] + crypto_value[index]) < cash_reserve:
        cash_value[index], crypto_value[index] = cash_reserve * (cash_value[index] + crypto_value[index]),\
                                                 (1 - cash_reserve) * (cash_value[index] + crypto_value[index])

    return portfolio_value, n_trades, cash_value, crypto_value


def fractional_change(value):

    """ Calculate fractional change in a value """

    return value[1:] / value[:-1]


def portfolio_value(self, portfolio_array):

    """ Calculate value of portfolio at every time step"""

    portfolio_array = norm_portfolio_array(portfolio_array)


