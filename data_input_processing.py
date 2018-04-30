import numpy as np
import pandas as pd

import time
from poloniex import Poloniex
#from price_prediction.API_settings import poloniex_API_secret, poloniex_API_key
SEC_IN_DAY = 86400


class Data:

    """ Class to retrieve poloniex candle data from API or file"""

    def __init__(
            self,
            currency_pair,
            period, web_flag,
            start=None,
            end=None,
            offset=None,
            n_days=None,
            filename=None):

        """ Initialise instance of class. """

        self.date = []
        self.price = []
        self.close = []
        self.open = []
        self.high = []
        self.low = []
        self.volume = []
        self.time = []
        self.period = float(period)
        self.price_quotient = []
        self.currency_pair = currency_pair
        self.poloniex_session = []
        self.bid_ask_spread = []
        self.liquidation_factor = []

        if web_flag:
            self.start_poloniex_session()

            self.candle_input_web(currency_pair, start, end, period)
        else:
            self.candle_input_file(filename, period, offset, n_days)

        self.calculate_price_quotient()

    def start_poloniex_session(self):

        """ Initiate poloniex session """

        self.poloniex_session = Poloniex()#(poloniex_API_key, poloniex_API_secret)

    def candle_input_file(self, filename, period, offset, n_days):

        """ Read candle data from file."""

        candle_array = pd.read_csv(filename).as_matrix()

        end_index = candle_array[-1, 4] - offset * SEC_IN_DAY
        start_index = end_index - n_days * SEC_IN_DAY

        end_index = (np.abs(candle_array[:, 4] - end_index)).argmin()
        start_index = (np.abs(candle_array[:, 4] - start_index)).argmin()

        period_index = period / 300

        self.volume = candle_array[start_index:end_index:period_index, 0]
        self.date = candle_array[start_index:end_index:period_index, 4]
        self.open = candle_array[(start_index + period_index):end_index:period_index, 6]
        self.close = candle_array[(start_index + period_index - 1):end_index:period_index, 5]
        self.close = self.close[-len(self.open):]
        self.high = np.zeros(len(self.close))
        self.low = np.zeros(len(self.close))

        for i in range(int(np.floor(len(self.high) / period_index))):
            loop_start = i * period_index
            self.high[i] = np.max(candle_array[loop_start:loop_start + period_index, 2])
            self.low[i] = np.min(candle_array[loop_start:loop_start + period_index, 3])

    def candle_input_web(self, currency_pair, start, end, period):

        """Read candle input from web."""

        while True:
            try:
                candle_json = self.poloniex_session.returnChartData(currency_pair, period, start=start, end=end)
                break
            except Exception as e:
                print(e)
                time.sleep(1)

        candle_length = len(candle_json)
        self.volume = nan_array_initialise(candle_length)
        self.date = nan_array_initialise(candle_length)
        self.close = nan_array_initialise(candle_length)
        self.open = nan_array_initialise(candle_length)
        self.high = nan_array_initialise(candle_length)
        self.low = nan_array_initialise(candle_length)
        self.liquidation_factor = np.zeros(candle_length)

        for loop_counter in range(candle_length):
            self.volume[loop_counter] = float(candle_json[loop_counter][u'volume'])
            self.date[loop_counter] = float(candle_json[loop_counter][u'date'])
            self.close[loop_counter] = float(candle_json[loop_counter][u'close'])
            self.open[loop_counter] = float(candle_json[loop_counter][u'open'])
            self.high[loop_counter] = float(candle_json[loop_counter][u'high'])
            self.low[loop_counter] = float(candle_json[loop_counter][u'low'])

    def candle_update_web(self, time):

        """Update candle data at new time step."""

        start = self.date[-1]

        candle_json = self.poloniex_session.returnChartData(
            self.currency_pair,
            self.period,
            start=start,
            end=time)

        for loop_counter in range(len(candle_json)):

            date = float(candle_json[loop_counter][u'date'])

            if date > start:
                self.volume= np.append(self.volume, float(candle_json[loop_counter][u'volume']))
                self.date = np.append(self.date, date)
                self.close = np.append(self.close, float(candle_json[loop_counter][u'close']))
                self.open = np.append(self.open, float(candle_json[loop_counter][u'open']))
                self.high = np.append(self.high, float(candle_json[loop_counter][u'high']))
                self.low = np.append(self.low, float(candle_json[loop_counter][u'low']))
                self.liquidation_factor= np.append(self.liquidation_factor, 0)

    def spread_update_ticker(self):

        """Update spread data using ticker."""

        ticker = self.poloniex_session.returnTicker()

        pair_ticker = ticker.get(self.currency_pair)

        self.liquidation_factor[-1] = (float(pair_ticker.get('lowestAsk')) - self.close[-1]) / self.close[-1]

        self.close[-1] = pair_ticker.get(float(pair_ticker.get('highestBid')))
        self.open[-1] = self.close[-2]

    def calculate_price_quotient(self):

        """ Calculate quotient of opening and closing prices"""

        self.price_quotient = self.close / self.open


def nan_array_initialise(size):

    """ Initialise an array of NaNs. """

    array = np.empty((size,))
    array[:] = np.NaN
    return array


def train_validation_test_indices(input_data, ratios):

    """ Calculate train, test, validation indices """

    train_factor = ratios[0]
    val_factor = ratios[1]
    data_length = len(input_data)

    train_indices_local = range(0, int(data_length * train_factor))
    validation_indices_local = range(train_indices_local[-1] + 1, int(data_length * (train_factor + val_factor)))

    test_indices_local = range(validation_indices_local[-1] + 1, data_length)

    return train_indices_local, validation_indices_local, test_indices_local


def bid_ask_spread(ticker_list, limit=10):

    """Retrieve current bid ask spread from exchange for particular cryptocurrencies"""

    polo = Poloniex()

    ticker = polo.returnTicker()

    bid_ask_vector = np.zeros((len(ticker_list)))

    for idx, pair in enumerate(ticker_list):
        if pair != "USDT":
            pair_ticker = ticker.get(pair)

            bid_ask_vector[idx]\
                += (pair_ticker.get('lowestAsk') - pair_ticker.get('highestBid'))\
                   / (limit * pair_ticker.get('highestBid'))

    return bid_ask_vector



