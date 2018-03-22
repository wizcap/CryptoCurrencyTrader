from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np


def imputer_transform(data, missing_values='NaN'):
    imputer = Imputer(missing_values=missing_values)
    return imputer.fit_transform(data)


def train_test_indices(input_data, train_factor):
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    test_indices_local = range(train_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local


def train_test_validation_indices(input_data, ratios):
    train_factor = ratios[0]
    val_factor = ratios[1]
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    validation_indices_local = range(train_indices_local[-1] + 1, int(data_length * (train_factor + val_factor)))

    test_indices_local = range(validation_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local, validation_indices_local
