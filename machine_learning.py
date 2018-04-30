from keras.layers import Input, Conv2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import os

from porftfolio_functions import calculate_portfolio_value_backend


def tensorflow_cnn_fitting(
        train_indices,
        test_indices,
        validation_indices,
        input_data,
        price_data,
        load_net=None,
        retrain=None,
        batch_size=109,
        epochs=500,
        fit_kwargs={}):

    """use keras/tensorflow to fit training inputs to targets and predict targets"""

    train_data = input_data[train_indices, :, :, :]

    test_data = input_data[test_indices, :, :, :]

    if validation_indices:
        val_data = input_data[validation_indices, :, :, :]

    input_size = input_data.shape

    input1 = Input(shape=(input_size[1], input_size[2], input_size[3]))

    train_price = price_data[train_indices, :]

    conv1 = Conv2D(
        3,
        (1, 4),
        activation='relu',)(input1)
    conv2 = Conv2D(
        10,
        (1, input_size[2] - 3),
        activation='relu',
        kernel_regularizer=l2(5E-9))(conv1)
    conv3 = Conv2D(
        1,
        (1, 1),
        activation='relu',
        kernel_regularizer=l2(5E-8))(conv2)
    flat1 = Flatten()(conv3)
    preds = Activation('softmax')(flat1)

    model = Model(input1, preds)

    opt = Adam(lr=3E-4)

    model.compile(
        loss=custom_loss,
        optimizer=opt)

    model.summary()

    if not fit_kwargs.get('steps_per_epoch'):
        fit_kwargs['steps_per_epoch'] = 1000

    if load_net:
        model.load_weights(os.getcwd() + load_net)

    if not load_net or (load_net and retrain):

        if retrain:
            epochs = retrain

        model.fit_generator(
            random_fit_generator(train_data, train_price, batch_size),
            validation_steps=0.08 * fit_kwargs['steps_per_epoch'],
            epochs=epochs,
            **fit_kwargs)

        model.save(os.getcwd() + 'model.h5')

    training_strategy_score = model.predict(train_data)
    fitted_strategy_score = model.predict(test_data)

    if validation_indices:
        validation_strategy_score = model.predict(val_data)

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
    }

    if validation_indices:
        fitting_dictionary['validation_strategy_score'] = validation_strategy_score,

    return fitting_dictionary


def random_fit_generator(data, labels, batch_size, sample_bias=5E-5):

    """ Pass random but continuous slices to fitting """

    data_length = len(data)

    while True:
        slice_start = data_length\
                      - int(data_length * np.log(np.random.random())
                          / np.log(sample_bias * (1 - sample_bias)))\
                      - batch_size - 1

        yield data[slice_start:slice_start+batch_size, :, :, :], labels[slice_start:slice_start+batch_size]


def custom_loss(y_true, y_pred):

    """ Keras wrapper for loss function """

    _, cum_log_return = calculate_portfolio_value_backend(y_pred, y_true)

    return 1 - cum_log_return



