from keras.layers import Input, Conv2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np
from porftfolio_functions import calculate_portfolio_value_backend


def tensorflow_cnn_fitting(
        train_indices,
        test_indices,
        validation_indices,
        input_data,
        price_data,
        load_net=None,
        retrain=False):

    """use keras/tensorflow to fit training inputs to targets and predict targets"""

    train_data = input_data[train_indices, :, :]

    test_data = input_data[test_indices, :, :]

    val_data = input_data[validation_indices, :, :]

    input_size = input_data.shape

    input1 = Input(shape=(input_size[1], input_size[2], input_size[3]))

    train_price = price_data[train_indices]

    early_stopping = EarlyStopping(monitor='loss', patience=2000)

    conv1 = Conv2D(
        2,
        (1, 3),
        activation='relu',
        kernel_regularizer=l2(1E-8))(input1)
    conv2 = Conv2D(
        1,
        (1, input_size[2] - 2),
        activation='relu',
        kernel_regularizer=l2(1E-8))(conv1)
    flat1 = Flatten()(conv2)
    preds = Activation('softmax')(flat1)

    model = Model(input1, preds)

    opt = Adam(lr=3E-5)

    model.compile(
        loss=custom_loss,
        optimizer=opt)

    model.summary()

    if load_net:
        model.load_weights(load_net)

    if not load_net or (load_net and retrain):
        batch_size = 50

        model.fit_generator(
            random_fit_generator(train_data, train_price, batch_size),
            steps_per_epoch=1000,
            validation_steps=150,
            epochs=100,
            callbacks=[early_stopping])

        model.save('model.h5')

    training_strategy_score = model.predict(train_data)
    fitted_strategy_score = model.predict(test_data)
    validation_strategy_score = model.predict(val_data)

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
        'validation_strategy_score': validation_strategy_score,
    }

    return fitting_dictionary


def random_fit_generator(data, labels, batch_size):

    """ Pass random but continuous slices to fitting """

    data_length = len(data)

    while True:
        slice_start = np.random.randint(0, data_length - batch_size)

        yield data[slice_start:slice_start+batch_size, :, :, :], labels[slice_start:slice_start+batch_size]


def custom_loss(y_true, y_pred):

    """ Keras wrapper for loss function """

    _, cum_log_return = calculate_portfolio_value_backend(y_pred, y_true)

    return 1 - cum_log_return



