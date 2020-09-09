import seaborn as seaborn
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from keras import backend as K


def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

  #model.compile(loss='mse', optimizer=optimizer, metrics=['MeanSquaredLogarithmicError', 'mse'])

  return model

def initi_net(Y, drop_cols, dataset):
    dataset = dataset.drop(columns=drop_cols)

    # splits the data into train and test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # inspects the data
    '''
    sns.pairplot(train_dataset[list(dataset.columns.values)], diag_kind="kde")
    plt.show()
    '''

    all_data = dataset.copy()

    all_data_stats = all_data.describe()
    all_data_stats.pop(Y)
    all_data_stats = all_data_stats.transpose()

    train_stats = train_dataset.describe()
    train_stats.pop(Y)
    train_stats = train_stats.transpose()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(train_stats)

    # Split features from labels
    all_data_labels = all_data.pop(Y)

    train_labels = train_dataset.pop(Y)
    test_labels = test_dataset.pop(Y)

    # Normalize the data
    norm_all_data = (all_data - all_data_stats['mean']) / all_data_stats['std']

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    # The model
    model = build_model(train_dataset)

    # inspect model
    # print(model.summary())

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # Train the model
    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylabel('MAE {}'.format(Y))

    plt.show()

    plotter.plot({'Basic': history}, metric="mse")
    plt.ylabel('MSE {}^2'.format(Y))
    plt.show()


    '''
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    early_history = model.fit(normed_train_data, train_labels,
                              epochs=EPOCHS, validation_split=0.2, verbose=0,
                              callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    plotter.plot({'Early Stopping': early_history}, metric="mae")
    plt.ylabel('MAE {}'.format(Y))
    plt.show()
    '''


    print('NOW RESULTS FROM TRAINING DATA')
    print(model.evaluate(normed_test_data, test_labels, verbose=2))

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} for {}".format(mae, Y))
    print("Testing set Mean Sqr Error: {:5.2f} for {}".format(mse, Y))


    # Make Predictions
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(normed_test_data)
    # print(len(list(dataset.columns.values)))


    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values {}'.format(Y))
    plt.ylabel('Predictions {}'.format(Y))
    lims = [0, mae*10]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()


    print("Testing set Mean Abs Error: {:5.2f} for {}".format(mae, Y))

    """
    # Apply to Entire Data set
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(norm_all_data)

    all_predictions = model.predict(norm_all_data).flatten()
    print(list(all_predictions))
    print(len(all_predictions))
    """


if __name__ == '__main__':
    raw_dataset = pd.read_csv('master_state_data_nets.csv')

    dataset_main = raw_dataset.copy()

    drop_cols = ['STATE', 'abbr', 'positiveScore', 'negativeScore', 'negativeRegularScore','commercialScore', 'Pop',
                 'totalTestResults', 'posativeperpop', 'deathsperpop', 'deathrate', 'negative', 'death']

    #initi_net('positive', drop_cols, raw_dataset.copy())

    drop_cols = ['STATE', 'abbr', 'positiveScore', 'negativeScore', 'negativeRegularScore', 'commercialScore', 'Pop',
                 'totalTestResults', 'posativeperpop', 'deathsperpop', 'deathrate', 'positive', 'negative']

    # initi_net('death', drop_cols, raw_dataset.copy())

    drop_cols = ['STATE', 'abbr', 'positiveScore', 'negativeScore', 'negativeRegularScore', 'commercialScore', 'death',
                 'totalTestResults', 'posativeperpop', 'deathsperpop', 'deathrate', 'positive', 'negative']

    #initi_net('Pop', drop_cols, raw_dataset.copy())


    ranker = pd.DataFrame()
    #print(ranker)
    ranker['abbr'] = raw_dataset.copy().pop('abbr')
    ranker['total'] = raw_dataset.copy().pop('totalTestResults')


    case_rank = []

    for i in ranker['total']:
        if i < 1000:
            case_rank.append(1)
        elif i >= 1000 and i < 10000:
            case_rank.append(2)
        elif i >= 10000 and i < 20000:
            case_rank.append(3)
        elif i >= 20000 and i < 50000:
            case_rank.append(4)
        elif i >= 50000 and i < 100000:
            case_rank.append(5)
        elif i >= 100000 and i < 120000:
            case_rank.append(6)
        elif i >= 120000 and i < 150000:
            case_rank.append(7)
        elif i >= 150000 and i < 180000:
            case_rank.append(8)
        elif i >= 180000 and i < 200000:
            case_rank.append(9)
        else:
            case_rank.append(10)

    #print(case_rank)

    ranker['infection score'] = case_rank

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(ranker)

    drop_cols = ['STATE', 'abbr', 'positiveScore', 'negativeScore', 'negativeRegularScore', 'commercialScore', 'Pop',
                 'totalTestResults', 'posativeperpop', 'deathsperpop', 'deathrate', 'negative', 'death', 'positive']

    rank_test = raw_dataset.copy()
    rank_test['infection score'] = case_rank

    initi_net('infection score', drop_cols, rank_test)