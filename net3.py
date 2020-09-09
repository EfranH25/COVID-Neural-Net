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

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

if __name__ == '__main__':
    raw_dataset = pd.read_csv('master_state_data_nets.csv')

    dataset = raw_dataset.copy()

    dataset = dataset.drop(columns=['STATE', 'abbr', 'positiveScore','negativeScore','negativeRegularScore',
                                    'commercialScore'])

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset)

    '''
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset)
        
        sns.pairplot(train_dataset[['Density','stateGDP','Automobiles', 'positive', 'death', 'totalTestResults',
                               'positiveScore', 'score']], diag_kind="kde")
        plt.show()
    '''

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset)

    '''
    posative = dataset[:, 11]
    negative = dataset[:, 10]
    death = dataset[:, 12]
    totaltestresults = dataset[:, 14]
    '''

    #first test posative
    train_stats = train_dataset.describe()
    train_stats.pop("positive")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('positive')
    test_labels = test_dataset.pop('positive')


    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = build_model()
    model.summary()

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)

    print(example_result)

    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [positive]')
    plt.show()

    plotter.plot({'Basic': history}, metric="mse")
    plt.ylim([0, 20])
    plt.ylabel('MSE [positive^2]')

    odel = build_model()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(normed_train_data, train_labels,
                              epochs=EPOCHS, validation_split=0.2, verbose=0,
                              callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    plotter.plot({'Early Stopping': early_history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [positive]')
    plt.show()

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} positive".format(mae))

    #Make Predictions
    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()