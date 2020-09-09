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

    dataset = dataset.drop(columns=['STATE', 'abbr', 'positiveScore', 'negativeScore', 'negativeRegularScore',
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

    '''
    posative = dataset[:, 11]
    negative = dataset[:, 10]
    death = dataset[:, 12]
    totaltestresults = dataset[:, 14]
    '''

    # first test posative
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

    print(normed_train_data[:10])