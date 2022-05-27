import logging
import os
import time
from enum import auto
from os.path import exists
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import TensorBoard

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import seaborn as sns

logger = logging.getLogger(__name__)
MODEL_PATH = "saved_model/"


class AutoEncoder(Model):
    """
  Parameters
  ----------
  output_units: int
    Number of output units

  code_size: int
    Number of units in bottle neck
  """

    def __init__(self, output_units, code_size=4):
        super().__init__()
        self.encoder = Sequential([
            Dense(128, activation='relu'), Dropout(0.1),
            Dense(64, activation='relu'), Dropout(0.1),
            Dense(32, activation='relu'), Dropout(0.1),
            Dense(16, activation='relu'), Dropout(0.1),
            Dense(8, activation='relu'), Dropout(0.1),
            Dense(code_size, activation='relu')
        ])

        self.decoder = Sequential([
            Dense(8, activation='relu'), Dropout(0.1),
            Dense(16, activation='relu'), Dropout(0.1),
            Dense(32, activation='relu'), Dropout(0.1),
            Dense(64, activation='relu'), Dropout(0.1),
            Dense(128, activation='relu'), Dropout(0.1),
            #Dense(output_units, activation= tf.keras.activations.softmax)
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def find_threshold(model, train_data_scaled):
    reconstructions = model.predict(train_data_scaled)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.msle(reconstructions, train_data_scaled)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
    return threshold


def get_predictions(model, test_data_scaled, threshold):
    predictions = model.predict(test_data_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, test_data_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds


def train_auto_encoder_model(dataset):
    import shutil

    # Clean previous training logs
    shutil.rmtree('logs/', ignore_errors=True)

    time.sleep(10)
    name = "cVu-auto-encoder-model"

    tensorboard = TensorBoard(log_dir="logs/{}".format(name))

    # Delete first two features, as we don't want them.

    features = np.delete(dataset, [0, 1], 1)
    rows, columns = features.shape
    logger.info(f"Dataset Size [ rows = {rows}, columns={columns}]")

    train_data, test_data = train_test_split(features, test_size=0.2, random_state=100)

    # Neural network model works better on scaled data. It will train faster and converge faster
    scaler = MinMaxScaler()
    # scaler function will learn the parameter using train data.Using these learned paramters,
    # will trnasform the test
    # dataset
    data_scaled = scaler.fit(train_data)
    # with open('standardscaler_object.ss', 'wb') as f:
    #     pickle.dump(data_scaled, f)
    # f.close()
    normal_train_data = data_scaled.transform(train_data)
    normal_test_data = data_scaled.transform(test_data)

    train_data_scaled = tf.cast(normal_train_data, tf.float32)
    test_data_scaled = tf.cast(normal_test_data, tf.float32)

    # train_data_scaled = data_scaled.transform(train_data)
    # test_data_scaled = data_scaled.transform(test_data)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1,
                                                  mode='min', restore_best_weights=True)



    model_exist = os.path.isdir('saved_model/')

    if model_exist:
        logger.warning(f"Model exist, Loading model from {MODEL_PATH}")
        auto_encoder_model = tf.keras.models.load_model(MODEL_PATH)
    else:
        logger.warning("Model does not exist")
        auto_encoder_model = AutoEncoder(output_units=train_data_scaled.shape[1])

    # model configuration
    auto_encoder_model.compile(loss='msle', metrics=['mse'], optimizer='adam')

    history = auto_encoder_model.fit(
        train_data_scaled,
        train_data_scaled,
        epochs=250,
        batch_size=1024,
        validation_data=(test_data_scaled, test_data_scaled),
        callbacks=[early_stop, tensorboard], verbose=1
    )

    auto_encoder_model.save('saved_model/')

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('MSLE Loss')
    # plt.legend(['loss', 'val_loss'])
    # plt.show()

    auto_encoder_model_threshold = find_threshold(auto_encoder_model, train_data_scaled)
    logger.info(f"Threshold: {auto_encoder_model_threshold}")
    model_predictions = get_predictions(auto_encoder_model, test_data_scaled, auto_encoder_model_threshold)

    #To plot loss graph
    # fig, ax = plt.subplots(nrows= 1, ncols=2, figsize=(15,5))
    # ax[0].plot(range(1,151),history.history['loss'])
    # ax[0].plot(range(1,151), history.history['val_loss'])
    #
    # ax[1].plot(range(1, 151), history.history['mse'])
    # ax[1].plot(range(1,151), history.history['val_mse'])
    # plt.show()
