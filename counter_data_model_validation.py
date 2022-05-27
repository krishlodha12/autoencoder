import os
import pickle
import sys
import time
from fileinput import filename
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Queue

import requests
import logging
import csv
import urllib3
import numpy as np
import pandas as pd
from threading import Thread

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

#from predict_anamoly import predict_anamoly

# Constansts
COUNTER_API_URL = "https://cvu.sdsu.edu/dp/port57/getcsv/counters"
API_LOGIN_URL = "https://cvu.sdsu.edu/__login__"
REQUEST_PAYLOAD = {"username": "counters", "password": "eipa8eikacooZe!"}
DOWNLOADING_RATE_IN_SECONDS = 60
PREDICTION_SLEEP_TIME_IN_SECONDS = 0.1
DATA_BUFFER_SIZE = 100000
LOG_DIR_PATH = 'validation_logs'
MODEL_THRESHOLD =  0.00247487518

# global variables
last_packet_epoch_time = 0
total_rows_saved = 0
print_header = True
data_buffer =Queue(DATA_BUFFER_SIZE)

# main dataset that will get updated based on provided frequency
counter_np_dataset = np.empty((0, 758))

def add_rows_to_data_buffer(new_rows):
    """
      Place downloaded packets to data buffer queue for prediction
      :return:
      """
    for row in new_rows:
        data_buffer.put(row)


def download_counter_data():
    """
    This function authenticate and download data from rest api endpoint
    It keeps the track of last processed packet only processes and store latest available data.
    :return:
    """
    session = requests.Session()
    logger.debug("Authenticating...")
    response = session.post(API_LOGIN_URL, json=REQUEST_PAYLOAD, verify=False)
    if response.status_code == 200:

        logger.debug("Authentication is complete.")
        logger.debug('Downloading counter data...')

        total_row_count = 0

        response = session.get(COUNTER_API_URL, verify=False)
        if response.status_code == 200:
            new_rows = []
            data = response.content.decode('utf-8')
            rows = csv.reader(data.splitlines(), delimiter=',')

            row_count = -1

            global print_header

            for row in rows:
                row_count = row_count + 1

                # Print header only once. We don't really need header here.
                if print_header and row_count == 0:
                #    logger.info(f'Header : {row}')
                    print_header = False
                    continue

                # Skip header
                if row_count == 0:
                    continue

                packet_epoch_time = int(row[1])
                global last_packet_epoch_time

                # Skip previously extracted rows and append new data only.
                if packet_epoch_time > last_packet_epoch_time:
                    last_packet_epoch_time = packet_epoch_time
                    total_row_count = total_row_count + 1
                    new_rows.append(row)

            add_rows_to_data_buffer(new_rows)

        elif response.status_code == 403:
            logger.error('Authentication failed : Forbidden request')
            logger.error(response.content)
        elif response.status_code == 401:
            logger.error('Authentication failed :Unauthorized request')
            logger.error(response.content)

    global total_rows_saved
    total_rows_saved += total_row_count

    logger.info(f'Total new rows downloaded/extracted from API : {total_row_count}')
    logger.info(f'Total data downloaded so far: {total_rows_saved}')
    logger.info(f'Data buffer size : {data_buffer.qsize()}')


def predict_data_anomaly(auto_encoder_model):
    if not data_buffer.empty():
        # Poll packet from data buffer queue.
        packet_info = data_buffer.get()

        logger.info(f'Packet Info : {packet_info}')
        new_rows = []
        new_rows.append(packet_info)
        test_data = np.array(new_rows)
        features = np.delete(test_data, [0, 1], 1)

        # with open('standardscaler_object.ss', 'rb') as f:
        #     scaled_data_object = pickle.load(f)
        #scaler = MinMaxScaler()
        #scaled_data = scaled_data_object.transform(features.copy())

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(features.copy())

        predicted_value = predict_anamoly(auto_encoder_model, scaled_data)
        logger.info(f'Predicted value : {predicted_value}')


def predict_anamoly(auto_encoder_model,scaled_data):

    prediction = auto_encoder_model.predict(scaled_data)
  # provides losses of individual instances
    error = tf.keras.losses.msle(prediction, scaled_data)
    logger.info(f'Error  {error[0]}')
    absolute_difference = abs(error[0] - MODEL_THRESHOLD)
    logger.info(f'Absolute value Difference between Error and Threshold  {"{:e}".format(absolute_difference)}')
  # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(error) > MODEL_THRESHOLD
    predicted = anomaly_mask.map(lambda x: 'Anomaly Detected' if x == True else 'No Anomaly Detected')
    return predicted.iloc[0]

def run_counter():
    """
    This function continously download data from rest api endpoint.
    :return:None
    """
    logger.info(f'Downloading counter data every {DOWNLOADING_RATE_IN_SECONDS} seconds.')
    while True:
        while True:
            download_counter_data()
            time.sleep(DOWNLOADING_RATE_IN_SECONDS)


def run_prediction():

    """
    This function responsible for model prediction.
    :return:None
    """
    auto_encoder_model = tf.keras.models.load_model('saved_model/')
    while True:
        time.sleep(PREDICTION_SLEEP_TIME_IN_SECONDS)
        predict_data_anomaly(auto_encoder_model)


if __name__ == "__main__":

    if not os.path.exists(LOG_DIR_PATH):
        os.makedirs(LOG_DIR_PATH)

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - [%(threadName)30s] : %(levelname)s - %(''message)s')

    log_file_path = LOG_DIR_PATH + '/' + "counter_prediction.log";
    log_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1)
    log_handler.suffix = "%Y%m%d"
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    logger.info(f"Writing logs to {log_file_path}")
    logger.info(f"Data buffer queue size is set to  {DATA_BUFFER_SIZE}")

    # Counter thread is responsible for data downloading/extraction using REST API
    counter_thread = Thread(name='counter_data_download_thread', target=run_counter)
    counter_thread.start()

    # Auto encoder thread is responsible for training our auto encoder
    prediction_thread = Thread(name='auto_encoder_prediction_thread', target=run_prediction)
    prediction_thread.start()

    counter_thread.join()
    prediction_thread.join()
