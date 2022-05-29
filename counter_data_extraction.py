import os
import time
import requests
import logging
import csv
import urllib3
import numpy as np
from threading import Thread

from auto_encoder_train import train_auto_encoder_model

logger = logging.getLogger(__name__)

# Constansts

COUNTER_API_URL = "https://cvu.sdsu.edu/dp/port57/getcsv/counters"
API_LOGIN_URL = "https://cvu.sdsu.edu/__login__"
REQUEST_PAYLOAD = {"username": "counters", "password": "********"}

SAMPLING_RATE_IN_SECONDS = 25  # 1 minute
TRAIN_AUTOENCODER_FREQUENCY_IN_SECONDS = 10  # 5 minutes
MAX_DATASET_TRAINING_SIZE = 50000
SLIDING_WINDOW_TRIM_SIZE = 50000

# global variables
last_packet_epoch_time = 0
total_rows_saved = 0
print_header = True

# main dataset that will get updated based on provided frequency
counter_np_dataset = np.empty((0, 758))


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
                    logger.info(f'Header : {row}')
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
                    if(len(row)==758):
                        new_rows.append(row)

            append_data_to_np(new_rows)

        elif response.status_code == 403:
            logger.error('Authentication failed : Forbidden request')
            logger.error(response.content)
        elif response.status_code == 401:
            logger.error('Authentication failed :Unauthorized request')
            logger.error(response.content)

    global total_rows_saved
    total_rows_saved += total_row_count

    logger.info(f'Total new rows downloaded/extracted from API : {total_row_count}')
    logger.info(f'Total rows saved so far: {total_rows_saved}')
    rows, columns = counter_np_dataset.shape
    logger.info(f'Total np dataset size : {rows}')


def append_data_to_np(new_data):
    """
    This function add new row to existing np ndarray.
    :param new_data:
    :return:None
    """

    if len(new_data)==0:
     return;

    global counter_np_dataset
    new_np_data = np.array(new_data)
    new_np_rows, new_np_columns = new_np_data.shape
    counter_np_rows, counter_np_columns = counter_np_dataset.shape
    logger.info(f'NEW NP DIMENSIONS : {new_np_rows} rows & {new_np_columns} columns')
    logger.info(f'COUNTER NP DIMENSIONS : {counter_np_rows} rows & {counter_np_columns} columns')
    counter_np_dataset = np.concatenate((counter_np_dataset, new_np_data))


def train_autoencoder():
    global counter_np_dataset
    rows, columns = counter_np_dataset.shape

    if(rows>= MAX_DATASET_TRAINING_SIZE):

        logger.info(f"Training auto encoder data set size {rows} rows.")
        train_auto_encoder_model(counter_np_dataset.copy())
        logger.info("Finished training auto encoder.")
        logger.info(f"Dataset size is greater than max dataset size {MAX_DATASET_TRAINING_SIZE}")
        logger.info(f"Trimming dataset size by {SLIDING_WINDOW_TRIM_SIZE} rows ")
        counter_np_dataset = counter_np_dataset[SLIDING_WINDOW_TRIM_SIZE:, :]


def run_counter():
    """
    This function continuously download data from rest api endpoint.
    :return:None
    """
    logger.info(f'Downloading counter data every {SAMPLING_RATE_IN_SECONDS} seconds.')
    while True:
        while True:
            download_counter_data()
            time.sleep(SAMPLING_RATE_IN_SECONDS)


def run_autoencoder():
    """
    This function responsible for training auto encoder model using latest available dataset
    :return:None
    """
  #  logger.info(f'Training autoencoder every {TRAIN_AUTOENCODER_FREQUENCY_IN_SECONDS} seconds.')
    while True:
        time.sleep(TRAIN_AUTOENCODER_FREQUENCY_IN_SECONDS)
        train_autoencoder()


if __name__ == "__main__":
    if not os.path.exists('Threshold_log'):
        os.makedirs('Threshold_log')
    # Initialize logging
    logging.basicConfig(filename='Threshold_log/Threshold_logs_20K.log', level=logging.INFO, format='%(asctime)s - %(name)s - [%(threadName)19s] : %(levelname)s - %('
                                                   'message)s', datefmt='%m/%d/%Y %H:%M:%S')
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Counter thread is responsible for data downloading/extraction using REST API
    counter_thread = Thread(name='counter_thread', target=run_counter)
    counter_thread.start()

    # Auto encoder thread is responsible for training our auto encoder
    auto_encoder_thread = Thread(name='auto_encoder_thread', target=run_autoencoder)
    auto_encoder_thread.start()

    counter_thread.join()
    auto_encoder_thread.join()
