import pickle
import random

import numpy as np
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        the_dict = pickle.load(fo, encoding='bytes')
    return the_dict


def get_training_set(i):
    the_dict = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    x = the_dict[b'data']
    x = format_x_data(x)
    y = the_dict[b'labels']
    y = np.array(y).reshape(-1, 1)
    return x, y


def get_dev_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][:5000]
    x = format_x_data(x)
    y = the_dict[b'labels'][:5000]
    y = np.array(y).reshape(-1, 1)
    return x, y


def get_test_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][5000:]
    x = format_x_data(x)
    y = the_dict[b'labels'][5000:]
    y = np.array(y).reshape(-1, 1)
    return x, y


def format_x_data(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def initialize_parameters():
    tf.set_random_seed(1)
    w11 = tf.get_variable('w11', [1, 1, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
    w13 = tf.get_variable('W13', [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    w15 = tf.get_variable('w15', [5, 5, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
    w21 = tf.get_variable('w21', [1, 1, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
    w23 = tf.get_variable('W23', [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    w25 = tf.get_variable('w25', [5, 5, 16, 16], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {'w11': w11,
                  'W13': w13,
                  'w15': w15,
                  'w21': w21,
                  'W23': w23,
                  'w25': w25
                  }

    return parameters


def random_mini_batches(x, y, size):
    m = x.shape[0]
    n = int(m / size)
    last_batch_size = m % size
    id_list = list(range(m))
    random.shuffle(id_list)
    mini_batches = []
    for i in range(n):
        start_id = i * size
        mini_id_list = id_list[start_id:(start_id + size)]
        mini_batches.append((x[mini_id_list], y[mini_id_list]))
    if last_batch_size:
        mini_id_list = id_list[-last_batch_size:]
        mini_batches.append((x[mini_id_list], y[mini_id_list]))
    return mini_batches
