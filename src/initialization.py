import pickle
import random

import numpy as np
import tensorflow as tf

NUMBER_OF_LABELS = 10
MID_CHANNEL = 4
CONV_1_CHANNEL = 8
CONV_3_CHANNEL = 8
CONV_5_CHANNEL = 4
MAX_POOL_CHANNEL = 4

INCEPTION_LAYER_CHANNEL = CONV_1_CHANNEL + CONV_3_CHANNEL + CONV_5_CHANNEL + MAX_POOL_CHANNEL


def unpickle(file):
    with open(file, 'rb') as fo:
        the_dict = pickle.load(fo, encoding='bytes')
    return the_dict


def get_training_set(i):
    the_dict = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    x = the_dict[b'data']
    y = the_dict[b'labels']
    return format_x_data(x), format_y_data(y)


def get_dev_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][:5000]
    y = the_dict[b'labels'][:5000]
    return format_x_data(x), format_y_data(y)


def get_test_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][5000:]
    y = the_dict[b'labels'][5000:]
    return format_x_data(x), format_y_data(y)


def format_x_data(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def format_y_data(y):
    m = len(y)
    k = np.ones((m, 1)) * range(NUMBER_OF_LABELS)
    y = np.array(y).reshape(-1, 1)
    return y == k


def initialize_parameters():
    tf.set_random_seed(1)
    with tf.name_scope('first_layer'):
        w11 = tf.get_variable('w11', [1, 1, INCEPTION_LAYER_CHANNEL, CONV_1_CHANNEL],
                              initializer=tf.contrib.layers.xavier_initializer())
        w13 = tf.get_variable('w13', [3, 3, MID_CHANNEL, CONV_3_CHANNEL],
                              initializer=tf.contrib.layers.xavier_initializer())
        w15 = tf.get_variable('w15', [5, 5, MID_CHANNEL, CONV_5_CHANNEL],
                              initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope('second_layer'):
        w21 = tf.get_variable('w21', [1, 1, INCEPTION_LAYER_CHANNEL, CONV_1_CHANNEL],
                              initializer=tf.contrib.layers.xavier_initializer())
        w23 = tf.get_variable('w23', [3, 3, MID_CHANNEL, CONV_3_CHANNEL],
                              initializer=tf.contrib.layers.xavier_initializer())
        w25 = tf.get_variable('w25', [5, 5, MID_CHANNEL, CONV_5_CHANNEL],
                          initializer=tf.contrib.layers.xavier_initializer())

    parameters = {'w11': w11,
                  'w13': w13,
                  'w15': w15,
                  'w21': w21,
                  'w23': w23,
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


def create_placeholders(h, w, c):
    x = tf.placeholder(shape=[None, h, w, c], dtype="float")
    y = tf.placeholder(shape=[None, NUMBER_OF_LABELS], dtype="float")
    return x, y
