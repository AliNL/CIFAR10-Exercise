import pickle
import random

import numpy as np
import tensorflow as tf

NUMBER_OF_LABELS = 10


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

    w1 = tf.get_variable('w1', [5, 5, 4, 8], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2', [3, 3, 8, 8], initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3', [3, 3, 8, 8], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {'w1': w1,
                  'w2': w2,
                  'w3': w3}

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
    x = tf.placeholder(shape=[None, h, w, c], dtype="float", name='X')
    y = tf.placeholder(shape=[None, NUMBER_OF_LABELS], dtype="float", name='Y')
    return x, y
