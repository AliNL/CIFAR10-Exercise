import numpy as np
import tensorflow as tf

from src.initialization import NUMBER_OF_LABELS


def compute_cost(z, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y))


def forward_prop(x, parameters):
    w1 = parameters['w1']  # 5x5x4x8
    w2 = parameters['w2']  # 3x3x8x8
    w3 = parameters['w3']  # 3x3x8x8

    x = tf.nn.conv2d(x, np.ones((1, 1, 3, 4)), strides=[1, 1, 1, 1], padding='SAME', name='Conv_1x1')  # 32x32x4

    x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')  # 28x28x8

    x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')  # 24x24x8

    x = tf.nn.conv2d(x, w2, strides=[1, 3, 3, 1], padding='VALID')  # 8x8x8

    x = tf.nn.conv2d(x, w3, strides=[1, 1, 1, 1], padding='SAME')  # 8x8x8

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 4x4x8

    x = tf.contrib.layers.flatten(x)

    z = tf.contrib.layers.fully_connected(x, NUMBER_OF_LABELS, activation_fn=None)

    return z
