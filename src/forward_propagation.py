import numpy as np
import tensorflow as tf

from src.initialization import MID_CHANNEL, MAX_POOL_CHANNEL, INCEPTION_LAYER_CHANNEL, NUMBER_OF_LABELS


def conv_1x1_layer(x, w):
    assert w.shape[0] == 1
    assert w.shape[1] == 1
    z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name='Conv_1x1')
    return z


def conv_3x3_layer(x, w):
    assert w.shape[0] == 3
    assert w.shape[1] == 3
    m = x.shape[-1]
    z = conv_1x1_layer(x, np.ones((1, 1, m, MID_CHANNEL)))
    z = tf.nn.conv2d(z, w, strides=[1, 1, 1, 1], padding='SAME', name='Conv_3x3')
    return z


def conv_5x5_layer(x, w):
    assert w.shape[0] == 5
    assert w.shape[1] == 5
    m = x.shape[-1]
    z = conv_1x1_layer(x, np.ones((1, 1, m, MID_CHANNEL)))
    z = tf.nn.conv2d(z, w, strides=[1, 1, 1, 1], padding='SAME', name='Conv_5x5')
    return z


def max_pool_layer(x):
    m = x.shape[-1]
    z = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='Max_Pool_3x3')
    z = conv_1x1_layer(z, np.ones((1, 1, m, MAX_POOL_CHANNEL)))
    return z


def inception_layer(x, w1, w3, w5):
    z1 = conv_1x1_layer(x, w1)
    z3 = conv_3x3_layer(x, w3)
    z5 = conv_5x5_layer(x, w5)
    zp = max_pool_layer(x)
    z = tf.concat([z1, z3, z5, zp], -1)
    return z


def compute_cost(z, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y))


def forward_prop(x, parameters):
    w11 = parameters['w11']
    w13 = parameters['w13']
    w15 = parameters['w15']
    w21 = parameters['w21']
    w23 = parameters['w23']
    w25 = parameters['w25']

    x = conv_1x1_layer(x, np.ones((1, 1, 3, INCEPTION_LAYER_CHANNEL)))
    with tf.name_scope('first_inception_layer'):
        z1 = inception_layer(x, w11, w13, w15)
    a1 = tf.nn.relu(z1, name='Relu')

    with tf.name_scope('second_inception_layer'):
        z2 = inception_layer(a1, w21, w23, w25)
    a2 = tf.nn.relu(z2, name='Relu')
    p2 = max_pool_layer(a2)

    p2 = tf.contrib.layers.flatten(p2)

    z3 = tf.contrib.layers.fully_connected(p2, NUMBER_OF_LABELS, activation_fn=None)

    return z3
