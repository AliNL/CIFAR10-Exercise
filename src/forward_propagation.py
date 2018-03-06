import numpy as np
import tensorflow as tf


def conv_1x1_layer(x, w):
    assert w.shape[0] == 1
    assert w.shape[1] == 1
    z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return z


def conv_3x3_layer(x, w):
    assert w.shape[0] == 3
    assert w.shape[1] == 3
    m = x.shape[-1]
    z = conv_1x1_layer(x, np.ones((1, 1, m, 16)))
    z = tf.nn.conv2d(z, w, strides=[1, 1, 1, 1], padding='SAME')
    return z


def conv_5x5_layer(x, w):
    assert w.shape[0] == 5
    assert w.shape[1] == 5
    m = x.shape[-1]
    z = conv_1x1_layer(x, np.ones((1, 1, m, 16)))
    z = tf.nn.conv2d(z, w, strides=[1, 1, 1, 1], padding='SAME')
    return z


def max_pool_layer(x):
    m = x.shape[-1]
    z = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    z = conv_1x1_layer(z, np.ones((1, 1, m, 16)))
    return z


def inception_layer(x, w1, w3, w5):
    z1 = conv_1x1_layer(x, w1)
    z3 = conv_3x3_layer(x, w3)
    z5 = conv_5x5_layer(x, w5)
    zp = max_pool_layer(x)
    z = tf.concat([z1, z3, z5, zp], -1)
    return z


def forward_prop(x, parameters):
    w11 = parameters['w11']
    w13 = parameters['W13']
    w15 = parameters['w15']
    w21 = parameters['w21']
    w23 = parameters['W23']
    w25 = parameters['w25']

    x = conv_1x1_layer(x, np.ones((1, 1, 3, 128)))

    z1 = inception_layer(x, w11, w13, w15)
    a1 = tf.nn.relu(z1)

    z2 = inception_layer(a1, w21, w23, w25)
    a2 = tf.nn.relu(z2)
    p2 = max_pool_layer(a2)

    p2 = tf.contrib.layers.flatten(p2)

    z3 = tf.contrib.layers.fully_connected(p2, 10, activation_fn=None)

    return z3
