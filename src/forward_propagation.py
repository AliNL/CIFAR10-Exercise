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
