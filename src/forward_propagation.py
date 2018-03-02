import tensorflow as tf


def conv_1x1_layer(x, w):
    assert w.shape[0] == 1
    assert w.shape[1] == 1
    z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return z
