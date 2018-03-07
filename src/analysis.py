import tensorflow as tf


def compute_accuracy(z, y):
    z = tf.argmax(z, axis=-1)
    y = tf.argmax(y, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(z, y), tf.float32))
