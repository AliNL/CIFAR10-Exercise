import unittest

import numpy as np
import tensorflow as tf


class ForwardPropagationTest(unittest.TestCase):
    def test_exceptions_of_conv_1x1_layer(self):
        from src.forward_propagation import conv_1x1_layer
        x = np.random.randn(2, 24, 24, 3)
        w = np.random.randn(5, 1, 3, 3)
        self.assertRaises(AssertionError, conv_1x1_layer, x, w)

    def test_result_of_conv_1x1_layer(self):
        from src.forward_propagation import conv_1x1_layer
        with tf.Session() as sess:
            x = sess.run(tf.cast(np.random.randn(1, 2, 2, 3), tf.float32))
            w = sess.run(tf.cast(np.ones((1, 1, 3, 1)), tf.float32))
            z = sess.run(conv_1x1_layer(x, w))
        self.assertTrue((np.sum(x, keepdims=True, axis=-1) == z).all())

    def test_exceptions_of_conv_3x3_layer(self):
        from src.forward_propagation import conv_3x3_layer
        x = np.random.randn(2, 24, 24, 3)
        w = np.random.randn(1, 1, 3, 3)
        self.assertRaises(AssertionError, conv_3x3_layer, x, w)

    def test_result_of_conv_3x3_layer(self):
        from src.forward_propagation import conv_3x3_layer
        c = 4
        with tf.Session() as sess:
            x = sess.run(tf.cast(np.ones((1, 3, 3, 1)), tf.float32))
            w = sess.run(tf.cast(np.ones((3, 3, c, 1)), tf.float32))
            z = sess.run(conv_3x3_layer(x, w))
        self.assertTrue(([[[[c * 4], [c * 6], [c * 4]],
                           [[c * 6], [c * 9], [c * 6]],
                           [[c * 4], [c * 6], [c * 4]]]] == z).all())

    def test_exceptions_of_conv_5x5_layer(self):
        from src.forward_propagation import conv_5x5_layer
        x = np.random.randn(2, 24, 24, 3)
        w = np.random.randn(3, 3, 3, 3)
        self.assertRaises(AssertionError, conv_5x5_layer, x, w)

    def test_result_of_conv_5x5_layer(self):
        from src.forward_propagation import conv_5x5_layer
        c = 4
        with tf.Session() as sess:
            x = sess.run(tf.cast(np.ones((1, 5, 5, 1)), tf.float32))
            w = sess.run(tf.cast(np.ones((5, 5, c, 1)), tf.float32))
            z = sess.run(conv_5x5_layer(x, w))
        self.assertTrue(([[[[c * 9], [c * 12], [c * 15], [c * 12], [c * 9]],
                           [[c * 12], [c * 16], [c * 20], [c * 16], [c * 12]],
                           [[c * 15], [c * 20], [c * 25], [c * 20], [c * 15]],
                           [[c * 12], [c * 16], [c * 20], [c * 16], [c * 12]],
                           [[c * 9], [c * 12], [c * 15], [c * 12], [c * 9]]]] == z).all())

    def test_result_of_max_pool_layer(self):
        from src.forward_propagation import max_pool_layer
        c = 4
        with tf.Session() as sess:
            x = sess.run(tf.cast(np.ones((1, 3, 3, 1)), tf.float32))
            z = sess.run(max_pool_layer(x))
        self.assertTrue((np.ones((1, 3, 3, c)) == z).all())

    def test_dimension_of_inception_layer(self):
        from src.forward_propagation import inception_layer
        x = tf.cast(np.random.randn(2, 24, 24, 24), tf.float32)
        w1 = tf.cast(np.random.randn(1, 1, 24, 8), tf.float32)
        w3 = tf.cast(np.random.randn(3, 3, 4, 8), tf.float32)
        w5 = tf.cast(np.random.randn(5, 5, 4, 4), tf.float32)
        with tf.Session() as sess:
            z = sess.run(inception_layer(x, w1, w3, w5))
        self.assertEqual((2, 24, 24, 24), z.shape)


if __name__ == '__main__':
    unittest.main()
