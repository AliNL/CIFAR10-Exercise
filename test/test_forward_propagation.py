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
        with tf.Session() as sess:
            x = sess.run(tf.cast(np.ones((1, 3, 3, 1)), tf.float32))
            w = sess.run(tf.cast(np.ones((3, 3, 16, 1)), tf.float32))
            z = sess.run(conv_3x3_layer(x, w))
        self.assertTrue(([[[[16 * 4], [16 * 6], [16 * 4]],
                           [[16 * 6], [16 * 9], [16 * 6]],
                           [[16 * 4], [16 * 6], [16 * 4]]]] == z).all())


if __name__ == '__main__':
    unittest.main()
