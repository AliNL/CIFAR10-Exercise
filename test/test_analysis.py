import unittest

import tensorflow as tf


class AnalysisTest(unittest.TestCase):
    def test_result_of_compute_accuracy(self):
        from src.analysis import compute_accuracy
        z = tf.constant([[0.6, 0.4, 0.8], [0.2, 0.6, 0.1]], dtype=tf.float32)
        y = tf.constant([[0, 0, 1], [1, 0, 0]], dtype=tf.float32)
        with tf.Session() as sess:
            accuracy = sess.run(compute_accuracy(z, y))
        self.assertEqual(0.5, accuracy)


if __name__ == '__main__':
    unittest.main()
