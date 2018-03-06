import unittest

import numpy as np


class InitializationTest(unittest.TestCase):
    def test_result_of_format_x_data(self):
        from src.initialization import format_x_data
        x = np.random.randint(256, size=(2, 3072))
        x_formatted = format_x_data(x)
        self.assertEqual(x[1][0], x_formatted[1][0][0][0])
        self.assertEqual(x[0][32], x_formatted[0][1][0][0])
        self.assertEqual(x[0][1], x_formatted[0][0][1][0])
        self.assertEqual(x[0][1024], x_formatted[0][0][0][1])

    def test_result_of_format_y_data(self):
        from src.initialization import format_y_data
        y = [3, 2, 8]
        y_formatted = format_y_data(y)
        self.assertTrue(([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] == y_formatted).all())

    def test_result_of_initialize_parameters(self):
        from src.initialization import initialize_parameters
        parameters = initialize_parameters()
        self.assertEqual((1, 1, 128, 64), parameters['w11'].shape)
        self.assertEqual((3, 3, 16, 32), parameters['W13'].shape)
        self.assertEqual((5, 5, 16, 16), parameters['w15'].shape)
        self.assertEqual((1, 1, 128, 64), parameters['w21'].shape)
        self.assertEqual((3, 3, 16, 32), parameters['W23'].shape)
        self.assertEqual((5, 5, 16, 16), parameters['w25'].shape)

    def test_result_of_random_mini_batches_without_last_batch(self):
        from src.initialization import random_mini_batches
        x = np.array(range(10)).reshape(10, 1)
        y = np.array(range(10)).reshape(10, 1)
        mini_batches = random_mini_batches(x, y, 2)
        self.assertEqual(5, len(mini_batches))
        sum_y = 0
        for mini_batch in mini_batches:
            this_x, this_y = mini_batch
            self.assertTrue((this_x == this_y).all())
            sum_y += np.sum(this_y)
        self.assertEqual(45, sum_y)

    def test_result_of_random_mini_batches_with_last_batch(self):
        from src.initialization import random_mini_batches
        x = np.array(range(10)).reshape(10, 1)
        y = np.array(range(10)).reshape(10, 1)
        mini_batches = random_mini_batches(x, y, 3)
        self.assertEqual(4, len(mini_batches))
        sum_y = 0
        for mini_batch in mini_batches:
            this_x, this_y = mini_batch
            self.assertTrue((this_x == this_y).all())
            sum_y += np.sum(this_y)
        self.assertEqual(45, sum_y)


if __name__ == '__main__':
    unittest.main()
