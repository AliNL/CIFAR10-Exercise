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


if __name__ == '__main__':
    unittest.main()
