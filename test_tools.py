import unittest
from tools import *

class TestTools(unittest.TestCase):

    def test_get_intercept(self):
        p0 = [0., 0.]
        p1 = [0., 4.]
        q  = [2., 2.]

        intercept, distance, linelength, error = get_intercept(p0, p1, q)
        intercept = list(intercept)

        self.assertListEqual(intercept, [0, 2])
        self.assertEqual(distance, 2)
        self.assertEqual(linelength, 4)
        self.assertLess(error, 1e-5)

        # Different order of arguments
        intercept1, distance1, linelength1, error1 = get_intercept(p1, p0, q)
        intercept1 = list(intercept1)

        self.assertListEqual(intercept, intercept1)
        self.assertEqual(distance, distance1)
        self.assertEqual(linelength, linelength1)
        self.assertEqual(error, error1)


if __name__ == '__main__':
    unittest.main()
