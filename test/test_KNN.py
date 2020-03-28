import unittest
import KNN_classical


class TestCase(unittest.TestCase):
    def test_voting(self):
        train = [
            (0, 'normal'),
            (1, 'normal'),
            (2, 'normal'),
            (3, 'anomaly'),
            (4, 'anomaly')
        ]

        instance = [(0.5, 'normal'), (1.9, 'normal'), (3.5, 'anomaly'), (4.5, 'anomaly')]

        for i in instance:
            vote = KNN_classical.vote(i, train, 3)
            self.assertEqual(vote, i[-1])


if __name__ == '__main__':
    unittest.main()
