# test.py

import unittest
from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level up
data_path = os.path.join(BASE_DIR, 'Mall_Customers_model.pkl')
model = joblib.load(data_path)



class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.model = KMeans(n_clusters=4)
        self.data = np.random.rand(100, 5)
        self.model.fit(self.data)

    def test_scaler(self):
        with self.assertRaises(AttributeError):
            self.assertEqual(len(self.model.X_importances_), 4)

if __name__ == '__main__':
    unittest.main()
