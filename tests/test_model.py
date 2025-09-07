# test.py

import unittest
from sklearn.cluster import KMeans
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level up
data_path = os.path.join(BASE_DIR, 'Mall_Customers_model.pkl')
model = joblib.load(data_path)



class TestModelTraining(unittest.TestCase):
    def test_scaler(self):
        # model = joblib.load(data_path)
        self.assertIsInstance(model, KMeans)  
        self.assertEqual(len(model.X_importances_), 4)

if __name__ == '__main__':
    unittest.main()