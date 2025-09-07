# test.py

import unittest
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level up
data_path = os.path.join(BASE_DIR, 'Mall_Customer_model.pkl')
model = joblib.load(Mall_Customer)



class TestModelTraining(unittest.TestCase):
    def test_scaler(self):
        # model = joblib.load('Mall_Customer_model.pkl')
        self.assertIsInstance(model, KMeans)  
        self.assertEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()