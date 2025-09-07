import unittest
import joblib
from sklearn.preprocessing import StandardScale
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level up
data_path = os.path.join(BASE_DIR, 'Mall_Customer_model.pkl')
model = joblib.load(Mall_Customer)



class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        # model = joblib.load('Mall_Customer_model.pkl')
        self.assertIsInstance(model, KMeans)  
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()