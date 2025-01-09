import unittest
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import DiabetesPrediction
from src.health_recommendations import HealthRecommendations
import os
import json

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.sample_data = pd.DataFrame({
            'Pregnancies': [1, 2, 3],
            'Glucose': [85, 0, 90],
            'BloodPressure': [66, 0, 70],
            'SkinThickness': [29, 0, 30],
            'Insulin': [0, 0, 155],
            'BMI': [26.6, 0, 30.1],
            'DiabetesPedigreeFunction': [0.351, 0.427, 0.672],
            'Age': [31, 22, 45],
            'Outcome': [0, 1, 1]
        })

    def test_preprocess_data(self):
        processed_df = self.preprocessor.preprocess_data(self.sample_data.copy())
        # Check that there are no zeros in specific columns
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']:
            self.assertTrue((processed_df[col] != 0).all())

    def test_scale_features(self):
        processed_df = self.preprocessor.preprocess_data(self.sample_data.copy())
        scaled_features = self.preprocessor.scale_features(processed_df, is_training=True)
        self.assertEqual(scaled_features.shape[1], 8)  # All features except Outcome
        self.assertTrue(np.abs(scaled_features.mean()).mean() < 1e-10)  # Centered around 0

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer()
        self.preprocessor = DataPreprocessor()
        # Create sample data
        self.X = np.random.randn(100, 8)
        self.y = np.random.randint(0, 2, 100)

    def test_train_model(self):
        model = self.trainer.train_model(self.X, self.y)
        self.assertIsNotNone(model)
        # Test prediction shape
        pred = model.predict(self.X[:1])
        self.assertEqual(len(pred), 1)

class TestDiabetesPrediction(unittest.TestCase):
    def setUp(self):
        self.predictor = DiabetesPrediction()
        self.sample_input = [1, 85, 66, 29, 0, 26.6, 0.351, 31]

    def test_prediction_format(self):
        result = self.predictor.predict(self.sample_input)
        self.assertIn('is_diabetic', result)
        self.assertIn('probability', result)
        self.assertIsInstance(result['is_diabetic'], bool)
        self.assertIsInstance(result['probability'], float)

class TestHealthRecommendations(unittest.TestCase):
    def setUp(self):
        # Use a dummy API key for testing
        self.health_advisor = HealthRecommendations(api_key="dummy_key")
        self.sample_patient_data = {
            'Glucose': 85,
            'BloodPressure': 66,
            'BMI': 26.6,
            'Age': 31
        }
        self.sample_prediction = {
            'is_diabetic': False,
            'probability': 0.3
        }

    def test_fallback_recommendations(self):
        recommendations = self.health_advisor._get_fallback_recommendations(is_diabetic=False)
        expected_categories = [
            'Diet and Nutrition',
            'Physical Activity',
            'Lifestyle Changes',
            'Monitoring and Prevention'
        ]
        self.assertEqual(sorted(recommendations.keys()), sorted(expected_categories))
        for category in expected_categories:
            self.assertGreater(len(recommendations[category]), 0)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()
