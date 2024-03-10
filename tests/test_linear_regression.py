import unittest
from models.linear_regression import LinearRegressionModel
import numpy as np

class TestLinearRegression(unittest.TestCase):
    def test_fit(self):
        # Dummy data
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 3, 4])

        # Create model
        model = LinearRegressionModel()
        model.fit(X_train, y_train)

        # Check that the model has been trained (example assertion)
        self.assertTrue(model.W is not None)
        self.assertTrue(model.b is not None)

    def test_predict(self):
        # Dummy data
        X_test = np.array([[4], [5]])
        expected_predictions = np.array([5, 6])

        # Create model
        model = LinearRegressionModel()
        model.W = np.array([1])
        model.b = 1

        # Check predictions
        predictions = model.predict(X_test)
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

if __name__ == '__main__':
    unittest.main()