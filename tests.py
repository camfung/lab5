from pandas import read_csv
from CrossValidation_Lab5 import CrossValidation_Lab5, PolnomialModel
import unittest
import numpy as np


class Tests(unittest.TestCase):
    def test_fit_model(self):
        x = [2, 3, 5]
        y = [4, 2, 3]
        expected_values = [3.71428571, -0.21428571]
        model = PolnomialModel(1)
        [beta1, beta0] = model.fit(x, y)
        assert beta0 - expected_values[0] < 1
        assert beta1 - expected_values[1] < 1

    def test_predictions_returns(self):
        x = [2, 3]
        y = [4, 2]

        x_test = [5, 6]
        y_test = [3, 2]
        model = PolnomialModel(1)
        model.fit(x, y)
        predictions = model.predict(x_test)
        assert predictions is not None

    def test_predictions_values(self):
        x = [2, 3]
        y = [4, 2]

        x_test = [5, 6]
        a = PolnomialModel(1)
        a.fit(x, y)
        predictions = a.predict(x_test)
        assert predictions[0] - -2 < 0.01 and predictions[1] - -4 < 0.01

    def test_get_mae(self):
        x = [2, 3]
        y = [4, 2]

        x_test = [5, 6]
        y_test = [3, 2]
        model = PolnomialModel(1)
        model.fit(x, y)
        mae = model.get_mae(x_test, y_test)
        assert mae is not None

    def test_get_mae_value(self):
        x = [2, 3]
        y = [4, 2]

        x_test = [5, 6]
        y_test = [3, 2]
        a = PolnomialModel(1)
        a.fit(x, y)
        mae = a.get_mae(x_test, y_test)
        assert mae - 5.5 < 0.01

    def test_cv(self):
        x = np.array([2, 3])
        y = np.array([4, 2])

        a = CrossValidation_Lab5(x, y, 1, 2)
        b = a.poly_kfoldCV()

        assert b is not None

    def test_cv_value(self):
        x = np.array([2, 3, 5, 6])
        y = np.array([4, 2, 3, 2])

        a = CrossValidation_Lab5(x, y, 1, 2)
        _, cv_value = a.poly_kfoldCV()
        assert np.absolute(cv_value - 4) < 0.01

    def test_train_value(self):
        x = np.array([2, 3, 5, 6])
        y = np.array([4, 2, 3, 2])

        a = CrossValidation_Lab5(x, y, 1, 2)
        train_error, _ = a.poly_kfoldCV()
        assert train_error < 0.01

    def test_chien(self):
        data = read_csv("./data_lab5.csv")
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()

        a = CrossValidation_Lab5(x, y, 1, 5)
        train_error, cv_value = a.poly_kfoldCV()
        assert train_error - 1.0355 < 0.01 and cv_value - 1.0848 < 0.01


if __name__ == "__main__":
    unittest.main()
