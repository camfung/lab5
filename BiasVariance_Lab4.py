from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np


# plot the dist using a hist
class BiasVariance_Lab4:
    def __init__(
        self, p_values=[1, 3, 5, 9, 15], n=1000, test_bias_x_value=5, data=[([], [])]
    ) -> None:
        self.p_values = p_values
        self.n = n
        self.beta_values = []
        self.data: List[Tuple[np.ndarray, np.ndarray]] = data
        self.test_bias_x_value = test_bias_x_value

    def fit_model(self, data_set: Tuple[np.ndarray, np.ndarray], p: int):
        """
        fit a polynomial of degree p to the dataset using numpy
        return the coefficients of the polynomial
        """
        x_values = data_set[0]
        y_values = data_set[1]

        model = np.polyfit(x_values, y_values, p)

        return model

    def fit_models(self):
        """
        get all the b values for each dataset
        """
        self.beta_values = []
        for index, data_set in enumerate(self.data):
            self.beta_values.append([])
            for p in self.p_values:
                betas = self.fit_model(data_set, p)
                self.beta_values[index].append(betas)

    def model_predict(self, x_values, betas):
        """
        return the predicitons of the current model on the test data
        """
        return np.polyval(betas, x_values)

    def models_predict(self):
        """
        calculate predtions for each model
        """
        self.predictions = []
        for i, trial in enumerate(self.data):
            self.predictions.append([])
            betas = self.beta_values[i]

            for weights in betas:
                x = None
                if self.test_bias_x_value is None:
                    x = trial[0]
                else:
                    x = self.test_bias_x_value
                y_pred = self.model_predict(x, weights)
                self.predictions[i].append(y_pred)

    def get_y_pred_hat(self, p_value):
        p_value -= 1
        sum = 0
        for row in self.predictions:
            sum += row[p_value]
        average = sum / len(self.predictions)

        return average

    def compute_variance(self, p_value):
        """
        compute the variance of all the ypreds for a particutlar value of x
        Var(ypred(x=5)) = 1/1000 * (sum m = 1 to 1000 (ypredm(x=5) - ypred(x=5) )^2 )
        """
        y_pred_hat = self.get_y_pred_hat(p_value)
        variance = 0
        for y_pred in self.predictions[p_value]:
            variance += y_pred - y_pred_hat
        self.variance = math.pow((1 / 1000) * variance, 2)

    def compute_bias(self, p_value):
        """
        compute the bias of all the data sets for a particular x
        """
        # ypred(x=5) - f (x=5)
        # ypred(x=5) = 1/1000 * (sum m = 1 to 1000 (ypredm(x=5)))
        self.bias = self.get_y_pred_hat(p_value) - lab4.f(self.test_bias_x_value)

    def create_histogram(self, data, p, ypred_mean, fun_x):
        fig, ax = plt.subplots()

        ax.hist(data, bins=15, color="lightblue", edgecolor="black")

        ax.set_xlabel(r"$y^{(pred)} (x=5)$")
        ax.set_ylabel("Counts")
        ax.set_title(f"Histogram for p = {p}")

        ax.set_ylim([0, 300])
        ax.set_yticks(np.arange(0, 301, 50))

        ax.set_xlim((3.5, 6.5))

        ax.axvline(ypred_mean, color="red", linestyle="-", label=f"mean of ypred(x=5)")
        ax.axvline(fun_x, color="black", linestyle="-", label=f"f(x=5)")

        ax.legend(loc="upper left")

        plt.show()


def main():
    for p_value in range(1, 6):
        a = BiasVariance_Lab4(test_bias_x_value=5)
        a.fit_models()
        a.models_predict()
        a.compute_variance(p_value)
        a.compute_bias(p_value)
        a.create_histogram(
            [pred[p_value - 1] for pred in a.predictions],
            p_value,
            a.get_y_pred_hat(p_value),
            lab4.f(5),
        )


if __name__ == "__main__":
    main()
