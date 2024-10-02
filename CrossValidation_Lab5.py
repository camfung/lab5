from typing import List, Tuple
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


class PolnomialModel:
    def __init__(self, p) -> None:
        self.p = p
        self._coeficients = None

    @property
    def coeficients(self):
        return self._coeficients

    @coeficients.setter
    def coeficients(self, value):
        self._coeficients = value

    def fit(self, x, y):
        """
        fit a polynomial of degree p to the dataset using numpy
        return the coefficients of the polynomial
        """

        self._coeficients = np.polyfit(x, y, self.p)

        return self._coeficients

    def predict(self, x_values) -> np.ndarray:
        """
        return the predicitons of the current model on the test data
        """
        if self._coeficients is None:
            raise Exception("Model needs to be trained before it can make prediction.")
        return np.polyval(self._coeficients, x_values)

    def get_mae(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(np.abs(y_test - predictions))


class CrossValidation_Lab5:

    def __init__(self, x: np.ndarray, y: np.ndarray, p_value=1, k=5) -> None:
        if len(x) != len(y):
            raise Exception("x and y must have the same number of elements")
        self.x = x
        self.y = y
        self.p = p_value
        self.k = k

    def poly_kfoldCV(self):
        x_folds = np.array_split(self.x, self.k)
        y_folds = np.array_split(self.y, self.k)
        cv_maes = 0
        train_maes = 0

        for i, x_fold in enumerate(x_folds):
            y_fold = y_folds[i]

            x_train_set = [item for x in x_folds if x is not x_fold for item in x]
            x_test_set = x_fold

            y_train_set = [item for y in y_folds if y is not y_fold for item in y]
            y_test_set = y_fold

            model = PolnomialModel(self.p)
            model.fit(x_train_set, y_train_set)

            train_mae = model.get_mae(x_train_set, y_train_set)
            test_mae = model.get_mae(x_test_set, y_test_set)

            cv_maes += test_mae
            train_maes += train_mae

        return train_maes / self.k, cv_maes / self.k


def part_3():
    data = read_csv("./data_lab5.csv")
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    p_values = range(1, 15)
    train_errors = []
    cv_values = []
    for p in p_values:
        a = CrossValidation_Lab5(x, y, p, 5)
        train_error, cv_value = a.poly_kfoldCV()
        train_errors.append(train_error)
        cv_values.append(cv_value)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, train_errors, label="Train Errors", marker="o")
    plt.plot(p_values, cv_values, label="Cv Values", marker="s")

    # Add title and labels
    plt.title("Line Graph of Two Lines")
    plt.xlabel("p values")
    plt.ylabel("error")

    # Add a legend to distinguish between the two lines
    plt.legend()

    best_p_value = 5
    plt.annotate(
        "Best p value",
        xy=(best_p_value, cv_values[best_p_value - 1]),
        xytext=(best_p_value + 2, cv_values[best_p_value - 1] + 0.05),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=10,
    )

    # Show the plot
    plt.grid(True)
    plt.show()


def part_4():
    data = read_csv("./data_lab5.csv")
    p_values = [2, 7, 10, 16]  # Polynomial degrees
    k = 5
    N = range(20, 105, 5)
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()

    # Set up subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns
    axes = axes.flatten()  # Flatten to easily iterate over

    # Define line styles and markers
    line_styles = ["-", "--", "-.", ":"]
    colors = ["blue", "green", "red", "purple"]
    markers = ["o", "s", "D", "^"]

    for i, p in enumerate(p_values):
        train_errors = []
        cv_values = []

        for n in N:
            # Pass 'p' instead of '[p]'
            train_error, cv_value = CrossValidation_Lab5(
                x[:n], y[:n], p, k
            ).poly_kfoldCV()
            train_errors.append(train_error)
            cv_values.append(cv_value)

        # Plot for each degree p on its corresponding subplot
        ax = axes[i]

        # Plot train error
        ax.plot(
            N,
            train_errors,
            label=f"Train Error (p={p})",
            color=colors[i],
            linestyle=line_styles[i],
            marker=markers[i],
            markersize=8,
            linewidth=2,
        )
        # Plot cross-validation error
        ax.plot(
            N,
            cv_values,
            label=f"CV Error (p={p})",
            color=colors[i],
            linestyle=line_styles[i],
            marker=markers[i],
            markersize=8,
            linewidth=2,
            alpha=0.7,
        )

        # Title and labels
        if p == 2:
            ax.set_title(
                f"Errors for p = {p}\nHighest Bias", fontsize=14, weight="bold"
            )
        elif p == 16:
            ax.set_title(
                f"Errors for p = {p}\nHighest variance", fontsize=14, weight="bold"
            )
        else:
            ax.set_title(f"Errors for p = {p}", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Data Points (n)", fontsize=12)
        if i % 2 == 0:  # Only label y-axis for the first plot in each row
            ax.set_ylabel("Error", fontsize=12)

        # Set y-axis limits from 0 to 2
        ax.set_ylim(0, 2)

        # Grid
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Legend
        ax.legend(fontsize=10)

    # Improve layout
    plt.tight_layout()
    plt.show()


part_3()
part_4()
