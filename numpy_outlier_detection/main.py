import numpy as np
import pandas as pd

from numpy_outlier_detection.data import (
    load_data,
    plot_data,
    plot_data_with_outliers_and_regression_line,
)
from numpy_outlier_detection.linear_regression import LinearRegressionNumpy


def distance_from_line(x, y, coef, intercept):
    return abs(y - (coef * x + intercept)) / np.sqrt(coef**2 + 1)


def z_score(data):
    return (data - data.mean()) / data.std()


def main():
    x, y = load_data()
    plot_data(x, y)

    model = LinearRegressionNumpy()
    model.fit(x, y)
    coef = model.coef
    intercept = model.intercept

    df = pd.DataFrame({"x": x[:, 0], "y": y[:, 0]})
    distances = distance_from_line(x, y, coef, intercept)
    z_scores = z_score(distances)
    outlier_indices = np.where(z_scores > 1.5)[0]
    outliers = df.iloc[outlier_indices]

    plot_data_with_outliers_and_regression_line(x, y, outliers, coef, intercept)
