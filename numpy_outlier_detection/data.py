import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(data_path="inputs.json"):
    with open(data_path) as f:
        inputs = json.load(f)
    data = np.array(inputs[0]["value"])
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


def plot_data(x, y, title="Data"):
    df = pd.DataFrame({"x": x[:, 0], "y": y[:, 0]})
    sns.set_style("whitegrid")

    sns.scatterplot(x="x", y="y", data=df)

    plt.title(title)
    plt.show()


def plot_data_with_outliers_and_regression_line(
    x, y, outliers, coef, intercept, title="Data"
):
    df = pd.DataFrame({"x": x[:, 0], "y": y[:, 0]})
    sns.set_style("whitegrid")

    sns.scatterplot(x="x", y="y", data=df)
    line = coef * x + intercept
    plt.plot(x, line, color="red")
    sns.scatterplot(x="x", y="y", data=outliers, color="red")

    plt.title(title)
    plt.show()
