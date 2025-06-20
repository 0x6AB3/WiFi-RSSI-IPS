import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
from ML.Structures.Model import Model
from .SplitData import split_data_classical
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Removes outliers that are beyond the given STD value
def std_filter(csv, ports, std_value):
    std = csv[[f"rssi_for_port{port}" for port in ports]].apply(zscore)
    filtered = csv[(std.abs() < std_value).all(axis=1)]
    filtered.to_csv("Fingerprinting_Filtered.csv", index=False)
    return filtered

# Trains multiple classical models with different STD threshold values
def find_best_std(csv, models, ports, min_std=1, max_std=5, increment=0.5):
    std_thresholds = [float(std) for std in np.arange(min_std, max_std+increment, increment)]
    columns = ["Model", "STD", "MAE", "RMSE"]
    results = pd.DataFrame(columns=columns)

    for std in std_thresholds:
        print(f"STD: {std}")
        filtered_csv = std_filter(csv, ports, std)
        x_train, x_test, y_train, y_test = split_data_classical(ports, filtered_csv, 0.2)
        for model in models:
            model.train(x_train, y_train)
            mae, rmse = model.test(x_test, y_test)
            results.loc[len(results)] = [model.name, std, mae, rmse]
    return results

def time_to_train(model, x_train, y_train):
    start = datetime.now()
    model.train(x_train, y_train)
    end = datetime.now()
    ms = (end - start).total_seconds() * 1000
    return ms

def optimise_lr_std(csv, ports):
    lr = Model(LinearRegression(), "LR")
    lr_results = find_best_std(csv, [lr], ports, increment=0.1)
    plot_optimisation(lr_results, lr.name)

    best_row = lr_results.loc[lr_results["RMSE"].idxmin()]
    best_std = float(best_row["STD"])

    filtered = std_filter(csv, ports, best_std)
    x_train, x_test, y_train, y_test = split_data_classical(ports, filtered, 0.2)

    ms = time_to_train(lr, x_train, y_train)
    print(f"Best LR (STD={best_std}) trained in {ms}ms")
    return best_std

# KNN hyperparameter optimisation
def optimise_knn(csv, ports, max_neighbors=30):
    knns = [Model(KNeighborsRegressor(n_neighbors=k), f"k={k}") for k in range(1, max_neighbors + 1)]
    results = find_best_std(csv, knns, ports)
    plot_optimisation(results, "KNN", "neighbour count")

    best_k = int(results.loc[results["RMSE"].idxmin()]["Model"][2:])
    best_knn = Model(KNeighborsRegressor(n_neighbors=best_k), f"k={best_k}")

    best_row = results.loc[results["RMSE"].idxmin()]
    best_std = float(best_row["STD"])

    filtered = std_filter(csv, ports, best_std)
    x_train, x_test, y_train, y_test = split_data_classical(ports, filtered, 0.2)

    ms = time_to_train(best_knn, x_train, y_train)
    print(f"Best KNN ({best_knn.name}, STD={best_std}) trained in {ms}ms")
    return best_knn

# RFR hyperparameter optimisation
def optimise_rfr(csv, ports, max_estimators=1000, increment=200):
    rfrs = [Model(RandomForestRegressor(n_estimators=n, random_state=1), f"n={n}")
            for n in range(increment, max_estimators + increment, increment)]
    results = find_best_std(csv, rfrs, ports)
    plot_optimisation(results, "RFR", "estimator count")

    best_row = results.loc[results["RMSE"].idxmin()]
    best_estimators = int(best_row["Model"][2:])
    best_std = float(best_row["STD"])

    filtered = std_filter(csv, ports, best_std)
    x_train, x_test, y_train, y_test = split_data_classical(ports, filtered, 0.2)

    best_rfr = Model(RandomForestRegressor(n_estimators=best_estimators, random_state=1), f"n={best_estimators}")

    ms = time_to_train(best_rfr, x_train, y_train)
    print(f"Best RFR ({best_rfr.name}, STD={best_std}) trained in {ms}ms")
    return best_rfr

# Unified plotting for any classical model type
def plot_optimisation(results, model_name, description=None):
    stds = results["STD"].unique()
    models = results["Model"].unique()

    for model in models:
        subset = results[results["Model"] == model]
        rmse_values = [subset[subset["STD"] == std]["RMSE"].values[0] if not subset[subset["STD"] == std].empty else None for std in stds]
        plt.plot(stds, rmse_values, label=model)

    plt.xlabel("Standard deviation threshold")
    plt.ylabel(f"{model_name} RMSE (cm)")
    if description is None:
        plt.title(f"How STD filtering affects {model_name} RMSE")
    else:
        plt.title(f"How {description} and STD threshold affect {model_name} RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n{model_name} results")
    min_mae = results.loc[results["MAE"].idxmin()]
    min_rmse = results.loc[results["RMSE"].idxmin()]
    print("Lowest MAE:\n", min_mae)
    print("Lowest RMSE:\n", min_rmse)
    return min_mae, min_rmse
