import sys
import time
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from Structures.SplitData import *
from Structures.Optimiser import *
from Structures.NeuralNet import *
from Structures.Model import Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

if torch.cuda.is_available():
    print("NN training will use GPU:", torch.cuda.get_device_name(0))
    torch.device("cuda")
else:
    print("GPU not found, CPU will be used for NN training")

ports = [5, 6, 7]
filename = "../Datasets/Fingerprinting.csv"  # File storing RSSI readings for 3 IPS nodes
csv = pd.read_csv(filename)
knn_filtered = std_filter(csv, ports, 4.5)
x_train_knn, x_test_knn, y_train_knn, y_test_knn = split_data_classical(ports, knn_filtered, 0.2)

max_k = 30
print(f"Optimising neighbour count for KNN (1-{max_k} neighbours)...")
knn_best = optimise_knn(csv, ports, max_k)
print(f"Using {knn_best.name} for evaluation")


max_estimators = 1000
estimator_increment = 100
print(f"Optimising estimator count for RFR (1-{max_estimators} estimators)...")
rfr_best = optimise_rfr(csv, ports, max_estimators, estimator_increment)
split_data_classical(ports, csv, 0.2)

best_lr_std = optimise_lr_std(csv, ports)
lr_filtered_csv = std_filter(csv, ports, best_lr_std)
x_train_lr, x_test_lr, y_train_lr, y_test_lr = split_data_classical(ports, csv, 0.2)
lr = Model(LinearRegression(), "LR")
lr.train(x_train_lr, y_train_lr)
best_lr_mae, best_lr_rmse = lr.test(x_test_lr, y_test_lr)

x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = split_data_nn(csv, ports, 0.2)

print("Running Random NN Optuna Study...")
random_study = optuna.create_study(direction="minimize")
random_study.optimize(
    lambda trial: randomObjective(trial, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor),
    n_trials=30
)
print("Best Random NN Params:", random_study.best_params)
print("Best Random NN Loss:", random_study.best_value)

# todo std filtering nn
x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = split_data_nn(csv, ports, 0.2)


print("\nRunning Pyramid NN Optuna Study...")
pyramid_study = optuna.create_study(direction="minimize")
pyramid_study.optimize(
    lambda trial: pyramidObjective(trial, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor),
    n_trials=30
)
print("Best PyramidNN Params:", pyramid_study.best_params)
print("Best PyramidNN Loss:", pyramid_study.best_value)
