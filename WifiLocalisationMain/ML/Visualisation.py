import queue
import torch
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Structures.SplitData import *
from Structures.Optimiser import *
from Structures.NeuralNet import *
from ML.Structures.Model import Model
from sklearn.neighbors import KNeighborsRegressor

from Fingerprinting.Structures.Samplers import Samplers

def sampling_thread():
    global rssi
    global testing_ports
    samplers = Samplers()
    samplers.create_samplers(testing_ports, 200)
    for sampler in samplers.samplers:
        sampler.prepare_for_sampling()
    while True:
        singles = samplers.get_singles()
        if None in singles:
            continue

        # print(singles)
        rssi.put(np.array([singles]))


def model_thread():
    global rssi, testing_model, to_plot

    testing_model.eval()

    scaler = StandardScaler()

    while True:
        x = scaler.fit_transform(rssi.get())
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = testing_model(x_tensor)  # shape [1,2]

        # 4) Convert back to real-world cm
        pred_cm = scaler.inverse_transform(pred_scaled.cpu().numpy())  # shape (1,2)
        x_cm, y_cm = pred_cm[0]

        to_plot.append((x_cm, y_cm))


training_ports = [5, 6, 7]
filename = "../Datasets/Fingerprinting.csv"  # File storing RSSI readings for 3 IPS nodes
csv = pd.read_csv(filename)


filtered = std_filter(csv, training_ports, 4.5)
x_train, x_test, y_train, y_test = split_data_classical(training_ports, filtered, 0.2)

lr_filtered = std_filter(csv, training_ports, 2.9)
x_train_lr, x_test_lr, y_train_lr, y_test_lr = split_data_classical(training_ports, lr_filtered, 0.2)

lr = Model(LinearRegression(), "LR")
lr.train(x_train_lr, y_train_lr)
lr_mae, lr_rmse = lr.test(x_test_lr, y_test_lr)

knn = Model(KNeighborsRegressor(n_neighbors=26), f"KNN")
knn.train(x_train, y_train)
knn_mae, knn_rmse = knn.test(x_test, y_test)

rfr = Model(RandomForestRegressor(n_estimators=300, random_state=1), f"RFR")
rfr.train(x_train, y_train)
rfr_mae, rfr_rmse = rfr.test(x_test, y_test)

nn = UniformRSSINN(10, 484)
x_train_nn, y_train_nn, x_test_nn, y_test_nn = split_data_nn(csv, training_ports, 0.15)

start = datetime.now()
nn_train_rmse, nn_test_rmse = train_nn(nn, x_train_nn, y_train_nn, x_test_nn, y_test_nn, lr=0.00115377)
end = datetime.now()
ms = (end - start).total_seconds() * 1000
print(f"UniformRSSINN took {ms}ms to train")

print("Using following models for visualisation:")
print(f"{lr.name} with test RMSE = {lr_rmse}")
print(f"{knn.name} with test RMSE = {knn_rmse}")
print(f"{rfr.name} with test RMSE = {rfr_rmse}")
print(f"UniformRSSINN with test RMSE = {nn_test_rmse}")

testing_ports = [9, 11, 12]
testing_model = lr
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(csv["x"], csv["y"], csv["rssi_for_port5"], c='r', marker='s', label='COM_PORT5')
ax.scatter(csv["x"], csv["y"], csv["rssi_for_port6"], c='g', marker='s', label='COM_PORT6')
ax.scatter(csv["x"], csv["y"], csv["rssi_for_port7"], c='b', marker='s', label='COM_PORT7')
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("RSSI (dBm)")
ax.set_title("How distance affects RSSI")
ax.view_init(elev=30, azim=45)
ax.legend()
plt.show()

testing_ports = [12, 11, 9]
testing_model = nn
rssi = queue.Queue()
t = threading.Thread(target=sampling_thread)
t.daemon = True
t.start()
t2 = threading.Thread(target=model_thread)
t2.daemon = True
t2.start()
to_plot = []
# IPS node coordinates
ipsx = [0, 345, 173, 0]
ipsy = [0, 0, 294, 0]
while True:
    if len(to_plot) > 0:
        x = to_plot[0][0]
        y = to_plot[0][1]
        del to_plot[0]
        print(f"Predicted distance ({x},{y})")
        plt.clf()
        plt.plot(ipsx, ipsy, marker='o', linestyle='-', color='g', label="IPS")
        plt.plot(x, y, marker='o', color='purple', label="UniformRSSINN predicted position")
        plt.legend()
        plt.show()