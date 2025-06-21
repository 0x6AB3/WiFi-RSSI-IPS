# Indoor Localisation Using Wi-Fi and Machine Learning

This project implements a real-time indoor localisation system using low-cost ESP32 microcontrollers and machine learning techniques, achieving **sub-meter accuracy** within a controlled environment. Developed as a final-year dissertation for an Integrated Master's degree in Computer Science at Cardiff University, it was awarded **First Class Honours**.

## 📍 Overview

Using the **Received Signal Strength Indicator (RSSI)** from Wi-Fi signals, the system estimates a device’s position within a triangular zone defined by three fixed ESP32 nodes. Over **300,000 RSSI samples** were collected across a fine-grained fingerprinting grid to train and evaluate classical and neural machine learning models.

## 🔧 Key Features

- Custom IPS (Indoor Positioning System) built with ESP32 boards and Python backend
- Real-time localisation via live RSSI streaming over serial and ML inference
- Fully automated data collection pipeline and grid-based fingerprinting system
- Live visualisation interface using Matplotlib for position tracking
- Evaluated multiple models:
  - 📈 Linear Regression
  - 🔍 K-Nearest Neighbours
  - 🌲 Random Forest Regressor
  - 🧠 Feedforward Neural Networks (PyTorch)

## 📊 Results

- **Best model (RFR):**
  - RMSE: **25.5 cm**
  - MAE: **15.0 cm**
- All models achieved sub-meter accuracy on test data
- Extensive use of z-score filtering, Optuna hyperparameter tuning, and GPU-accelerated neural network training

## 📁 Project Structure

- `Soft_AP.ino` – ESP32 reference node broadcasting Wi-Fi signal  
- `RSSI_Client.ino` – ESP32 clients collecting RSSI values  
- `Fingerprinting.py` – Multi-threaded data collection and CSV storage  
- `MachineLearning.py` – Model training, tuning, and evaluation  
- `Visualisation.py` – Real-time display of predicted device position  

## 🧠 Skills & Technologies

ESP32 · PySerial · NumPy · Scikit-learn · PyTorch · Matplotlib  
Regression Models · Data Preprocessing · Outlier Filtering  
Real-Time Systems · Signal Processing · Indoor Navigation

## 📹 Demo

Watch a video demonstration of live localisation:  
[📺 YouTube Demo](https://youtu.be/yXP_AVsCl2I)

## 📚 Future Work

- Integrating IEEE 802.11mc RTT for improved distance accuracy  
- Using advanced features like CSI and AoA  
- Adapting system for dynamic, real-world environments  
- Exploring probabilistic/Bayesian localisation models
