import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Classical models instantiate from this class (LR, KNN, RFR)
class Model:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name
        if name is None:
            self.name = f"{type(self.model)}".split(".")[-1].strip("'>")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return mae, rmse

    def predict(self, x):
        predicted_xy = self.model.predict(x).flatten()
        return predicted_xy