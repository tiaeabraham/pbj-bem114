# stop_loss_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class StopLossModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        # Stop-loss logic: Binary outcome (0/1) based on negative deviation beyond 1 std dev
        residuals = y_train - X_train[:, 0]
        std_dev = np.std(residuals)
        y_train_stop_loss = (residuals < -std_dev).astype(int)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train_stop_loss)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X)
