# sizing_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class SizingModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        # Sizing logic: Target size based on y_train scaled between 0.5% and 20%
        y_train_sizing = np.clip(
            np.abs(y_train) / np.max(np.abs(y_train)) * 20, 0.5, 20)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train_sizing)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X)
