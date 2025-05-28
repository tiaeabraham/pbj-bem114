import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import pickle
import signal
import threading
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Hyperparameter tuning callback


class SharpeRatioCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_sharpe_ratio = -np.inf
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > 5:
            y_pred = self.model.predict(self.X_val)
            mse = mean_squared_error(self.y_val, y_pred)
            r2 = r2_score(self.y_val, y_pred)
            sharpe_ratio = self.calculate_sharpe_ratio(self.y_val, y_pred)
            self.start_time = time.time()
            print(
                f"Epoch {epoch}: MSE = {mse:.4f}, R² = {r2:.4f}, Sharpe Ratio = {sharpe_ratio:.4f}")

    def calculate_sharpe_ratio(self, y_true, y_pred):
        residuals = y_pred.flatten() - y_true
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        if std_residual == 0:
            return 0
        sharpe_ratio = mean_residual / std_residual
        return sharpe_ratio

# Neural network model


class NeuralNetworkModel:
    def __init__(self, X_train, y_train, X_val, y_val, goal_sharpe_ratio=1.0, goal_r2=0.8):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.goal_sharpe_ratio = goal_sharpe_ratio
        self.goal_r2 = goal_r2
        self.model = self.load_or_initialize_model()

    def load_or_initialize_model(self):
        model_files = [f for f in os.listdir() if f.startswith(
            'keras_model_') and f.endswith('.h5')]
        if model_files:
            latest_model_file = max(model_files, key=lambda x: int(
                x.split('_')[-1].split('.')[0]))
            print(f"Loading model from {latest_model_file}")

            # Rebuild model with the same architecture as the one used in tuning
            try:
                # hp=None to use default architecture
                model = self.build_model(hp=None)
                model.load_weights(latest_model_file)
                print("Model weights loaded successfully.")
            except ValueError as e:
                print(f"Error loading weights: {e}")
                # If weights cannot be loaded, initialize a new model
                print("Initializing a new model.")
                # This will also tune and return the best model
                model = self.tune_hyperparameters()
        else:
            print(
                "No existing model found. Initializing a new model with hyperparameter tuning.")
            model = self.tune_hyperparameters()

        return model

    def tune_hyperparameters(self):
        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=1,
            directory='tuner_dir',
            project_name='nn_tuning'
        )

        tuner.search(self.X_train, self.y_train, epochs=10,
                     validation_data=(self.X_val, self.y_val))
        best_model = tuner.get_best_models(num_models=1)[0]
        print("Best model found and trained.")
        return best_model

    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1],)))
        for i in range(hp.Int('num_layers', 2, 5)):
            model.add(
                Dense(units=hp.Int(f'units_{i}', 32, 128, step=32), activation='relu'))
            model.add(Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
        model.add(Dense(1))  # Output layer should match the target shape

        model.compile(
            optimizer=Adam(learning_rate=hp.Float(
                'learning_rate', 1e-4, 1e-2, sampling='log')),
            loss='mse'
        )
        return model

    def train(self):
        sharpe_callback = SharpeRatioCallback(self.X_val, self.y_val)
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=32,
            validation_data=(self.X_val, self.y_val),
            callbacks=[sharpe_callback]
        )

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        sharpe_ratio = self.calculate_sharpe_ratio(y_test, y_pred)
        return mse, r2, sharpe_ratio

    def calculate_sharpe_ratio(self, y_true, y_pred):
        residuals = y_pred.flatten() - y_true
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        if std_residual == 0:
            return 0
        sharpe_ratio = mean_residual / std_residual
        return sharpe_ratio

    def save_model(self):
        model_files = [f for f in os.listdir() if f.startswith(
            'keras_model_') and f.endswith('.h5')]
        if model_files:
            latest_model_file = max(model_files, key=lambda x: int(
                x.split('_')[-1].split('.')[0]))
            latest_n = int(latest_model_file.split('_')[-1].split('.')[0])
        else:
            latest_n = -1
        new_model_file = f'keras_model_{latest_n + 1}.h5'
        self.model.save(new_model_file)
        print(f"Model saved to {new_model_file}.")

# Gradient Boosting Model


class GradientBoostingModel:
    def __init__(self, goal_sharpe_ratio=1.0, goal_r2=0.8):
        self.goal_sharpe_ratio = goal_sharpe_ratio
        self.goal_r2 = goal_r2
        self.model = GradientBoostingRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        sharpe_ratio = self.calculate_sharpe_ratio(y_test, y_pred)
        return mse, r2, sharpe_ratio

    def calculate_sharpe_ratio(self, y_true, y_pred):
        residuals = y_pred - y_true
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        if std_residual == 0:
            return 0
        sharpe_ratio = mean_residual / std_residual
        return sharpe_ratio

    def save_model(self):
        model_files = [f for f in os.listdir() if f.startswith(
            'gb_model_') and f.endswith('.pkl')]
        if model_files:
            latest_model_file = max(model_files, key=lambda x: int(
                x.split('_')[-1].split('.')[0]))
            latest_n = int(latest_model_file.split('_')[-1].split('.')[0])
        else:
            latest_n = -1
        new_model_file = f'gb_model_{latest_n + 1}.pkl'
        with open(new_model_file, 'wb') as f:
            pickle.dump(self.model, f)


# Initialize plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_title('R² Value Over Time')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('R²')
ax1.set_ylim(0, 1)

ax2.set_title('Sharpe Ratio Over Time')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_ylim(-1, 5)

r2_values = []
sharpe_ratios = []
iteration = 0


def plot_metrics():
    global iteration
    ax1.clear()
    ax2.clear()
    ax1.plot(range(len(r2_values)), r2_values,
             marker='o', linestyle='-', color='b')
    ax1.set_title('R² Value Over Time')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('R²')
    ax1.set_ylim(0, 1)

    ax2.plot(range(len(sharpe_ratios)), sharpe_ratios,
             marker='o', linestyle='-', color='r')
    ax2.set_title('Sharpe Ratio Over Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_ylim(-1, 5)

    plt.pause(0.1)


def main():
    def handle_keyboard_interrupt(signum, frame):
        print("Keyboard interrupt received. Saving models...")
        if nn_model:
            nn_model.save_model()
        if gb_model:
            gb_model.save_model()
        plt.ioff()
        plt.show()
        exit(0)

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    # Ensure this file is in the correct format and path
    df = pd.read_csv('/Users/bramschork/Desktop/data.csv')

    # Print column names for debugging
    print("DataFrame columns:", df.columns)

    # Create a dummy 'target' column for testing
    # Random target values for demonstration
    df['target'] = np.random.randn(len(df))

    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    nn_model = NeuralNetworkModel(
        X_train, y_train, X_val, y_val,
        goal_sharpe_ratio=1.0, goal_r2=0.8
    )

    gb_model = GradientBoostingModel(goal_sharpe_ratio=1.0, goal_r2=0.8)

    while True:
        nn_model.train()
        mse, r2, sharpe_ratio = nn_model.evaluate(X_test, y_test)
        print(
            f"Neural network model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}\n")
        r2_values.append(r2)
        sharpe_ratios.append(sharpe_ratio)
        plot_metrics()

        if sharpe_ratio >= 1.0 and r2 >= 0.8:
            nn_model.save_model()
            break

        # Train Gradient Boosting Model
        gb_model.train(X_train, y_train)
        mse, r2, sharpe_ratio = gb_model.evaluate(X_test, y_test)
        print(
            f"Gradient Boosting model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}\n")

        r2_values.append(r2)
        sharpe_ratios.append(sharpe_ratio)
        plot_metrics()

        # Save models
        gb_model.save_model()

        # Retrain neural network with updated data
        nn_model.train()
        mse, r2, sharpe_ratio = nn_model.evaluate(X_test, y_test)
        print(
            f"Neural network model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}\n")
        r2_values.append(r2)
        sharpe_ratios.append(sharpe_ratio)
        plot_metrics()

        if sharpe_ratio >= 1.0 and r2 >= 0.8:
            nn_model.save_model()
            break


if __name__ == "__main__":
    main()
