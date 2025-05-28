import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from itertools import product


class SpreadModel:
    def __init__(self, n_components=2, covariance_type='diag', backtest_period=52, n_iter=2000, tol=1e-6):
        self.backtest_period = backtest_period
        self.model = None
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol

    def initialize_model(self, random_state=42):
        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_components, covariance_type=self.covariance_type,
                n_iter=self.n_iter, tol=self.tol, random_state=random_state)
            print(f"HMM initialized with n_components={self.n_components}, covariance_type={self.covariance_type}, "
                  f"n_iter={self.n_iter}, tol={self.tol}, random_state={random_state}")
        except KeyError as e:
            print(f"KeyError during HMM initialization: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during HMM initialization: {e}")
            raise

    def prepare_data(self, df):
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        df = df.sort_values(by='Date')
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.year

        weekly_df = df.groupby(['Year', 'Week']).agg({
            'Supply': 'mean',
            'Spread': 'mean',
            'Date': 'first'
        }).reset_index()

        weekly_df['Supply_Lag1'] = weekly_df['Supply'].shift(1)
        weekly_df['Spread_Lag1'] = weekly_df['Spread'].shift(1)

        weekly_df = weekly_df.dropna()
        return weekly_df

    def train(self, df, optimize_sharpe=False):
        weekly_df = self.prepare_data(df)
        X = weekly_df[['Supply', 'Supply_Lag1', 'Spread_Lag1']].values
        X_scaled = self.scaler.fit_transform(X)

        X_train_initial = X_scaled[:-self.backtest_period]
        X_test_initial = X_scaled[-self.backtest_period:]
        y_test_initial = weekly_df['Spread'].values[-self.backtest_period:]
        dates_initial = weekly_df['Date'].values[-self.backtest_period:]

        if not optimize_sharpe:
            try:
                self.initialize_model()
                self.model.fit(X_train_initial)
                print("Model training completed successfully.")
            except Exception as e:
                print(f"Error occurred during model training: {e}")
                raise
            return X_test_initial, y_test_initial, dates_initial

        # Optimization loop with reinitialization
        best_sharpe_ratio = -np.inf
        best_params = None
        best_random_state = None
        param_grid = {
            'n_components': [2, 3],
            'covariance_type': ['diag', 'spherical']
        }
        random_states = [42, 100, 123]  # Different seeds for reinitialization

        for params in product(*param_grid.values()):
            for random_state in random_states:
                try:
                    self.n_components = params[0]
                    self.covariance_type = params[1]
                    self.initialize_model(random_state=random_state)
                    self.model.fit(X_train_initial)
                    predictions = self.predict(X_test_initial)

                    # Simulate trading to calculate Sharpe Ratio
                    sharpe_ratio = self.calculate_sharpe_ratio(
                        y_test_initial, predictions, dates_initial)
                    if sharpe_ratio > best_sharpe_ratio:
                        best_sharpe_ratio = sharpe_ratio
                        best_params = params
                        best_random_state = random_state

                except Exception as e:
                    print(
                        f"Error occurred during optimization with params {params} and random_state {random_state}: {e}")
                    continue

        print(
            f"Best parameters: {best_params} with Sharpe Ratio: {best_sharpe_ratio:.4f} using random_state {best_random_state}")

        if best_params is None:
            print("Model failed to converge with any parameter combination.")
            return X_test_initial, y_test_initial, dates_initial

        # Re-train with best parameters and random state
        self.n_components = best_params[0]
        self.covariance_type = best_params[1]
        self.initialize_model(random_state=best_random_state)
        self.model.fit(X_train_initial)

        return X_test_initial, y_test_initial, dates_initial

    def predict(self, X_test):
        try:
            hidden_states = self.model.predict(X_test)
            predictions = np.dot(self.model.transmat_, self.model.means_[:, 0])[
                hidden_states]
            print("Prediction completed successfully.")
            return predictions
        except Exception as e:
            print(f"Error occurred during prediction: {e}")
            raise

    def calculate_sharpe_ratio(self, y_test, predictions, dates):
        # Replace with your actual trading simulation and Sharpe Ratio calculation
        weekly_balance = []
        for i in range(len(y_test) - 1):
            # Simple trading strategy: buy if prediction is greater than actual, else sell
            action = 1 if predictions[i] > y_test[i] else -1
            profit_loss = action * (y_test[i + 1] - y_test[i])
            weekly_balance.append(profit_loss)

        if len(weekly_balance) == 0:
            return -np.inf

        returns = np.array(weekly_balance) / np.abs(weekly_balance[0])
        average_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = average_return / return_std
        return sharpe_ratio
