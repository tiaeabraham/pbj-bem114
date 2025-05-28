import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class SpreadModel:
    def __init__(self, max_iter, tol, random_state, backtest_period):
        self.backtest_period = backtest_period
        self.model = SGDRegressor(
            max_iter=max_iter, tol=tol, random_state=random_state, warm_start=True)
        self.scaler = StandardScaler()
        self.is_first_iteration = True

    def prepare_data(self, df):
        # Convert the Date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

        # Sort the data by date
        df = df.sort_values(by='Date')

        # Group by week and calculate the mean supply and spread for each week
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.year

        # Group by year and week, and calculate mean supply and spread for each week
        weekly_df = df.groupby(['Year', 'Week']).agg({
            'Supply': 'mean',
            'Spread': 'mean',
            'Date': 'first'  # We will keep the first date for reference of the week
        }).reset_index()

        # Create lagged features for the supply and spread
        weekly_df['Supply_Lag1'] = weekly_df['Supply'].shift(1)
        weekly_df['Spread_Lag1'] = weekly_df['Spread'].shift(1)

        # Drop the first row because it will have NaN values for the lagged columns
        weekly_df = weekly_df.dropna()

        return weekly_df

    def train(self, df, batch_size=1):
        weekly_df = self.prepare_data(df)

        # Create the feature matrix X and target vector y
        X = weekly_df[['Supply', 'Supply_Lag1', 'Spread_Lag1']].values
        y = weekly_df['Spread'].values
        X_scaled = self.scaler.fit_transform(X)

        X_train_initial = X_scaled[:-self.backtest_period]
        y_train_initial = y[:-self.backtest_period]

        # Iterative training in batches
        for start in range(0, len(X_train_initial), batch_size):
            end = min(start + batch_size, len(X_train_initial))
            X_batch = X_train_initial[start:end]
            y_batch = y_train_initial[start:end]

            if self.is_first_iteration:
                self.model.partial_fit(X_batch, y_batch)
                self.is_first_iteration = False
            else:
                self.model.partial_fit(X_batch, y_batch)

        return X_scaled[-self.backtest_period:], y[-self.backtest_period:], weekly_df['Date'].values[-self.backtest_period:]

    def predict(self, X_test):
        return self.model.predict(X_test)
