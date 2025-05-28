# Imports
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.metrics import mean_squared_error, r2_score
from sizing_model import SizingModel  # Import your SizingModel

'''
Is this a job for an ML algorithm or fundamental statistics?

Rough idea for how this works: you run the modle many times and for each action_time it makes a rando decision.
This descision could obviously be right or wrong, but let's put a slight bias on the decision being right.
Then, the model calculates the ideal position sizing, winning the most when we are right but losing
the least when we are wrong.
'''


def evaluate_sizing_model(X, y):
    model = SizingModel()

    # Split data for training and testing (assuming X and y are numpy arrays or pandas DataFrames)
    train_size = int(0.8 * len(X))  # 80% for training
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Train the model
    model.train(X_train, y_train)

    # Predict using the model
    predictions = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Generate dates for plotting (assuming y has a time series aspect)
    # Adjust as per your data
    dates = pd.date_range(start="2022-01-01", periods=len(y_test), freq="W")

    # Plotting the results
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, subplot_titles=('Actual vs Predicted Sizing', 'Error Over Time'))

    # Plot actual vs predicted
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode='lines',
                  name='Actual Sizing'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines',
                  name='Predicted Sizing'), row=1, col=1)

    # Plot error over time
    fig.add_trace(go.Scatter(x=dates, y=y_test - predictions,
                  mode='lines', name='Prediction Error'), row=2, col=1)

    fig.update_layout(height=600, width=800,
                      title_text="Sizing Model Performance")
    fig.show()
