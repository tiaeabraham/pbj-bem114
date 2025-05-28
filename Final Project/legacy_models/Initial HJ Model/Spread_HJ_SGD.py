'''
Stochastic Gradient Descent Model for NYMEX RBOB.
Trained on S. Schork's supply surplus/defecit calculation and spread of March/April contracts.

Author: Bram Schork
Version: V0.1
Date: 08/21/2024

Current Model Stats
    Trained on weekly data: 10/01/1999 to 08/16/2024

    52-Week:
        Variance: 0.001561
        Portfolio 1 Return: 443.658%
        Portfolio 2 Return: 63.488%

    Notes:
        - Portfolio 1 return drops drasticlly (into negatives) for longer time frames
        - Both models have large plataeu time periods, likely due to overfitting to periods of high-volatility

ToDo:
 - Train model on day-to-day quotes
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Backtest period
backtest_period = 52

# Load the CSV file
df = pd.read_csv('/Users/bramschork/Desktop/data.csv')

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

# Create the feature matrix X and target vector y
X = weekly_df[['Supply', 'Supply_Lag1', 'Spread_Lag1']].values
y = weekly_df['Spread'].values
dates = weekly_df['Date'].values

# Standardize the data (scaling is important for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (e.g., last backtest_period weeks are for testing)
X_train_initial = X_scaled[:-backtest_period]
y_train_initial = y[:-backtest_period]

X_test = X_scaled[-backtest_period:]
y_test = y[-backtest_period:]

# Get the corresponding test dates manually
test_dates = dates[-backtest_period:]

# --- Incremental Learning and Backtest for Two Portfolios ---
initial_capital = 1000


# Portfolio 2: Buy/Hold/Sell strategy
capital_2 = initial_capital
capital_performance_over_time_2 = [100]  # Start at 100%
# Keep track of the last action for the second portfolio (None, 'Buy', 'Sell')
last_action = None

predictions = []
r2_values = []  # Track R² values over time

# Start with the initial training data
X_train = X_train_initial
y_train = y_train_initial

print("\nBuy/Sell Simulation for Portfolio 2:")
for i in range(backtest_period):
    # Initialize the model and train on the current training set
    model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)

    # Predict the spread for the current test week
    next_week_prediction = model.predict(X_test[i].reshape(1, -1))[0]
    predictions.append(next_week_prediction)

    if i > 0:  # We need at least 2 data points to compare spreads
        # Determine the action for this week for both portfolios
        if predictions[i] < predictions[i - 1]:
            action_2 = "Sell"
        else:
            action_2 = "Buy"

        # --- Portfolio 2: Buy/Hold/Sell ---
        if last_action is None:  # First action
            if action_2 == "Sell":
                sold_amount_2 = capital_2 * 0.10
                capital_2 += sold_amount_2 * \
                    ((y_test[i - 1] - y_test[i]) / y_test[i - 1])
                last_action = "Sell"
            else:
                bought_amount_2 = capital_2 * 0.10
                capital_2 += bought_amount_2 * \
                    ((y_test[i] - y_test[i - 1]) / y_test[i - 1])
                last_action = "Buy"
        elif action_2 != last_action:  # Change position
            if action_2 == "Sell":
                sold_amount_2 = capital_2 * 0.10
                capital_2 += sold_amount_2 * \
                    ((y_test[i - 1] - y_test[i]) / y_test[i - 1])
                last_action = "Sell"
            else:
                bought_amount_2 = capital_2 * 0.10
                capital_2 += bought_amount_2 * \
                    ((y_test[i] - y_test[i - 1]) / y_test[i - 1])
                last_action = "Buy"
        # If action_2 == last_action, we hold (do nothing)

        # Calculate performance for Portfolio 2
        performance_percentage_2 = (capital_2 / initial_capital) * 100
        capital_performance_over_time_2.append(performance_percentage_2)

        # Calculate the R² value for the current week
        r2_current = r2_score(y_test[:i+1], predictions[:i+1])
        r2_values.append(r2_current)

        # Print weekly results with R² value
        # print(
        #   f"              Portfolio 2 Action: {last_action}, Portfolio 2 Performance: {performance_percentage_2:.2f}%")

    # Update the training set with the latest data point
    # Add the new test feature
    X_train = np.vstack([X_train, X_test[i].reshape(1, -1)])
    y_train = np.append(y_train, y_test[i])  # Add the new actual outcome

# --- Plotly Visualization with Triple Y-Axes ---
fig = go.Figure()

# Add actual spread values (left y-axis)
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_test,
    mode='lines+markers',
    name='Actual Spread',
    line=dict(color='blue'),
    marker=dict(size=8),
    yaxis="y1"
))

# Add predicted spread values (left y-axis)
fig.add_trace(go.Scatter(
    x=test_dates,
    y=predictions,
    mode='lines+markers',
    name='Predicted Spread',
    line=dict(color='red', dash='dash'),
    marker=dict(size=8),
    yaxis="y1"
))


# Add portfolio 2 performance over time (right y-axis)
fig.add_trace(go.Scatter(
    # Capital performance over time starts from the second week
    x=test_dates[1:],
    # Now displaying portfolio performance as a percentage
    y=capital_performance_over_time_2[1:],
    mode='lines+markers',
    name='Portfolio 2 Performance (%)',
    line=dict(color='orange'),
    marker=dict(size=8),
    yaxis="y2"
))

# Add R² values over time (third y-axis)
fig.add_trace(go.Scatter(
    x=test_dates[1:],  # R² starts from the second week
    y=r2_values,
    mode='lines',
    name='R² Over Time',
    line=dict(color='purple', dash='dot'),
    opacity=0.6,
    yaxis="y3"
))

# Update layout to include triple y-axes with better positioning and visibility
fig.update_layout(
    title="Actual vs Predicted Spread, Portfolio Performances, and R² Over Time",
    xaxis_title="Date",
    yaxis=dict(title="Spread ($)", side="left", showgrid=False),
    yaxis2=dict(title="Portfolio Performance (%)",
                overlaying="y", side="right", showgrid=False),
    yaxis3=dict(
        title="R² Value",
        overlaying="y",
        side="right",
        position=0.95,  # Move the third axis further inward for better visibility
        tickmode="auto",  # Ensure ticks are auto-generated
        tickformat=".2f",  # Ensure proper formatting for the tick marks
        showgrid=False,  # Disable grid lines for this axis
        # Make the tick labels purple for better visibility
        tickfont=dict(color="purple"),
        # Make the title font purple to match the line color
        titlefont=dict(color="purple")
    ),
    legend_title="Legend"
)

########## NEXT WEEK PREDICTION ##########
# Prepare features for next week's prediction
last_supply = weekly_df['Supply'].iloc[-1]
last_supply_lag1 = weekly_df['Supply_Lag1'].iloc[-1]
last_spread_lag1 = weekly_df['Spread_Lag1'].iloc[-1]

# Create a feature array for the next week's prediction
next_week_features = np.array(
    [[last_supply, last_supply_lag1, last_spread_lag1]])

# Standardize the features using the same scaler
next_week_features_scaled = scaler.transform(next_week_features)

# Predict the spread for next week
next_week_prediction = model.predict(next_week_features_scaled)[0]

# Output the predicted spread for next week
print(f"\nPredicted Spread for Next Week: {next_week_prediction:.4f}")
########## NEXT WEEK PREDICTION ##########


#### ALL THE END STATS ####
# Initialize statistics tracking
# Initialize Portfolio 2 statistics
total_actions_portfolio_2 = 0
buy_actions_portfolio_2 = 0
sell_actions_portfolio_2 = 0
profitable_buy_actions_portfolio_2 = 0
profitable_sell_actions_portfolio_2 = 0
buy_profits_portfolio_2 = []
sell_profits_portfolio_2 = []
within_1_stddev_portfolio_2 = 0
within_1_stddev_and_direction_correct_portfolio_2 = 0
direction_correct_portfolio_2 = 0

# Calculate the final R² score after all predictions
final_r2 = r2_score(y_test, predictions)

# Print initial and final capital
print(f"\nInitial Capital: ${initial_capital}")
print(f"Final Capital Portfolio 2: ${capital_2:.2f}")

# Print the final R² score
print(f"\nFinal R² Score: {final_r2:.3f}")

# Calculate residuals and variance of model's predictions
residuals = y_test - np.array(predictions)
variance = np.var(residuals)
std_dev = np.std(residuals)

# Loop over all actions to calculate the statistics for Portfolio 2
for i in range(1, len(predictions)):  # Start from second week (as we need i-1 data)
    # Increment total actions
    total_actions_portfolio_2 += 1

    # Portfolio 2 - Buy Action
    if predictions[i] > predictions[i - 1]:
        buy_actions_portfolio_2 += 1
        profit = (y_test[i] - y_test[i - 1]) / \
            y_test[i - 1] * 100  # Percentage profit
        buy_profits_portfolio_2.append(profit)

        if profit > 0:
            profitable_buy_actions_portfolio_2 += 1

    # Portfolio 2 - Sell Action
    elif predictions[i] < predictions[i - 1]:
        sell_actions_portfolio_2 += 1
        profit = (y_test[i - 1] - y_test[i]) / \
            y_test[i - 1] * 100  # Percentage profit
        sell_profits_portfolio_2.append(profit)

        if profit > 0:
            profitable_sell_actions_portfolio_2 += 1

    # Check if prediction is within 1 standard deviation
    if abs(predictions[i] - y_test[i]) <= std_dev:
        within_1_stddev_portfolio_2 += 1

    # Check if direction was correct (both predictions and actual values moved in the same direction)
    if (predictions[i] - predictions[i - 1]) * (y_test[i] - y_test[i - 1]) > 0:
        direction_correct_portfolio_2 += 1

        # Check if within 1 standard deviation and direction was correct
        if abs(predictions[i] - y_test[i]) <= std_dev:
            within_1_stddev_and_direction_correct_portfolio_2 += 1

# --- Print the statistics for Portfolio 2 ---
print(f"\n--- Portfolio 2 Statistics ---")
# Print variance of model's predictions
print(f"\nVariance of Model's Predictions: {variance:.6f}\n")
print(f"Number of total actions/trades: {total_actions_portfolio_2}")
print(f"Number of buy actions: {buy_actions_portfolio_2}")
print(f"Number of sell actions: {sell_actions_portfolio_2}")

# Profitable actions and average profit
print(
    f"Number of profitable buy actions: {profitable_buy_actions_portfolio_2}")
print(
    f"Number of profitable sell actions: {profitable_sell_actions_portfolio_2}")
print(
    f"Average buy action profit as a percent: {np.mean(buy_profits_portfolio_2):.2f}%")
print(
    f"Average sell action profit as a percent: {np.mean(sell_profits_portfolio_2):.2f}%")

# Calculate win rate percentages
total_profitable_actions_portfolio_2 = profitable_buy_actions_portfolio_2 + \
    profitable_sell_actions_portfolio_2
total_actions_portfolio_2 = buy_actions_portfolio_2 + sell_actions_portfolio_2
print(
    f"Total action win rate: {(total_profitable_actions_portfolio_2 / total_actions_portfolio_2) * 100:.2f}%")

# Percentage of actions within 1 standard deviation
print(
    f"Percentage of actions within 1 standard deviation: {(within_1_stddev_portfolio_2 / total_actions_portfolio_2) * 100:.2f}%")

# Percentage of actions where the direction was correct
print(
    f"Percentage of actions where direction was correct: {(direction_correct_portfolio_2 / total_actions_portfolio_2) * 100:.2f}%")

# Percentage of actions within 1 standard deviation and direction was correct
print(
    f"Percentage of actions within 1 standard deviation and direction correct: {(within_1_stddev_and_direction_correct_portfolio_2 / total_actions_portfolio_2) * 100:.2f}%")


# Show plot
fig.show()
