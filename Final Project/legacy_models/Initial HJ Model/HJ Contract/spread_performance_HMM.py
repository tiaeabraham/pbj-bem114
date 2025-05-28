# Imports
from sklearn.metrics import mean_squared_error, r2_score
from spread_model_HMM import SpreadModel  # Import your HMM model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


def calculate_sharpe_ratio(weekly_balance, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio of the trading strategy.

    Args:
        weekly_balance (list): A list of weekly profit or loss values from the trading simulation.
        risk_free_rate (float): The risk-free rate of return. Default is 0.

    Returns:
        float: The Sharpe Ratio of the trading strategy.
    """

    # Convert weekly balances to returns
    # Assuming initial capital is the absolute value of the first week's balance
    returns = np.array(weekly_balance) / np.abs(weekly_balance[0])

    # Calculate average return
    average_return = np.mean(returns)

    # Calculate standard deviation of returns
    return_std = np.std(returns)

    # Calculate Sharpe Ratio
    sharpe_ratio = (average_return - risk_free_rate) / return_std

    return sharpe_ratio


def evaluate_spread_model(df, n_components, covariance_type, random_state, backtest_period, bet_size):
    """
    Evaluates a spread model using the given parameters.

    Args:
        df (DataFrame): The input dataframe containing the data for training and testing the model.
        n_components (int): The number of hidden states in the HMM.
        covariance_type (str): The type of covariance to use ('full', 'diag', etc.).
        random_state (int): The random state for reproducibility of the model.
        backtest_period (int): The number of periods to use for backtesting the model.
        bet_size (float): The size of the bet placed during trading simulation.

    Returns:
        tuple: A tuple containing the mean squared error (mse), R-squared (r2) score,
               the actual values (y_test), predicted values (predictions), and dates.
    """

    # Initialize the spread model with specified parameters
    try:
        model = SpreadModel(n_components=n_components,
                            covariance_type=covariance_type, backtest_period=backtest_period)
        X_test, y_test, dates = model.train(df)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(
            f"Parameters: n_components={n_components}, covariance_type={covariance_type}, backtest_period={backtest_period}")
        print(f"Data shape: {df.shape}")
        raise

    # Generate predictions using the trained model
    predictions = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Initialize variables for trading simulation
    trades = []
    weekly_balance = []
    current_position = 0  # No position initially

    buy_count = 0
    sell_count = 0
    profitable_buy_count = 0
    profitable_sell_count = 0
    same_direction_count = 0
    within_1_sd_count = 0
    within_2_sd_count = 0
    same_direction_within_1_sd_count = 0

    # Calculate standard deviation of prediction errors
    error_std = np.std(y_test - predictions)

    # Simulate trading for each period in the backtest
    for i in range(backtest_period - 1):
        date = dates[i]
        historical_spread = y_test[i]
        predicted_spread = predictions[i]
        next_historical_spread = y_test[i + 1]
        action = None
        profit_loss = 0

        # Decide whether to buy or sell based on predicted spread
        if predicted_spread < historical_spread:
            action = 'Sell'
            current_position -= bet_size
            sell_count += 1
        elif predicted_spread > historical_spread:
            action = 'Buy'
            current_position += bet_size
            buy_count += 1

        # Calculate profit or loss based on the change in spread
        profit_loss = current_position * \
            (next_historical_spread - historical_spread)
        weekly_balance.append(profit_loss)

        # Record trade details
        trades.append({
            'Date': date,
            'Action': action,
            'Current Price': historical_spread,
            'Predicted Price': predicted_spread,
            'Next Week Price': next_historical_spread,
            'Profit/Loss': profit_loss
        })

        # Track profitable actions
        if action == 'Buy' and profit_loss > 0:
            profitable_buy_count += 1
        elif action == 'Sell' and profit_loss > 0:
            profitable_sell_count += 1

        # Track other statistics
        if (predicted_spread < historical_spread and next_historical_spread < historical_spread) or \
           (predicted_spread > historical_spread and next_historical_spread > historical_spread):
            same_direction_count += 1

        if abs(next_historical_spread - predicted_spread) <= error_std:
            within_1_sd_count += 1

        if abs(next_historical_spread - predicted_spread) <= 2 * error_std:
            within_2_sd_count += 1

        if ((predicted_spread < historical_spread and next_historical_spread < historical_spread) or
                (predicted_spread > historical_spread and next_historical_spread > historical_spread)) and \
                abs(next_historical_spread - predicted_spread) <= error_std:
            same_direction_within_1_sd_count += 1

    # Calculate cumulative profit/loss
    cumulative_balance = [sum(weekly_balance[:i+1])
                          for i in range(len(weekly_balance))]

    # Calculate cumulative R and R^2
    cumulative_r = np.cumsum(y_test - predictions)
    cumulative_r2 = np.cumsum((y_test - predictions) ** 2)

    # Calculate instantaneous R and R^2
    instantaneous_r = y_test - predictions
    instantaneous_r2 = (y_test - predictions) ** 2

    # Calculate additional statistics
    total_trades = buy_count + sell_count
    final_portfolio_balance = cumulative_balance[-1] if cumulative_balance else 0
    percentage_profitable_buy_actions = (
        profitable_buy_count / buy_count) * 100 if buy_count > 0 else 0
    percentage_profitable_sell_actions = (
        profitable_sell_count / sell_count) * 100 if sell_count > 0 else 0
    ratio_buy_to_sell = buy_count / \
        sell_count if sell_count > 0 else float('inf')
    percentage_same_direction = (
        same_direction_count / total_trades) * 100 if total_trades > 0 else 0
    percentage_within_1_sd = (
        within_1_sd_count / total_trades) * 100 if total_trades > 0 else 0
    percentage_within_2_sd = (
        within_2_sd_count / total_trades) * 100 if total_trades > 0 else 0
    percentage_same_direction_within_1_sd = (
        same_direction_within_1_sd_count / total_trades) * 100 if total_trades > 0 else 0

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(weekly_balance)
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Create the table separately
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=["Statistic", "Value"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[
            ["Number of Buy Actions", "Number of Sell Actions", "Profitable Buy Actions",
             "Profitable Sell Actions", "Percentage of Profitable Buy Actions",
             "Percentage of Profitable Sell Actions", "Ratio of Buy to Sell Actions",
             "Final Portfolio Balance", "Percentage Same Direction",
             "Percentage Within 1 SD", "Percentage Within 2 SD",
             "Percentage Same Direction within 1 SD"],
            [buy_count, sell_count, profitable_buy_count, profitable_sell_count,
             f"{percentage_profitable_buy_actions:.2f}%", f"{percentage_profitable_sell_actions:.2f}%",
             f"{ratio_buy_to_sell:.2f}", f"${final_portfolio_balance:.2f}",
             f"{percentage_same_direction:.2f}%", f"{percentage_within_1_sd:.2f}%",
             f"{percentage_within_2_sd:.2f}%", f"{percentage_same_direction_within_1_sd:.2f}%"]
        ])
    )])

    table_fig.show()  # Display the table

    # Create subplots for charts: 4 rows, 1 column (no table here)
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Actual vs Predicted Spread', 'Prediction Error vs MSE',
                        'Cumulative and Instantaneous R and R² Over Time', 'Week-over-Week and Cumulative Performance'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]]
    )

    # Actual vs Predicted Spread Plot
    fig.add_trace(go.Scatter(
        x=dates, y=y_test, mode='lines', name='Actual Spread'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=predictions, mode='lines', name='Predicted Spread'), row=1, col=1)
    fig.update_yaxes(title_text="Spread ($)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    # Prediction Error vs MSE Plot
    fig.add_trace(go.Scatter(
        x=dates, y=y_test - predictions, mode='lines', name='Prediction Error'), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=(y_test - predictions) ** 2, mode='lines', name='MSE', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.update_yaxes(title_text="Error / MSE", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Cumulative and Instantaneous R and R² Plot with Secondary Y-Axis
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative_r, mode='lines', name='Cumulative R'), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative_r2, mode='lines', name='Cumulative R²', line=dict(color='blue', dash='dot')), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=instantaneous_r, mode='lines', name='Instantaneous R', line=dict(color='green', dash='dash')), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=dates, y=instantaneous_r2, mode='lines', name='Instantaneous R²', line=dict(color='purple', dash='dot')), row=3, col=1, secondary_y=True)

    # Adjust the y-axis range to ensure visibility
    fig.update_yaxes(title_text="Cumulative R / R²",
                     row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Instantaneous R / R²",
                     row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Week-over-Week and Cumulative Performance Plot
    fig.add_trace(go.Scatter(
        x=dates[:-1], y=weekly_balance, mode='lines+markers', name='Week-over-Week Performance'), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=dates[:-1], y=cumulative_balance, mode='lines+markers', name='Cumulative Performance'), row=4, col=1)
    fig.update_yaxes(title_text="Profit/Loss", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    # Update layout to include a single legend for the first subplot
    fig.update_layout(
        height=1600,
        width=1000,
        title_text="Model Performance Metrics",
        showlegend=True  # Ensure that the legend is shown
    )

    # Show the figure with charts
    fig.show()

    print('SPREAD MODEL PERFORMANCE')
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return mse, r2, y_test, predictions, dates
