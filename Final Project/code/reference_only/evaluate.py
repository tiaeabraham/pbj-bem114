import pandas as pd
import numpy as np

#### LOAD DATA ####
portfolio_data = pd.read_csv('predicted_vs_actual.csv')
sp500_data = pd.read_csv("sp500_adjusted_close.csv")

# Empty DataFrame to hold statistics
statistics = pd.DataFrame()

#### BETA ####
# Calculate daily returns for the portfolio
portfolio_data['Return'] = portfolio_data['Predicted'] - \
    portfolio_data['Actual']

# Calculate daily returns for the S&P 500
sp500_data['Return'] = sp500_data['Adj Close'].pct_change()

# Convert 'Date' columns to datetime format
portfolio_data['Date'] = pd.to_datetime(portfolio_data['Date'])
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])


# Merge the portfolio and S&P 500 data on 'Date'
merged_data = pd.merge(portfolio_data[['Date', 'Return']], sp500_data[[
                       'Date', 'Return']], on='Date', suffixes=('_portfolio', '_sp500'))

# Drop any rows with missing values
merged_data.dropna(inplace=True)

# Calculate beta: covariance of portfolio and S&P 500 returns divided by the variance of S&P 500 returns
covariance = merged_data['Return_portfolio'].cov(merged_data['Return_sp500'])
variance = merged_data['Return_sp500'].var()
beta = covariance / variance
statistics['Beta'] = beta
print(f'Beta: {beta}')

#### ALPHA ####
# Assume you have a risk-free rate and benchmark return (for example purposes)
risk_free_rate = 0.0498 / 90  # 90 Day T-Bill converted to daily return
benchmark_return = 0.07 / 252  # Convert annual benchmark return to daily

# Calculate the average daily return of your portfolio
average_portfolio_return = portfolio_data['Return'].mean()

alpha = average_portfolio_return - \
    (risk_free_rate + beta * (benchmark_return - risk_free_rate))

# Add alpha to the statistics DataFrame
statistics['Alpha'] = [alpha]


#### WIN RATE ####
'''
The portfolio_data looks like this:
Date	  Actual	    Predicted
6/10/19	  0.172413793	0.41327834

Win rate is how often I make the right direction and right sizing.
'''
# Directional win rate
directional_rate = 0
total = 0
for i in range(len(portfolio_data['Predicted'])-1):
    next_prediction = portfolio_data['Predicted'].iloc[i+1]
    current_price = portfolio_data['Actual'].iloc[i]
    next_price = portfolio_data['Actual'].iloc[i+1]
    # If I think the spread will increase, next prediction > current price
    if next_prediction > current_price:
        if next_price > current_price:
            directional_rate += 1
            # print('Increase')
    # If I think the spread will decrease, next prediction < current price
    elif next_prediction < current_price:
        if next_price < current_price:
            directional_rate += 1
            # print('Decrease')
    total += 1
statistics['Directional Win Rate'] = directional_rate / total

'''print(directional_rate)
print(total)
print(directional_rate/total)'''

print(statistics)
