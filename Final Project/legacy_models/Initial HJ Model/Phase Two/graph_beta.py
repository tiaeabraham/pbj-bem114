import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the merged data and pairs list
merged_df = pd.read_csv('merged_data_day_to_day_change.csv')
pairs_df = pd.read_csv('pairs.csv')

# Convert 'Date' column to datetime for proper comparison
merged_df['Date'] = pd.to_datetime(merged_df['Date'], utc=True)

# Read original stock prices data
stock_prices = {}
directory = 'yfinance'
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        stock_name = filename.replace('.csv', '')
        stock_prices[stock_name] = pd.read_csv(
            file_path, usecols=['Date', 'Close'])
        stock_prices[stock_name]['Date'] = pd.to_datetime(
            stock_prices[stock_name]['Date'], utc=True)

# Identify dates with specified price changes
thresholds = {
    'High Change': 0.06,  # 6%
    'Low Change': 0.02    # 2%
}

dates_with_changes = []

for _, row in pairs_df.iterrows():
    stock1 = row['Stock 1']
    stock2 = row['Stock 2']

    stock1_prices = stock_prices[stock1]
    stock2_prices = stock_prices[stock2]

    for i in range(len(stock1_prices) - 1):
        for j in range(i + 1, len(stock1_prices)):
            stock1_start_price = stock1_prices.iloc[i]['Close']
            stock1_end_price = stock1_prices.iloc[j]['Close']
            stock2_start_price = stock2_prices.iloc[i]['Close']
            stock2_end_price = stock2_prices.iloc[j]['Close']

            stock1_gain = (stock1_end_price -
                           stock1_start_price) / stock1_start_price
            stock2_gain = (stock2_end_price -
                           stock2_start_price) / stock2_start_price

            if stock1_gain > thresholds['High Change'] and stock2_gain <= thresholds['Low Change']:
                dates_with_changes.append(
                    (stock1_prices.iloc[i]['Date'], stock1_prices.iloc[j]['Date'], stock1, stock2))
            elif stock2_gain > thresholds['High Change'] and stock1_gain <= thresholds['Low Change']:
                dates_with_changes.append(
                    (stock2_prices.iloc[i]['Date'], stock2_prices.iloc[j]['Date'], stock2, stock1))

# Print the list of date ranges with changes
print("Date ranges with specified price changes:")
for idx, (start_date, end_date, stock1, stock2) in enumerate(dates_with_changes, start=1):
    print(f"{idx}. {start_date.date()} to {end_date.date()} - {stock1} and {stock2}")

# Prompt the user to select a date range to plot
selection = int(
    input("Enter the number of the date range you want to plot: ")) - 1

if 0 <= selection < len(dates_with_changes):
    start_date, end_date, stock1, stock2 = dates_with_changes[selection]

    # Get stock prices for the selected date range
    stock1_prices = stock_prices[stock1]
    stock2_prices = stock_prices[stock2]

    stock1_plot_data = stock1_prices[(stock1_prices['Date'] >= start_date) & (
        stock1_prices['Date'] <= end_date)]
    stock2_plot_data = stock2_prices[(stock2_prices['Date'] >= start_date) & (
        stock2_prices['Date'] <= end_date)]

    # Calculate day-over-day changes as percentage
    stock1_plot_data['Day-over-Day Change'] = stock1_plot_data['Close'].pct_change()
    stock2_plot_data['Day-over-Day Change'] = stock2_plot_data['Close'].pct_change()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price')
    ax1.plot(stock1_plot_data['Date'],
             stock1_plot_data['Close'], label=stock1, color='tab:blue')
    ax1.plot(stock2_plot_data['Date'], stock2_plot_data['Close'],
             label=stock2, color='tab:orange')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Day-over-Day Change (%)')
    ax2.plot(stock1_plot_data['Date'], stock1_plot_data['Day-over-Day Change'],
             label=f'{stock1} Day-over-Day Change', linestyle='dotted', color='tab:blue', alpha=0.7)
    ax2.plot(stock2_plot_data['Date'], stock2_plot_data['Day-over-Day Change'],
             label=f'{stock2} Day-over-Day Change', linestyle='dotted', color='tab:orange', alpha=0.7)
    ax2.legend(loc='upper right')

    plt.title(
        f"Stock Prices and Day-over-Day Changes of {stock1} and {stock2} from {start_date.date()} to {end_date.date()}")
    fig.tight_layout()
    plt.show()
else:
    print("Invalid selection.")
