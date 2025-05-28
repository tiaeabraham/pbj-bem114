'''
Look for stocks pairs where one stock has experienced a large change and the second stock has not yet followed

confirm tuples
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import *


def main(sorted_pairs_total_corr, sorted_pairs_five_day_corr):

    # Variable to hold dates with pairs that meet the criteria List of lists [[Stock1, Stock1, Date]]
    valid_dates = [()]

    print(sorted_pairs_total_corr)

    for _, row in pairs_df.iterrows():
        stock1 = row['Stock 1']
        stock2 = row['Stock 2']
        for _, price_change1 in merged_df[['Date', stock1]].dropna().iterrows():
            date = price_change1['Date']
            stock1_day_change = merged_df.loc[merged_df['Date']
                                              == date, stock1].values[0]
            stock2_day_change = merged_df.loc[merged_df['Date']
                                              == date, stock2].values
            if len(stock2_day_change) > 0:
                stock2_day_change = stock2_day_change[0]
                if (abs(stock1_day_change) >= thresholds['High Change'] and abs(stock2_day_change) <= thresholds['Low Change']) or \
                        (abs(stock2_day_change) >= thresholds['High Change'] and abs(stock1_day_change) <= thresholds['Low Change']):
                    dates_with_changes.append((date, stock1, stock2))

    # Print the list of dates with changes
    print("Dates with specified price changes:")
    for idx, (date, stock1, stock2) in enumerate(dates_with_changes, start=1):
        print(f"{idx}. {date.date()} - {stock1} and {stock2}")

    # Prompt the user to select a date to plot
    selection = int(
        input("Enter the number of the date you want to plot: ")) - 1

    if 0 <= selection < len(dates_with_changes):
        selected_date, stock1, stock2 = dates_with_changes[selection]

        # Get stock prices for the selected date
        stock1_prices = stock_prices[stock1]
        stock2_prices = stock_prices[stock2]

        # Filter data to get the 5-day window
        start_date = selected_date
        end_date = selected_date + pd.DateOffset(days=4)

        stock1_plot_data = stock1_prices[(stock1_prices['Date'] >= start_date) & (
            stock1_prices['Date'] <= end_date)]
        stock2_plot_data = stock2_prices[(stock2_prices['Date'] >= start_date) & (
            stock2_prices['Date'] <= end_date)]

        # Calculate day-over-day changes as percentage
        stock1_plot_data['Day-over-Day Change'] = (
            stock1_plot_data['Close'] - stock1_plot_data['Open']) / stock1_plot_data['Open']
        stock2_plot_data['Day-over-Day Change'] = (
            stock2_plot_data['Close'] - stock2_plot_data['Open']) / stock2_plot_data['Open']

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
            f"Stock Prices and Day-over-Day Changes of {stock1} and {stock2} Starting from {selected_date.date()}")
        fig.tight_layout()
        plt.show()
    else:
        print("Invalid selection.")
