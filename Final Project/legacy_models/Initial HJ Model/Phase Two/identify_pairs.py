'''
ToDo:
Check correlation matrix, make sure it is right!
Check total_corr and 5daycorr

Identify pairs of stocks with correlated stock prices.

If needed implement this: merged_df['Date'] = pd.to_datetime(merged_df['Date'], utc=True)

Extension to see dataframe during debug

'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

from config import *


def get_stock_prices(stock_filename):
    '''
    Returns DF of stock prices. Common Use:

    for stock_filename in os.listdir(yfinance_stock_price_directory):
        df = get_stock_prices(stock_filename)
    '''
    file_path = os.path.join(
        yfinance_stock_price_directory, stock_filename)
# try:
    df = pd.read_csv(file_path, usecols=['Date', 'Open', 'Close'])
    return df


def calculate_day_to_day_change():
    """
    Creates two new DFs, CSV from the stock price raw files that contains the day to day change of each stock relative to itself.

    Returns: tailored_day_to_day_change_DF
    """

    # DF to contain the day to day changes of all stocks
    day_to_day_change_DF = pd.DataFrame()

    # DF to contain day to day changes of stocks that meet certain criteria as detailed below
    tailored_day_to_day_change_DF = pd.DataFrame()

    # Get list of all stocks in the yfinance directory
    yfinance_filenames = os.listdir(yfinance_stock_price_directory)

    for stock_filename in yfinance_filenames:
        if stock_filename.endswith('.csv'):
            df = get_stock_prices(stock_filename)
        else:
            # Else: given file is not a CSV. Log and continue to exit this iteration of the for loop.
            logging.info(
                f'{stock_filename} is not a CSV file. Skipping day_to_day_change calculation.')
            continue

        print(stock_filename)
        # Spread of the price of the stock. Highest closing price minus lowest
        spread = df['Close'].max() - df['Close'].min()

        # What percentage of the stocks value is the spread. iloc removes index. Ex of spread_percentage: 0.6085074568247432
        try:
            spread_percentage = spread / df['Close'].iloc[-1]
        except IndexError as e:
            logging.warning(
                f'Error occured parsing {stock_filename}, file is empty.\nError: "{e}"\n')

            # Create new column in the DF with the DtD percentage change
        df['Day-to-Day Change'] = (df['Close'] -
                                   df['Open']) / df['Open']

        # Drop all DF columns except Date and DtD change
        df = df[['Date', 'Day-to-Day Change']]

        # Rename 'Day-to-Day change column. New name is derived from the stock_filename variable and removing the
        # '.csv' extension. Inplace=true ensures that the renaming is done directly on the original DataFrame df
        # without creating a new DataFrame
        df.rename(
            columns={'Day-to-Day Change': stock_filename.replace('.csv', '')}, inplace=True)

        # Add all the changes to the DtD change DF
        '''
        day_to_day_change_DF was initialized when this function was called. The first time the first time through the for loop it will be empty.
        '''
        if day_to_day_change_DF.empty:
            day_to_day_change_DF = df
        else:
            day_to_day_change_DF = pd.merge(
                day_to_day_change_DF, df, on='Date', how='outer')

        # Add only stocks that pass the thresholds to the tailored DtD DF

        if spread >= price_difference_threshold and spread_percentage >= price_change_percentage_threshold:
            if tailored_day_to_day_change_DF.empty:
                tailored_day_to_day_change_DF = df
            else:
                tailored_day_to_day_change_DF = pd.merge(
                    tailored_day_to_day_change_DF, df, on='Date', how='outer')

    return tailored_day_to_day_change_DF


def calculate_correlation_pairs():
    """
    Format of pairs in list [[stock_price_1, stock_price_2, total_corr, 5-day_corr]]
    Calculate the correlation between day-to-day changes of stocks and find pairs with correlation above the specified thresholds.

    Parameters:
    merged_df (pd.DataFrame): The merged DataFrame containing the day-to-day changes of stocks.
    total_threshold (float): The threshold for the total correlation.
    five_day_threshold (float): The threshold for the correlation of the final five values.

    Returns:
    list: A list of pairs of stocks with their total and final five values correlations.
    """

    # We have here a date frame that by date, lusts the DtD percent change of a stock
    '''
    What I want to do: go through each column, and see if it is correlated with any other colun
    '''

    # Essentially tailored_day_to_day_change_DF (what is returned from the calculate_day_to_day_chane function), without the 'Date' column
    tailored_day_to_day_change_DF = calculate_day_to_day_change().drop(columns=[
        'Date'])

    pairs = []
    columns = tailored_day_to_day_change_DF.columns

    # For every column, ie every stock
    for i in range(len(columns)):
        # For every column + 1 in range
        for j in range(i + 1, len(columns)):
            # total_corr and final_five_corr are np.float numbers here, ex. "np.float64(0.3145)"
            col1 = columns[i]
            col2 = columns[j]
            total_corr = tailored_day_to_day_change_DF[[col1, col2]
                                                       ].dropna().corr().iloc[0, 1]
            # print(f'{col1} to {col2}: {total_corr}')
            final_five_corr = tailored_day_to_day_change_DF[[col1, col2]
                                                            ].dropna().tail(5).corr().iloc[0, 1]
            if total_corr > total_correlation_threshold and final_five_corr > five_day_correlation_threshold:
                pair = [col1, col2, total_corr, final_five_corr]
                pairs.append(pair)

    # Print correlation matrix
    # print(tailored_day_to_day_change_DF.corr())

    # Sort pairs by highest total correlation and highest five-day correlation
    sorted_pairs_total_corr = sorted(pairs, key=lambda x: x[2], reverse=True)
    sorted_pairs_five_day_corr = sorted(
        pairs, key=lambda x: x[3], reverse=True)

    '''# Print top 5 pairs from each sorted list

    print(
        f"\nTop {len(sorted_pairs_five_day_corr[:5])} pairs by total correlation:")
    for pair in sorted_pairs_total_corr[:5]:
        print(pair)

    print(
        f"\nTop {len(sorted_pairs_total_corr[:5])} pairs by five-day correlation:")
    for pair in sorted_pairs_five_day_corr[:5]:
        print(pair)

    # Plot stock prices and rolling correlation for the top three pairs with the highest total correlation
    for i, pair in enumerate(sorted_pairs_total_corr[:3]):
        plot_pair(pair)

    # Show figures
    plt.show()'''

    # Return format: ([], [])
    return sorted_pairs_total_corr, sorted_pairs_five_day_corr


def plot_pair(pair):
    """
    Plot the stock prices of a pair of stocks over the past year along with their correlation over time.

    Parameters:
    directory (str): The directory containing the CSV files.
    pair (list): The pair of stocks with their correlations.
    """
    col1, col2 = pair[0], pair[1]
    file1 = os.path.join(
        yfinance_stock_price_directory, col1 + '.csv')
    file2 = os.path.join(
        yfinance_stock_price_directory, col2 + '.csv')

    df1 = pd.read_csv(file1, usecols=['Date', 'Close'])
    df2 = pd.read_csv(file2, usecols=['Date', 'Close'])

    df1.rename(columns={'Close': col1}, inplace=True)
    df2.rename(columns={'Close': col2}, inplace=True)

    merged_prices = pd.merge(df1, df2, on='Date', how='outer').dropna()

    merged_prices['Date'] = pd.to_datetime(merged_prices['Date'], utc=True)
    merged_prices = merged_prices.set_index('Date')

    one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
    merged_prices = merged_prices.loc[merged_prices.index >= one_year_ago]

    rolling_5_corr = merged_prices[col1].rolling(
        window=5).corr(merged_prices[col2])
    rolling_total_corr = merged_prices[col1].expanding(
        min_periods=5).corr(merged_prices[col2])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price')
    ax1.plot(merged_prices.index,
             merged_prices[col1], label=col1, color='tab:blue')
    ax1.plot(merged_prices.index,
             merged_prices[col2], label=col2, color='tab:orange')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Correlation')
    ax2.plot(merged_prices.index, rolling_5_corr,
             label='Rolling 5-day Correlation', color='tab:green', alpha=0.7)
    ax2.plot(merged_prices.index, rolling_total_corr,
             label='Rolling Total Correlation', color='tab:red')
    ax2.legend(loc='upper right')

    plt.title(
        f'Stock Prices and Rolling Correlations of {col1} and {col2} Over the Past Year')
    fig.tight_layout()
