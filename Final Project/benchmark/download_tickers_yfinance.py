import yfinance as yf
import pandas as pd
import os

# List of tickers for the top 15 oil companies
oil_companies_tickers = [
    'XOM',  # Exxon Mobil
    'CVX',  # Chevron
    'BP',   # BP
    'COP',  # ConocoPhillips
    'EOG',  # EOG Resources
    'OXY',  # Occidental Petroleum
    'PSX',  # Phillips 66
    'SLB',  # Schlumberger
    'HAL',  # Halliburton
    'VLO',  # Valero Energy
    'PBR',  # Petrobras
    '^GSPC'
]

# Directory to save the downloaded data
output_directory = 'oil_companies_data'
os.makedirs(output_directory, exist_ok=True)

# Function to download and save data


def download_stock_data(ticker):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, period="max", interval="1d")
    file_path = os.path.join(output_directory, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Data for {ticker} saved to {file_path}.")


# Downloading data for each ticker
for ticker in oil_companies_tickers:
    download_stock_data(ticker)

print("All data downloaded successfully.")
