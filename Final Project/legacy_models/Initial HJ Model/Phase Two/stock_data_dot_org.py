'''
TODO: Program does not create folder for sector

Must account for usage limit error:
<Response [402]>
{'error': {'code': 'usage_limit_reached', 'message': 'The usage limit for this account has been reached.'}}
Traceback (most recent call last):
  File "/Users/bramschork/Documents/Correlate/stock_data_dot_org.py", line 57, in <module>
    data_list = data['data']
KeyError: 'data'
'''

from keys import api_key
import requests
import json
import csv
from datetime import datetime
import os

current_sector = 'Health Care'

csv_file = f"CSV Files/Sector Stock Data/{current_sector}/{current_sector}_{datetime.now().strftime('%m_%d_%y')}.csv"
if not os.path.isfile(csv_file):
    # Create the file if it does not exist
    with open(csv_file, mode='w') as file:
        # Optionally write some initial content to the file
        file.write('')  # Write an empty string to create the file

# Read the CSV file into a list excluding the header


def read_csv_to_list(filename):
    data_list = []
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            data_list.append(row[0])  # one item list
    return data_list


# Read the CSV file
all_symbols = read_csv_to_list(
    f'CSV Files/NASDAQ Symbols by Sector/nasdaq_{current_sector}_symbols.csv')

# break into chunks of 3
# Function to break a list into sublists of n items


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# Split the list into sublists of 3 items each
sublists = split_list(all_symbols, 3)


for chunk in sublists:
    print(chunk)
    symbols_string = ",".join(chunk)
    json_data = requests.get(
        f'https://api.stockdata.org/v1/data/quote?symbols={symbols_string}&api_token={api_key}')

    print(json_data)

    # Parse JSON data
    data = json_data.json()
    print(data)
    # Extract the data list
    data_list = data['data']

    # Define the CSV file headers
    headers = data_list[0].keys()

    # Write data to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for item in data_list:
            writer.writerow(item)

    print(f"Data has been written to {csv_file}")
