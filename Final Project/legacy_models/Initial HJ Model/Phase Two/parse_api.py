import csv
import os

# Sample API data
data = [
    {'Global Quote': {'01. symbol': 'ACDC', '02. open': '8.5600', '03. high': '8.6700', '04. low': '8.0600', '05. price': '8.1600', '06. volume': '831486',
                      '07. latest trading day': '2024-07-18', '08. previous close': '8.5700', '09. change': '-0.4100', '10. change percent': '-4.7841%'}},
    {'Global Quote': {'01. symbol': 'AE', '02. open': '27.5100', '03. high': '28.4350', '04. low': '27.3492', '05. price': '27.6200', '06. volume': '14528',
                      '07. latest trading day': '2024-07-18', '08. previous close': '28.3700', '09. change': '-0.7500', '10. change percent': '-2.6436%'}}
]

# Define the CSV file name
csv_file = 'stock_data.csv'

# Define the headers from the first data entry
headers = list(data[0]['Global Quote'].keys())

# Check if the CSV file already exists
file_exists = os.path.isfile(csv_file)

# Write to the CSV file
with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers)

    # Write the headers only if the file doesn't exist
    if not file_exists:
        writer.writeheader()

    # Write each row of data
    for entry in data:
        writer.writerow(entry['Global Quote'])

print(f"Data has been appended to {csv_file}")
