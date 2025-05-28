import pandas as pd
import json
import os
from pymongo import MongoClient
from tqdm import tqdm  # Import the tqdm library for the progress bar

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client['AtterburyCapital']  # Use your database name
collection = db['EIA NG Bulk Download']  # Use your collection name

# Define the file path
file_path = '/Users/bramschork/Downloads/NG.txt'

# Determine the total number of lines in the file for the progress bar
total_lines = sum(1 for _ in open(file_path, 'r'))

# Open the text file
with open(file_path, 'r') as file:
    line_number = 0
    # Iterate over each line in the file with a progress bar
    for line in tqdm(file, total=total_lines, desc="Processing Lines"):
        try:
            # Parse the JSON data from the current line
            parsed_data = json.loads(line)

            # Extract metadata and data points
            series_id = parsed_data.get("series_id", "")
            name = parsed_data.get("name", "")
            units = parsed_data.get("units", "")
            description = parsed_data.get("description", "")
            source = parsed_data.get("source", "")
            data_points = parsed_data.get("data", [])

            # Truncate the data to only 10 points
            truncated_data = data_points[:10]

            # Prepare data for MongoDB insertion
            for date, value in truncated_data:
                document = {
                    "Series ID": series_id,
                    "Name": name,
                    "Units": units,
                    "Description": description,
                    "Source": source,
                    "Date": date,
                    "Value": value
                }
                # Insert the document into the collection
                collection.insert_one(document)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number+1}: {e}")

        line_number += 1

print("Data processing and insertion completed successfully!")
