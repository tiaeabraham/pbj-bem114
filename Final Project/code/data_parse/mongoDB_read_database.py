import pymongo
import re
import pandas as pd
from tqdm import tqdm

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.AtterburyCapital
collection = db["EIA NG Bulk Download"]

# Initialize list to store daily series IDs
daily_series_ids = []

# Regular expression to match 8-digit date format
date_regex = re.compile(r'^\d{8}$')

# Aggregate query to filter documents with valid date formats
pipeline = [
    {
        "$match": {
            # Match only documents with 8-digit Date values
            "Date": {"$regex": r'^\d{8}$'}
        }
    },
    {
        "$group": {
            "_id": "$Series ID"  # Group by Series ID
        }
    }
]

# Execute the aggregation pipeline with progress bar
matching_series_ids = collection.aggregate(pipeline)

# Add matching series IDs to the list
for result in tqdm(matching_series_ids, desc="Processing Series IDs"):
    daily_series_ids.append(result["_id"])

# Now retrieve all documents corresponding to the daily series IDs
documents = collection.find({"Series ID": {"$in": daily_series_ids}})

# Convert the documents to a list of dictionaries (records)
data = list(documents)

# Create a Pandas DataFrame from the records
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
# You can change this path as needed
output_path = "/Users/bramschork/Downloads/daily_series_data.xlsx"
df.to_excel(output_path, index=False)

print(f"Data exported successfully to {output_path}")
