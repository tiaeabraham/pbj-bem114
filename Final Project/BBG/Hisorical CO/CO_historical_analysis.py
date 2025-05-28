import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'CO_historical_20_24.xlsx'  # Update with your file path if different
excel_data = pd.ExcelFile(file_path)

# Initialize a DataFrame to store data from each sheet
consolidated_data = pd.DataFrame()

# Loop through each sheet, parsing data starting from row 8 with specified columns
for sheet_name in excel_data.sheet_names:
    try:
        # Parse the sheet with data starting at row 8 and row 7 as headers
        sheet_df = excel_data.parse(sheet_name, skiprows=6, usecols="A:C", names=["Date", "PX_LAST", "PX_VOLUME"])
        sheet_df.replace('#N/A N/A', pd.NA, inplace=True)  # Mark '#N/A N/A' as NaN
        sheet_df['Date'] = pd.to_datetime(sheet_df['Date'])  # Ensure date format
        sheet_df['PX_LAST'] = pd.to_numeric(sheet_df['PX_LAST'], errors='coerce')  # Convert to numeric
        sheet_df['PX_VOLUME'] = pd.to_numeric(sheet_df['PX_VOLUME'], errors='coerce')  # Convert to numeric
        sheet_df['Contract'] = sheet_name  # Add a column for the contract name
        consolidated_data = pd.concat([consolidated_data, sheet_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing sheet {sheet_name}: {e}")

# Sort data by date for better plotting
consolidated_data = consolidated_data.sort_values(by=['Contract', 'Date'])

# Separate March and April contracts
march_contracts = consolidated_data[consolidated_data['Contract'].str.startswith('H')]
april_contracts = consolidated_data[consolidated_data['Contract'].str.startswith('J')]

# Group by date and calculate the average PX_LAST for March and April contracts
march_avg = march_contracts.groupby('Date')['PX_LAST'].mean().reset_index()
april_avg = april_contracts.groupby('Date')['PX_LAST'].mean().reset_index()

# Merge the March and April averages on Date to calculate the difference
merged_avg = pd.merge(march_avg, april_avg, on='Date', suffixes=('_march', '_april'))
merged_avg['Difference'] = merged_avg['PX_LAST_march'] - merged_avg['PX_LAST_april']

# Plot the average price of March and April contracts over time
plt.figure(figsize=(12, 6))
plt.plot(march_avg['Date'], march_avg['PX_LAST'], label='Average March Contracts', color='blue')
plt.plot(april_avg['Date'], april_avg['PX_LAST'], label='Average April Contracts', color='orange')

plt.title('Average Price of March vs. April Contracts Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the difference between March and April contract averages over time
plt.figure(figsize=(12, 6))
plt.plot(merged_avg['Date'], merged_avg['Difference'], color='purple', label='March - April Price Difference')

plt.title('Difference Between Average March and April Contracts Over Time')
plt.xlabel('Date')
plt.ylabel('Price Difference (March - April)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
