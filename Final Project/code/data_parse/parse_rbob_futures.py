import pandas as pd

raw_data_path = 'data/raw/Gasoline RBOB Futures Historical Data.csv'
export_path = 'data/clean/clean_gasoline_RBOB_Futures_historical_data.csv'

# Read the raw data
df = pd.read_csv(raw_data_path)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop 'Vol.' column as it has unreliable data
df = df.drop(columns=['Vol.'])

# Convert the 'Change %' column to float (assuming this column has a '%' symbol that needs to be stripped)
df['Change %'] = df['Change %'].str.rstrip('%').astype(float)

# Convert all other columns to numeric, excluding 'Date' and 'Change %'
for col in df.columns:
    if col != 'Date' and col != 'Change %':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Detect consecutive NaNs for all columns except 'Date'
consec_nan_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

for col in df.columns:
    if col != 'Date':  # Skip 'Date' column
        consec_nan_mask[col] = df[col].isna() & df[col].shift(1).isna()

# Drop rows where any column (excluding 'Date') has consecutive NaNs
rows_to_drop = consec_nan_mask.any(axis=1)
df_cleaned = df[~rows_to_drop].reset_index(drop=True)

# Perform linear interpolation on numeric columns, excluding 'Date' and 'Change %'
df_interpolated = df_cleaned.interpolate(method='linear')

# Save the cleaned and interpolated DataFrame
df_interpolated.to_csv(export_path, index=False)

# Check for NaNs and print
# print(df_interpolated.isnull().values.any())
# print(df_interpolated)
