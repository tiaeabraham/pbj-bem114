import pandas as pd

raw_data_path = 'data/raw/NOAA_weather.csv'
export_path = 'data/clean/clean_NOAA_weather.csv'

# Read the raw data
df = pd.read_csv(raw_data_path)

# List of columns to keep
columns_to_keep = ['NAME', 'PRCP', 'ELEVATION', 'DATE']

# Select only the columns you want to keep
df = df[columns_to_keep]
print(df)

'''# Drop the last two rows for incomplete download data
df = df.iloc[:-2]

# Detect consecutive NaNs for all columns
consec_nan_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

for col in df.columns:
    consec_nan_mask[col] = df[col].isna() & df[col].shift(1).isna()

# Drop rows where any column has consecutive NaNs
rows_to_drop = consec_nan_mask.any(axis=1)
print(len(rows_to_drop))

df_cleaned = df[~rows_to_drop].reset_index(drop=True)

# Use infer_objects to handle object types
df_cleaned = df_cleaned.infer_objects()

# Convert any remaining object columns to numeric (if applicable)
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Perform linear interpolation
df_interpolated = df_cleaned.interpolate(method='linear')

# Save the cleaned and interpolated DataFrame
#  df_interpolated.to_csv(export_path, index=False)

# print(df_interpolated.isnull().values.any())
# print(df_interpolated)
'''
