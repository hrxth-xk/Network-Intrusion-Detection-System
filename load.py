import pandas as pd
import glob
import os

# Path to your raw data folder
raw_data_path = "data/raw"

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))

# List to store DataFrames
df_list = []

print(f"Found {len(csv_files)} CSV files.")

# Loop through and read each CSV file
for file in csv_files:
    print(f"Loading {file} ...")
    df = pd.read_csv(file)
    df_list.append(df)

# Combine all into one DataFrame
data = pd.concat(df_list, ignore_index=True)

print("\nData Loaded Successfully!")
print("Shape of combined dataset:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Optional: check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())
