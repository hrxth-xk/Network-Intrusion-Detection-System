import pandas as pd
import os

# Path to your dataset folder
dataset_path = "data/raw"

# Load all CSV files
all_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
dataframes = []

for file in all_files:
    print(f"Loading {file}...")
    df = pd.read_csv(os.path.join(dataset_path, file))
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Identify label column name
    possible_labels = ["Label", "label", "class", "Class", "Category"]
    label_col = None
    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"❌ No label column found in {file}. Columns are: {df.columns.tolist()}")
    
    # Rename label column to 'Label' for consistency
    df.rename(columns={label_col: "Label"}, inplace=True)
    
    # Convert to binary (0 = BENIGN, 1 = ATTACK)
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
    
    dataframes.append(df)

# Combine all CSVs
data = pd.concat(dataframes, ignore_index=True)
print(f"Dataset shape before cleaning: {data.shape}")

# Example: drop NaN and infinity
data = data.replace([pd.NA, float('inf'), float('-inf')], 0)

print(f"Dataset shape after cleaning: {data.shape}")
print("✅ Data preprocessing complete!")
# After your cleaning code...

# Save cleaned dataset
import os

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cleaned_dataset.csv")

df.to_csv(output_path, index=False)
print(f"✅ Data preprocessing complete! Saved cleaned dataset to {output_path}")
