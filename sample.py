import pandas as pd

# Load your cleaned dataset
data = pd.read_csv('data/processed/cleaned_dataset.csv')

# Take a small random sample (e.g., 1000 rows)
sample_data = data.sample(n=1000, random_state=42)

# Save it as test_sample.csv
sample_data.to_csv('data/raw/test_sample.csv', index=False)

print("Sample CSV created successfully!")
