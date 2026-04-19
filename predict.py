# predict.py
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Load the model
print("Loading trained model...")
model = joblib.load("models/rf_model.pkl")

# Check if input file is given
if len(sys.argv) < 2:
    print("Please provide a CSV file to predict. Example:")
    print("python predict.py data/raw/new_traffic.csv")
    sys.exit()

input_file = sys.argv[1]

if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    sys.exit()

# Load new data
print(f"Reading data from {input_file}...")
df = pd.read_csv(input_file)

# Preprocessing: same cleaning as training
print("Cleaning data...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Ensure only numeric columns are used
df = df.select_dtypes(include=[np.number])

# Predict
print("Predicting...")
predictions = model.predict(df)

# Map predictions to labels (0 = Normal, 1 = Attack)
pred_labels = ["Normal" if p == 0 else "Attack" for p in predictions]

# Save results
output_file = "predictions.csv"
df_results = pd.DataFrame({"Prediction": pred_labels})
df_results.to_csv(output_file, index=False)

print(f" Predictions saved to {output_file}")
