# predict_with_summary.py
import pandas as pd
import joblib
import numpy as np
import sys
import os

# 1️⃣ Load trained model
print("📂 Loading trained model...")
model = joblib.load("models/rf_model.pkl")

# 2️⃣ Check input file
if len(sys.argv) < 2:
    print("❌ Please provide a CSV file to predict. Example:")
    print("python predict_with_summary.py data/raw/test_sample.csv")
    sys.exit()

input_file = sys.argv[1]

if not os.path.exists(input_file):
    print(f"❌ File not found: {input_file}")
    sys.exit()

# 3️⃣ Load new data
print(f"📄 Reading data from {input_file}...")
df = pd.read_csv(input_file)

# 4️⃣ Preprocessing: clean infinities/NaN, only numeric columns
print("🧹 Cleaning data...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
df = df.select_dtypes(include=[np.number])

# 5️⃣ Predict
print("🤖 Predicting...")
predictions = model.predict(df)
pred_labels = ["Normal" if p == 0 else "Attack" for p in predictions]

# 6️⃣ Save predictions
output_file = "predictions.csv"
df_results = pd.DataFrame({"Prediction": pred_labels})
df_results.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}")

# 7️⃣ Print summary
total = len(pred_labels)
normal_count = pred_labels.count("Normal")
attack_count = pred_labels.count("Attack")

print("\n📊 Prediction Summary")
print(f"Total rows: {total}")
print(f"Normal traffic: {normal_count}")
print(f"Attack traffic: {attack_count}")
print(f"Attack percentage: {attack_count/total*100:.2f}%")
