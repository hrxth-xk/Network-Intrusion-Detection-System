# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load preprocessed dataset
print("📂 Loading cleaned dataset...")
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# 2. Handle infinite and missing values
print("🧹 Cleaning infinite and missing values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Optional: Clip extreme values (e.g., keep within -1e6 to 1e6)
df = df.clip(lower=-1e6, upper=1e6)

# 3. Split features & target
X = df.drop(columns=["Label"])
y = df["Label"]

# 4. Train-Test Split
print("✂ Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Random Forest model
print("🚀 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 6. Predictions
print("📊 Evaluating model...")
y_pred = rf_model.predict(X_test)

# 7. Metrics
print("\n✅ Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(rf_model, "models/rf_model.pkl")
print("💾 Model saved to models/rf_model.pkl")
