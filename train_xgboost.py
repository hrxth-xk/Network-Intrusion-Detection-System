# train_xgboost.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# 1️⃣ Load cleaned dataset
print("📂 Loading cleaned dataset...")
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# 2️⃣ Preprocessing: handle infinities, NaNs, numeric only
print("🧹 Cleaning data...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df = df.select_dtypes(include=[np.number])

# 3️⃣ Split features & target
X = df.drop(columns=["Label"])
y = df["Label"]

# 4️⃣ Train-Test Split
print("✂ Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Train XGBoost Classifier
print("🚀 Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 6️⃣ Predictions
print("📊 Evaluating model...")
y_pred = xgb_model.predict(X_test)

# 7️⃣ Metrics
print("\n✅ Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8️⃣ Save model
import os
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("💾 Model saved to models/xgb_model.pkl")
