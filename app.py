# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")
st.title("🛡️ Network Intrusion Detection System")

st.markdown("""
Welcome! This app predicts whether network traffic is **Normal** or **Attack** using your trained models.
- **Step 1:** Select a model (Random Forest or XGBoost)
- **Step 2:** Upload a CSV file with network traffic data
- **Step 3:** See predictions and download results
""")

# 1️⃣ Model selection
model_option = st.selectbox(
    "Select Model",
    ("Random Forest", "XGBoost")
)

# Load model with error handling
model_path = "models/rf_model.pkl" if model_option == "Random Forest" else "models/xgb_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success(f"✅ Loaded {model_option} model successfully!")
else:
    st.warning(f"⚠ Model not found: {model_path}")
    st.stop()  # Stop execution until model exists

st.markdown("---")

# 2️⃣ File uploader
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is None:
    st.info("📌 Please upload a CSV file to get started.")
else:
    # 3️⃣ Load CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    # 4️⃣ Preprocess
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    df_clean = df_clean.select_dtypes(include=[np.number])

    if df_clean.empty:
        st.error("❌ No numeric columns found for prediction. Check your CSV.")
        st.stop()

    # 5️⃣ Predict
    predictions = model.predict(df_clean)
    pred_labels = ["Normal" if p == 0 else "Attack" for p in predictions]
    df_result = df.copy()
    df_result["Prediction"] = pred_labels

    # 6️⃣ Summary
    total = len(pred_labels)
    normal_count = pred_labels.count("Normal")
    attack_count = pred_labels.count("Attack")

    st.markdown("### 📊 Prediction Summary")
    st.write(f"Total rows: {total}")
    st.write(f"Normal traffic: {normal_count}")
    st.write(f"Attack traffic: {attack_count}")
    st.write(f"Attack percentage: {attack_count/total*100:.2f}%")

    # 7️⃣ Show predictions
    st.markdown("### 📝 Prediction Results")
    st.dataframe(df_result)

    # 8️⃣ Download button
    csv = df_result.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
