# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")
st.title("Network Intrusion Detection System")

st.markdown("""
Welcome! This app predicts whether network traffic is **Normal** or **Attack** using your trained models.
- **Step 1:** Select a model (Random Forest or XGBoost)
- **Step 2:** Upload a CSV file with network traffic data (optionally with true `Label` column for evaluation)
- **Step 3:** See predictions, visualizations, and download results
""")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ("Random Forest", "XGBoost")
)

# Load model with error handling
model_path = "models/rf_model.pkl" if model_option == "Random Forest" else "models/xgb_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success(f"Loaded {model_option} model successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.warning(f"Model not found: {model_path}")
    st.stop()  # Stop execution until model exists

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
else:
    # Load CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Save a copy for later (we'll use df_result for display)
    original_df = df.copy()

    #  Preprocess: replace inf, fillna, select numeric
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    df_clean = df_clean.select_dtypes(include=[np.number])

    if df_clean.empty:
        st.error("No numeric columns found for prediction. Check your CSV.")
        st.stop()

    # Drop target/label column if present in numeric features (we'll keep original for evaluation)
    if "Label" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Label"])
        st.info("Dropped 'Label' column before prediction (kept original for evaluation).")

    # If model exposes expected feature names, align columns (fills missing with 0, drops extras)
    try:
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            # Add missing features with zeros
            missing_features = [f for f in expected_features if f not in df_clean.columns]
            for mf in missing_features:
                df_clean[mf] = 0
            # Reindex to expected order (extra cols will be dropped)
            df_clean = df_clean.reindex(columns=expected_features, fill_value=0)
            st.info("Aligned input features to model's expected feature names.")
        else:
            st.warning("Model does not expose `feature_names_in_`. Make sure your CSV columns match training features.")
    except Exception as e:
        st.warning(f"Could not auto-align features: {e}")

    # Final check
    if df_clean.shape[0] == 0:
        st.error("No rows available for prediction after preprocessing.")
        st.stop()

    # Predict with error handling
    try:
        predictions = model.predict(df_clean)
        pred_labels = ["Normal" if p == 0 else "Attack" for p in predictions]
        df_result = original_df.copy()
        df_result["Prediction"] = pred_labels

        # Summary numbers
        total = len(pred_labels)
        normal_count = pred_labels.count("Normal")
        attack_count = pred_labels.count("Attack")

        st.markdown("## Prediction Summary")
        st.write(f"Total rows: **{total}**")
        st.write(f"Normal traffic: **{normal_count}**")
        st.write(f"Attack traffic: **{attack_count}**")
        st.write(f"Attack percentage: **{attack_count/total*100:.2f}%**")

        # ---------------------------
        # VISUALIZATIONS
        # ---------------------------
        st.markdown("## Visualizations")

        # 1) Bar chart (counts)
        summary_df = pd.DataFrame({
            "Label": ["Normal", "Attack"],
            "Count": [normal_count, attack_count]
        })
        bar = alt.Chart(summary_df).mark_bar().encode(
            x=alt.X("Label:N", title=None),
            y=alt.Y("Count:Q"),
            color=alt.Color("Label:N", legend=None)
        ).properties(width=300, height=300, title="Prediction Counts")
        text = bar.mark_text(
            align='center',
            baseline='middle',
            dy=-10  # Nudges text above bars
        ).encode(text='Count:Q')
        st.altair_chart((bar + text).configure_title(anchor="start"))

        # 2) Donut / Pie chart
        pie = alt.Chart(summary_df).transform_calculate(
            percent="datum.Count / " + str(total)
        ).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Label", type="nominal"),
            tooltip=["Label", "Count"]
        ).mark_arc(innerRadius=60).properties(width=300, height=300, title="Prediction Distribution")
        st.altair_chart(pie.configure_title(anchor="start"))

        # 3) If true labels provided, show confusion matrix and classification report
        if "Label" in original_df.columns:
            # Expect original Label is encoded 0/1 or string; normalize to 0/1
            true_labels_raw = original_df["Label"].astype(str).tolist()
            # Map common variants to 0/1 if necessary (try numeric first)
            try:
                true_numeric = pd.to_numeric(original_df["Label"])
                true = [0 if v == 0 else 1 for v in true_numeric]
            except Exception:
                # fallback: map string values
                map_dict = {}
                uniq = list(pd.Series(true_labels_raw).unique())
                # if two unique values and one contains 'normal' map accordingly
                if len(uniq) == 2:
                    if any("normal" in s.lower() for s in uniq):
                        for s in uniq:
                            map_dict[s] = 0 if "normal" in s.lower() else 1
                    else:
                        map_dict[uniq[0]] = 0
                        map_dict[uniq[1]] = 1
                else:
                    # default attempt: treat first as 0, others as 1
                    for i, s in enumerate(uniq):
                        map_dict[s] = 0 if i == 0 else 1
                true = [map_dict.get(s, 1) for s in true_labels_raw]

            pred_numeric = [0 if p == "Normal" else 1 for p in pred_labels]

            cm = confusion_matrix(true, pred_numeric)
            cm_df = pd.DataFrame(cm, index=["True Normal", "True Attack"], columns=["Pred Normal", "Pred Attack"])
            st.markdown("### Confusion Matrix")
            # prepare cm for altair
            cm_long = cm_df.reset_index().melt(id_vars="index")
            cm_long.columns = ["True", "Pred", "Count"]
            cm_chart = alt.Chart(cm_long).mark_rect().encode(
                x=alt.X("Pred:N", title="Predicted"),
                y=alt.Y("True:N", title="Actual"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="greens")),
                tooltip=["True", "Pred", "Count"]
            ).properties(width=400, height=300)
            cm_text = cm_chart.mark_text(baseline="middle", fontSize=14).encode(text="Count:Q")
            st.altair_chart(cm_chart + cm_text)

            # Classification report
            st.markdown("### Classification Report")
            report = classification_report(true, pred_numeric, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()

            # Round numeric columns to 2 decimals for readability
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            report_df[numeric_cols] = report_df[numeric_cols].round(2)

            # Show as a plain DataFrame (avoids stylist/formatter issues)
            st.dataframe(report_df, use_container_width=True)

        else:
            st.info("Upload a CSV with a `Label` column (ground truth) to see confusion matrix and classification metrics.")

        # 4) Probability distribution (if available)
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df_clean)
                # assume positive class is index 1
                if probs.shape[1] >= 2:
                    positive_probs = probs[:, 1]
                    prob_df = pd.DataFrame({"probability": positive_probs})
                    hist = alt.Chart(prob_df).mark_bar().encode(
                        alt.X("probability:Q", bin=alt.Bin(maxbins=40)),
                        y='count()',
                        tooltip=['count()']
                    ).properties(width=500, height=250, title="Predicted Probability (positive class)")
                    st.altair_chart(hist)
                else:
                    st.info("Model.predict_proba returned a single-column array; skipping probability histogram.")
            except Exception as e:
                st.warning(f"Could not compute predict_proba: {e}")

        # 5) Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            try:
                fi = model.feature_importances_
                feat_names = list(df_clean.columns)
                fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
                fi_df = fi_df.sort_values("importance", ascending=False).head(30)  # top 30
                fi_chart = alt.Chart(fi_df).mark_bar().encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort='-x', title=None),
                    tooltip=["feature", "importance"]
                ).properties(width=600, height=400, title="Feature Importances (top 30)")
                st.altair_chart(fi_chart)
            except Exception as e:
                st.warning(f"Could not compute feature importances: {e}")

        # ---------------------------
        # END VISUALIZATIONS
        # ---------------------------

        # Show predictions table and download button below visualizations
        st.markdown("### Prediction Results")
        st.dataframe(df_result)

        csv = df_result.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except ValueError as ve:
        st.error(f"Prediction failed: {ve}")
        st.write("Tip: Ensure your input CSV has the same feature columns (and names) used during training.")
    except Exception as e:
        st.error(f"Unexpected error during prediction: {e}")
        raise
