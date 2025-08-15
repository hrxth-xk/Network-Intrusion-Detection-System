Network Intrusion Detection System (NIDS) with Machine Learning

This project implements a Network Intrusion Detection System using machine learning, capable of detecting normal vs malicious network traffic. It uses the CICIDS2017 dataset and supports Random Forest and XGBoost models with a user-friendly Streamlit dashboard.

Features

Train Random Forest and XGBoost models on CICIDS2017 data.

Preprocessing: handle missing/infinite values, numeric columns only.

Predict network traffic from CSV files.

Streamlit dashboard:

Upload CSVs for predictions

View prediction summary: total traffic, attack count, normal count

Download prediction results

Easily deployable on Streamlit Cloud.