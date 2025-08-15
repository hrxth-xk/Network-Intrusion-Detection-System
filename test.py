import pandas as pd

df = pd.read_csv("data/processed/cleaned_dataset.csv")
df_test = df.sample(1000, random_state=42)  # take 1000 random rows
df_test.drop(columns=["Label"], inplace=True)  # remove label so model predicts
df_test.to_csv("data/raw/test_sample.csv", index=False)
print("Test CSV created at data/raw/test_sample.csv")