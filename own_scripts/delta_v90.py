# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# Load the data, skipping the first row which is a header row, and ensuring proper column names
file_path = "tau_data.csv"

# Skip the first row, which is treated as data, and provide correct column names
df = pd.read_csv(file_path, header=1, names=["Ray Number", "Tau Value", "X Value"])

# Inspect the first few rows of the dataframe to ensure correct column names
print(df.head())

# Remove any rows where Ray Number is -1 (i.e., delimiters)
df = df[df["Ray Number"] != -1]

# Check if there are any rows that still have NaN values and drop them
df = df.dropna(subset=["Ray Number", "Tau Value", "X Value"])

# Convert necessary columns to appropriate data types
df["X Value"] = pd.to_numeric(df["X Value"], errors="coerce")  # Ensure this is a numeric column
df["Tau Value"] = pd.to_numeric(df["Tau Value"], errors="coerce")  # Convert Tau to float

# Group by Ray Number and calculate delta_v_90 for each group
results = []
for ray_number, group in df.groupby("Ray Number"):
    # Compute cumulative sum of Tau for the group
    group["Cumulative Tau"] = group["Tau Value"].cumsum()

    # Drop any rows with NaN values in Cumulative Tau after summing (if any)
    group = group.dropna(subset=["Cumulative Tau"])

    # Normalize cumulative Tau using scikit-learn
    group["Normalized Tau"] = normalize(group["Cumulative Tau"].values.reshape(1, -1), norm="max").flatten()

    # Identify 5% and 95% normalized Tau values
    five_percent_tau = 0.05
    ninety_five_percent_tau = 0.95
    v_5 = group.iloc[(group["Normalized Tau"] - five_percent_tau).abs().argmin()]['X Value']
    v_95 = group.iloc[(group["Normalized Tau"] - ninety_five_percent_tau).abs().argmin()]['X Value']
    delta_v_90 = abs(v_95 - v_5)

    # Append result for this ray
    results.append({"Ray Number": ray_number, "Delta V_90": delta_v_90})

# Convert results to a DataFrame
delta_v_90_df = pd.DataFrame(results)

# Save results to a new CSV
delta_v_90_df.to_csv("delta_v90.csv", index=False)

# Print the results for verification
print(delta_v_90_df)
