# Script debugged and modified with the help of OpenAI. (2025). ChatGPT (June 11 version) [Large language model]. https://chat.openai.com/



import pandas as pd
import re

# Path to your .dat file
file_path = "/Users/patrickbates/bachelors_cloudflex/lise_table/table.dat"

# Read file contents
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = f.read()

# Fix unusual characters
raw_data = raw_data.replace('−', '-')            # Replace unicode minus
raw_data = re.sub(r'[<←]', '-', raw_data)        # Replace < or ← with minus (treat as negative)

# Remove header and divider lines
lines = [line for line in raw_data.strip().split('\n') 
         if line.strip() and not line.startswith('---') and not line.startswith('NAME')]

records = []
for line in lines:
    parts = line.strip().split()
    
    # First column is NAME — could contain a space (e.g., Q1313+1441)
    if len(parts) >= 11:
        name = parts[0]
        rest = parts[1:]
    else:
        # If first part and second part form the name (e.g. 'Q1313+1441 1.7941')
        name = parts[0]
        rest = parts[1:]

    if len(rest) == 10:
        records.append([name] + rest)
    else:
        print(f"⚠️ Skipped malformed line: {line}")

# Define column names
columns = ["NAME", "zabs", "logNHI", "err_logNHI", "M/H", "err_M/H", 
           "logM*", "lower_logM*", "b", "errb", "v90"]

# Create DataFrame
df = pd.DataFrame(records, columns=columns)

# Convert to numeric and handle missing values
df.replace('-99', pd.NA, inplace=True)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Show result
print(df.head())

# Save DataFrame to CSV
df.to_csv("/Users/patrickbates/bachelors_cloudflex/lise_table/cleaned_table.csv", index=False)
