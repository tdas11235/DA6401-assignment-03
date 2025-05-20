import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("transliteration_results_van.csv")
df2 = pd.read_csv("transliteration_results.csv")

# Find rows where:
# - Correctness differs
# - file2 is correct
mask = (
    df1["Match"] != df2["Match"]
) & (
    df2["Match"] == True
)

# Pick those rows from file2 (or df1 for comparison)
file2_correct_only = df2[mask].copy()

# Optional: add file1's prediction for comparison
# file2_correct_only["file1_prediction"] = df1.loc[mask, "Predicted"].values

# Save or view
file2_correct_only.to_csv("file2_correct_only.csv", index=False)
print(file2_correct_only)
