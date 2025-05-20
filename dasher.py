import pandas as pd

# Load CSVs
file1 = pd.read_csv("transliteration_results_van.csv")
file2 = pd.read_csv("transliteration_results.csv")

# Get rows where file1 is wrong and file2 is correct
mask = (file1["Match"] == False) & (file2["Match"] == True)
file1_wrong_file2_correct = file1[mask]

# Save to CSV
file1_wrong_file2_correct.to_csv("file1_wrong_file2_correct.csv", index=False)
print("Saved entries where file1 was wrong and file2 was correct to 'file1_wrong_file2_correct.csv'")
