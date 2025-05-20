import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("transliteration_results_van.csv")
df2 = pd.read_csv("transliteration_results.csv")

# Get correct predictions from each
correct_file1 = df1[df1["Match"] == True]
correct_file2 = df2[df2["Match"] == True]

# Save to separate files
correct_file1.to_csv("correct_preds_file1.csv", index=False)
correct_file2.to_csv("correct_preds_file2.csv", index=False)

wrong_file2 = df2[df2["Match"] == False]
wrong_file2.to_csv("wrong_preds_file2.csv", index=False)

print("Saved correct predictions to:")
print("- correct_preds_file1.csv")
print("- correct_preds_file2.csv")
