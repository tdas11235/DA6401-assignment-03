import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv(
    "dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.aligned.cased_nopunct.tsv",
    sep="\t",
    names=["target", "input"],
    dtype=str,
)

df = df.dropna()

os.makedirs("data", exist_ok=True)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv("data/train.tsv", sep="\t", index=False, header=False)
val_df.to_csv("data/val.tsv", sep="\t", index=False, header=False)
test_df.to_csv("data/test.tsv", sep="\t", index=False, header=False)

print("Split complete. Files written to data/train.tsv, val.tsv, and test.tsv")
