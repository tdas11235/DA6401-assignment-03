import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.font_manager as fm


font_path = "./fonts/NotoSansDevanagari-Regular.ttf"
prop = fm.FontProperties(fname=font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = prop.get_name()

df = pd.read_csv("./predictions_vanilla/transliteration_results_van.csv")
preds = df["Predicted (Devanagari)"].astype(str)
trues = df["True (Devanagari)"].astype(str)

confusions = Counter()

for pred, true in zip(preds, trues):
    min_len = min(len(pred), len(true))
    for i in range(min_len):
        confusions[(true[i], pred[i])] += 1
    for i in range(min_len, len(true)):
        confusions[(true[i], "<PAD>")] += 1
    for i in range(min_len, len(pred)):
        confusions[("<PAD>", pred[i])] += 1

unique_chars = sorted(set(t for t, _ in confusions) |
                      set(p for _, p in confusions))
matrix = np.zeros((len(unique_chars), len(unique_chars)), dtype=int)
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}

for (true_char, pred_char), count in confusions.items():
    i = char_to_idx[true_char]
    j = char_to_idx[pred_char]
    matrix[i, j] = count

conf_df = pd.DataFrame(matrix, index=unique_chars, columns=unique_chars)

true_chars = [t for t, _ in confusions]
freq = Counter(true_chars)

# Get top-k frequent chars
top_k = 30
top_chars = [ch for ch, _ in freq.most_common(top_k)]
conf_df_top = conf_df.loc[top_chars, top_chars]


plt.figure(figsize=(16, 14))
ax = sns.heatmap(conf_df_top, annot=False, fmt='d',
                 cmap="YlOrRd", cbar=True, linewidths=0.5)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=prop,
                   rotation=90, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop,
                   rotation=0, fontsize=16)
plt.xlabel("Predicted Character", fontsize=12)
plt.ylabel("True Character", fontsize=12)
plt.title("Character-Level Confusion Matrix", fontproperties=prop, fontsize=20)
plt.savefig("character_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
