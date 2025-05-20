################ Correct plotting code ###############################


import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import random

# Load Devanagari font
font_path = "./fonts/NotoSansDevanagari-Regular.ttf"
font = ImageFont.truetype(font_path, size=28)

correct_file1 = pd.read_csv("./correct_preds_file1.csv")
only_file2_correct = pd.read_csv("./file1_wrong_file2_correct.csv")

file1_samples = correct_file1.sample(n=10, random_state=52)
rows_model1 = [
    (row["Input (Roman)"], row["Predicted (Devanagari)"],
     row["True (Devanagari)"], True)
    for _, row in file1_samples.iterrows()
]
file2_samples = only_file2_correct.sample(n=10, random_state=52)
rows_model2 = [
    (row["Input (Roman)"], row["Predicted (Devanagari)"],
     row["True (Devanagari)"], False)
    for _, row in file2_samples.iterrows()
]

rows = rows_model1 + rows_model2
random.shuffle(rows)

# Layout configuration
row_height = 50
padding = 20
img_width = 1000
img_height = padding * 2 + row_height * (len(rows) + 1)
x_positions = [50, 350, 650]
headers = ['Input', 'Predicted', 'True']

img = Image.new("RGB", (img_width, img_height), "white")
draw = ImageDraw.Draw(img)

for x, header in zip(x_positions, headers):
    draw.text((x, padding), header, fill="black", font=font)

for i, (inp, pred, true, correct) in enumerate(rows):
    y = padding + row_height * (i + 1)
    bg_colors = ['lightyellow', 'lightgreen' if correct else 'lightcoral', 'lightblue']
    values = [inp, pred, true]
    for j in range(3):
        draw.rectangle([x_positions[j]-10, y-5, x_positions[j]+260, y+35], fill=bg_colors[j])
        draw.text((x_positions[j], y), values[j], fill="black", font=font)

plt.figure(figsize=(12, 0.5 + 0.5 * len(rows)))
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
img.save("hindi_predictions.png")
plt.show()
