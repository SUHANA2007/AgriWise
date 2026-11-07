
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent

ERROR_FILE = (
    BASE_DIR / "models" / "leafguard"
    / "validation_misclassified.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "models" / "leafguard"
    / "validation_error_samples.png"
)

df = pd.read_csv(ERROR_FILE)

if df.empty:
    print("No misclassified images found.")
    raise SystemExit

# Select up to 20 examples.
samples = df.head(20).reset_index(drop=True)

fig, axes = plt.subplots(4, 5, figsize=(20, 17))

for index, ax in enumerate(axes.flat):

    if index >= len(samples):
        ax.axis("off")
        continue

    row = samples.iloc[index]

    image_path = Path(row["image_path"])

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    actual = row["actual_class"]
    predicted = row["predicted_class"]

    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        fontsize=7
    )

    ax.axis("off")

plt.suptitle(
    "LeafGuard - Validation Misclassified Samples",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Error visualization saved:")
print(OUTPUT_FILE)
