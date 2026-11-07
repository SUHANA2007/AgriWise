
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "leafguard"
MODEL_PATH = BASE_DIR / "models" / "leafguard" / "leafguard_best_model.pth"
OUTPUT_DIR = BASE_DIR / "models" / "leafguard"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
IMAGE_SIZE = 224

print("=" * 65)
print("LEAFGUARD - VALIDATION CONFUSION ANALYSIS")
print("=" * 65)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_mapping = checkpoint["class_mapping"]
class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

df = pd.read_csv(DATA_DIR / "validation.csv")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["mean"],
        std=checkpoint["std"]
    )
])


class ValidationDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = transform(image)

        return image, int(row["class_index"])


dataset = ValidationDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

all_actual = []
all_predicted = []
misclassified = []

print("\nRunning validation predictions...")

with torch.inference_mode():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_actual.extend(labels.numpy().tolist())
        all_predicted.extend(predictions.cpu().numpy().tolist())

print("Validation predictions completed.")

matrix = confusion_matrix(
    all_actual,
    all_predicted,
    labels=list(range(num_classes))
)

matrix_df = pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names
)

matrix_df.to_csv(
    OUTPUT_DIR / "validation_confusion_matrix.csv"
)

for index, (actual, predicted) in enumerate(
    zip(all_actual, all_predicted)
):

    if actual != predicted:
        misclassified.append({
            "image_path": df.iloc[index]["image_path"],
            "actual_class": class_names[actual],
            "predicted_class": class_names[predicted]
        })

errors_df = pd.DataFrame(misclassified)

errors_df.to_csv(
    OUTPUT_DIR / "validation_misclassified.csv",
    index=False
)

print("\n" + "=" * 65)
print("TOP 15 CLASS CONFUSIONS")
print("=" * 65)

confusions = []

for actual_index in range(num_classes):

    for predicted_index in range(num_classes):

        if actual_index != predicted_index:
            count = matrix[actual_index, predicted_index]

            if count > 0:
                confusions.append({
                    "Actual class": class_names[actual_index],
                    "Predicted as": class_names[predicted_index],
                    "Count": int(count)
                })

confusions.sort(
    key=lambda item: item["Count"],
    reverse=True
)

for item in confusions[:15]:
    print(
        f"{item['Actual class']} -> "
        f"{item['Predicted as']}: {item['Count']}"
    )

print("\nTotal validation images:", len(all_actual))
print("Misclassified images:", len(misclassified))

print("\nSaved:")
print(OUTPUT_DIR / "validation_confusion_matrix.csv")
print(OUTPUT_DIR / "validation_misclassified.csv")

print("\nCONFUSION ANALYSIS COMPLETED")
print("=" * 65)
