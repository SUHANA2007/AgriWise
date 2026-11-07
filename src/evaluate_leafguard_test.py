
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "leafguard"
MODEL_PATH = BASE_DIR / "models" / "leafguard" / "leafguard_best_model.pth"

OUTPUT_DIR = BASE_DIR / "models" / "leafguard"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LEAFGUARD - INDEPENDENT TEST DATASET EVALUATION")
print("=" * 70)


# ============================================================
# 2. LOAD TRAINED MODEL CHECKPOINT
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_mapping = checkpoint["class_mapping"]
class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

IMAGE_SIZE = checkpoint["image_size"]

print("\nModel information:")
print("Architecture:", checkpoint["architecture"])
print("Number of classes:", num_classes)
print("Image size:", IMAGE_SIZE)
print("Best training epoch:", checkpoint["best_epoch"])
print("Best validation loss:", checkpoint["best_validation_loss"])


# ============================================================
# 3. LOAD UNTOUCHED TEST DATASET
# ============================================================

TEST_CSV = DATA_DIR / "test.csv"

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

df = pd.read_csv(TEST_CSV)

required_columns = {"image_path", "class_index"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Test CSV must contain these columns: {required_columns}"
    )

if df.empty:
    raise ValueError("Test dataset is empty.")

if df["class_index"].isna().any():
    raise ValueError("Test dataset contains missing class indices.")

df["class_index"] = df["class_index"].astype(int)

valid_indices = set(range(num_classes))

if not set(df["class_index"].unique()).issubset(valid_indices):
    raise ValueError("Test dataset contains invalid class indices.")

if set(df["class_index"].unique()) != valid_indices:
    raise ValueError("One or more classes are missing from the test dataset.")

print("\nTest dataset records:", len(df))
print("Test dataset classes:", df["class_index"].nunique())


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["mean"],
        std=checkpoint["std"]
    )
])


# ============================================================
# 5. TEST DATASET CLASS
# ============================================================

class LeafGuardTestDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(row["image_path"])

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = transform(image)

        label = int(row["class_index"])

        return image, label


# ============================================================
# 6. VALIDATE TEST IMAGE PATHS
# ============================================================

print("\nChecking test image paths...")

missing_files = []

for image_path in df["image_path"]:

    path = Path(image_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        missing_files.append(str(path))

if missing_files:

    print("Missing image files:", len(missing_files))

    raise FileNotFoundError(
        "Test dataset contains missing images. "
        f"First missing file: {missing_files[0]}"
    )

print("All test image paths verified.")


# ============================================================
# 7. CREATE DATA LOADER
# ============================================================

dataset = LeafGuardTestDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. REBUILD AND LOAD MODEL
# ============================================================

model = mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("\nTrained model loaded successfully.")


# ============================================================
# 9. RUN TEST PREDICTIONS
# ============================================================

all_actual = []
all_predicted = []
all_confidence = []

print("\nRunning test predictions...")

with torch.inference_mode():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        all_actual.extend(labels.tolist())
        all_predicted.extend(predictions.tolist())
        all_confidence.extend(confidence.tolist())

print("Test predictions completed.")


# ============================================================
# 10. CALCULATE OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_actual,
    all_predicted
)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_actual,
    all_predicted,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_actual,
        all_predicted,
        average="weighted",
        zero_division=0
    )
)

correct = sum(
    actual == predicted
    for actual, predicted in zip(all_actual, all_predicted)
)

incorrect = len(all_actual) - correct


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_actual,
    all_predicted,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

print("\n" + "=" * 70)
print("LEAFGUARD - TEST CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predicted,
    labels=list(range(num_classes))
)

confusion_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

confusion_path = OUTPUT_DIR / "test_confusion_matrix.csv"

confusion_df.to_csv(confusion_path)


# ============================================================
# 13. SAVE PER-IMAGE PREDICTIONS
# ============================================================

prediction_records = []

for i in range(len(all_actual)):

    actual = all_actual[i]
    predicted = all_predicted[i]

    prediction_records.append({
        "image_path": df.iloc[i]["image_path"],
        "actual_class": class_names[actual],
        "predicted_class": class_names[predicted],
        "confidence": round(all_confidence[i], 6),
        "correct": actual == predicted
    })

predictions_df = pd.DataFrame(prediction_records)

predictions_path = OUTPUT_DIR / "test_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 14. SAVE MISCLASSIFIED TEST IMAGES
# ============================================================

misclassified_df = predictions_df[
    predictions_df["correct"] == False
].copy()

misclassified_path = OUTPUT_DIR / "test_misclassified.csv"

misclassified_df.to_csv(
    misclassified_path,
    index=False
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    all_actual,
    all_predicted,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = OUTPUT_DIR / "test_classification_report.csv"

report_df.to_csv(report_path)


# ============================================================
# 16. SAVE EVALUATION SUMMARY
# ============================================================

summary = f"""
LEAFGUARD - TEST EVALUATION SUMMARY
===================================

Architecture: {checkpoint['architecture']}
Best Training Epoch: {checkpoint['best_epoch']}

Test Images: {len(all_actual)}
Number of Classes: {num_classes}

Correct Predictions: {correct}
Incorrect Predictions: {incorrect}

Test Accuracy: {accuracy * 100:.2f}%

Macro Precision: {precision * 100:.2f}%
Macro Recall: {recall * 100:.2f}%
Macro F1 Score: {f1 * 100:.2f}%

Weighted Precision: {weighted_precision * 100:.2f}%
Weighted Recall: {weighted_recall * 100:.2f}%
Weighted F1 Score: {weighted_f1 * 100:.2f}%

Evaluation completed successfully.
"""

summary_path = OUTPUT_DIR / "test_evaluation_summary.txt"

summary_path.write_text(
    summary,
    encoding="utf-8"
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LEAFGUARD - FINAL TEST RESULTS")
print("=" * 70)

print(f"Total test images : {len(all_actual)}")
print(f"Correct           : {correct}")
print(f"Incorrect         : {incorrect}")

print(f"\nTest Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1 Score    : {f1 * 100:.2f}%")

print("\nSaved evaluation files:")

print(summary_path)
print(report_path)
print(confusion_path)
print(predictions_path)
print(misclassified_path)

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)
