
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "leafguard"
MODEL_DIR = BASE_DIR / "models" / "leafguard"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
CLASS_MAPPING_FILE = DATA_DIR / "class_mapping.json"

BEST_MODEL_PATH = MODEL_DIR / "leafguard_best_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"
REPORT_PATH = MODEL_DIR / "validation_report.txt"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE CONFIGURATION
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(8, torch.get_num_threads()))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("LEAFGUARD - MOBILENETV2 TRANSFER LEARNING")
print("=" * 65)

print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")


# ============================================================
# 3. LOAD CLASS MAPPING AND DATA
# ============================================================

with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)

if not isinstance(class_mapping, dict):
    raise ValueError("class_mapping.json must contain a class-to-index dictionary.")

class_names = sorted(class_mapping, key=lambda name: class_mapping[name])
num_classes = len(class_names)

if num_classes != 38:
    raise ValueError(f"Expected 38 classes, found {num_classes}.")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

required_columns = {"image_path", "class_name", "class_index"}

for split_name, df in [("Training", train_df), ("Validation", val_df)]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} CSV is missing columns: {missing_columns}"
        )

    if df["image_path"].isna().any() or df["class_name"].isna().any():
        raise ValueError(f"{split_name} CSV contains missing values.")

    unknown_classes = set(df["class_name"]) - set(class_mapping)

    if unknown_classes:
        raise ValueError(
            f"Unknown classes in {split_name}: {unknown_classes}"
        )

    expected_indices = df["class_name"].map(class_mapping)

    if not np.array_equal(
        expected_indices.to_numpy(),
        df["class_index"].to_numpy()
    ):
        raise ValueError(
            f"{split_name} class indices do not match class_mapping.json."
        )

print(f"\nClasses: {num_classes}")
print(f"Training images: {len(train_df):,}")
print(f"Validation images: {len(val_df):,}")


# ============================================================
# 4. IMAGE TRANSFORMATIONS
# ============================================================

# ImageNet normalization values used by pretrained MobileNetV2.

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ============================================================
# 5. CUSTOM DATASET LOADER
# ============================================================

class LeafGuardDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])
        label = int(row["class_index"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            ) from error

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = LeafGuardDataset(
    train_df,
    transform=train_transform
)

val_dataset = LeafGuardDataset(
    val_df,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print("\nImage datasets and DataLoaders prepared.")


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["class_index"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .sort_index()
)

if (class_counts == 0).any():
    raise ValueError("At least one class has no training images.")

class_weights = len(train_df) / (
    num_classes * class_counts.to_numpy(dtype=np.float64)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights calculated from training data.")
print(f"Lowest class count: {class_counts.min()}")
print(f"Highest class count: {class_counts.max()}")


# ============================================================
# 7. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading pretrained MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)

# Freeze the complete feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace the original 1000-class output layer with 38 classes.
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)

model = model.to(DEVICE)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("MobileNetV2 loaded.")
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print("Feature extractor: Frozen")
print("Classification layer: Trainable")


# ============================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# 9. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate_one_epoch():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 10. TRAINING LOOP WITH EARLY STOPPING
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = []

training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    print("\n" + "-" * 65)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("-" * 65)

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate_one_epoch()

    epoch_time = time.time() - epoch_start

    print(f"Training loss:   {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Epoch duration: {epoch_time / 60:.2f} minutes")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "duration_minutes": epoch_time / 60
    })

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_mapping": class_mapping,
            "class_names": class_names,
            "num_classes": num_classes,
            "image_size": IMAGE_SIZE,
            "mean": mean,
            "std": std,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "architecture": "MobileNetV2"
        }

        torch.save(checkpoint, BEST_MODEL_PATH)

        print("\nBest model checkpoint saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo validation loss improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# ============================================================
# 11. LOAD BEST CHECKPOINT AND VALIDATE
# ============================================================

print("\nLoading best model checkpoint...")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_labels = []

with torch.inference_mode():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())


report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
    digits=4
)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(report)


# ============================================================
# 12. FINAL TRAINING SUMMARY
# ============================================================

total_time = time.time() - training_start

print("\n" + "=" * 65)
print("LEAFGUARD TRAINING SUMMARY")
print("=" * 65)

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(
    f"Best epoch validation accuracy: "
    f"{history[best_epoch - 1]['validation_accuracy'] * 100:.2f}%"
)

print(f"Total training time: {total_time / 60:.2f} minutes")

print(f"\nBest model: {BEST_MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")
print(f"Validation report: {REPORT_PATH}")

print("\nTraining and validation completed.")
print("The test dataset was not used.")
print("=" * 65)
