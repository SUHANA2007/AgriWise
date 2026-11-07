
from pathlib import Path
import argparse

import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import mobilenet_v2


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "leafguard"
    / "leafguard_best_model.pth"
)

DEVICE = torch.device("cpu")


# ============================================================
# 2. LOAD TRAINED CHECKPOINT
# ============================================================

print("=" * 65)
print("LEAFGUARD - PLANT DISEASE PREDICTION")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
image_size = checkpoint["image_size"]

print("\nTrained model:", checkpoint["architecture"])
print("Number of classes:", num_classes)


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["mean"],
        std=checkpoint["std"]
    )
])


# ============================================================
# 4. REBUILD MODEL
# ============================================================

model = mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_leaf(image_path):

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "The selected file is not a valid readable image."
        ) from error

    input_tensor = input_tensor.to(DEVICE)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(3, num_classes)
        )

    results = []

    for probability, index in zip(
        top_probabilities.tolist(),
        top_indices.tolist()
    ):

        results.append({
            "class": class_names[index],
            "confidence": probability * 100
        })

    return results


# ============================================================
# 6. DISPLAY PREDICTION
# ============================================================

def display_prediction(image_path):

    results = predict_leaf(image_path)

    print("\n" + "=" * 65)
    print("PREDICTION RESULTS")
    print("=" * 65)

    print("\nImage:", image_path)

    print("\nTop prediction:")
    print("Plant / Disease:", results[0]["class"])
    print(
        "Confidence: "
        f"{results[0]['confidence']:.2f}%"
    )

    print("\nTop 3 predictions:")

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. {result['class']} "
            f"({result['confidence']:.2f}%)"
        )

    print("\n" + "-" * 65)

    if results[0]["confidence"] < 60:

        print(
            "WARNING: Prediction confidence is relatively low."
        )
        print(
            "Consider checking the image quality and "
            "seeking expert verification."
        )

    else:

        print(
            "The model has produced a prediction. "
            "It is not a confirmed plant disease diagnosis."
        )

    print("\nSupported classes:", num_classes)
    print(
        "This model is trained on PlantVillage images. "
        "Performance on real-world field photographs "
        "may differ."
    )

    print("=" * 65)


# ============================================================
# 7. COMMAND-LINE INPUT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict plant disease using LeafGuard."
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the leaf image."
    )

    args = parser.parse_args()

    image_path = args.image

    if not image_path:
        image_path = input(
            "\nEnter leaf image path: "
        ).strip().strip('"')

    try:
        display_prediction(image_path)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nPrediction error: {error}")
