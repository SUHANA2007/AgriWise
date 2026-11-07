
import os
import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "soilsense", "soilsense_train.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "soilsense", "best_soilsense_model.joblib"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports", "soilsense")
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURES = [
    "N", "P", "K", "pH", "EC", "OC",
    "S", "Zn", "Fe", "Cu", "Mn", "B"
]


# --------------------------------------------------
# LOAD MODEL AND TRAINING DATA
# --------------------------------------------------

print("=" * 55)
print("SOILSENSE - NEW SOIL FERTILITY PREDICTION")
print("=" * 55)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("SoilSense model not found.")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError("SoilSense training dataset not found.")

model = joblib.load(MODEL_PATH)
train_df = pd.read_csv(TRAIN_PATH)

# Use training data to establish valid observed ranges
minimums = train_df[FEATURES].min()
maximums = train_df[FEATURES].max()

print("\nEnter the 12 soil measurements.")
print("Values must be within the observed training-data ranges.\n")


# --------------------------------------------------
# GET AND VALIDATE INPUTS
# --------------------------------------------------

soil_values = {}

for feature in FEATURES:

    while True:
        try:
            value = float(input(f"Enter {feature}: "))

            if not np.isfinite(value):
                print("Invalid input. Enter a finite number.")
                continue

            if value < minimums[feature] or value > maximums[feature]:
                print(
                    f"Out of observed range! "
                    f"Allowed: {minimums[feature]:.3f} "
                    f"to {maximums[feature]:.3f}"
                )
                continue

            soil_values[feature] = value
            break

        except ValueError:
            print("Invalid input. Please enter a number.")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

input_df = pd.DataFrame(
    [soil_values],
    columns=FEATURES
)

prediction = model.predict(input_df)[0]

print("\n" + "=" * 55)
print("SOILSENSE PREDICTION RESULT")
print("=" * 55)

print(f"\nPredicted soil class: {prediction}")

# Show class probabilities if supported by the model
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    print("\nClass probability scores:")

    for class_value, probability in sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"Class {class_value}: {probability * 100:.2f}%")

    confidence = float(max(probabilities))

else:
    confidence = None

# --------------------------------------------------
# SAVE PREDICTION
# --------------------------------------------------

result = input_df.copy()
result["Predicted_Class"] = prediction

if confidence is not None:
    result["Highest_Class_Probability_Percent"] = round(
        confidence * 100, 2
    )

output_path = os.path.join(
    REPORT_DIR, "new_soil_prediction.csv"
)

result.to_csv(output_path, index=False)

print("\nPrediction saved to:")
print(output_path)

print("\nNote:")
print("- Class labels are shown as numerical values.")
print("- Their meanings have not been independently verified.")
print("- Probability scores are model outputs, not guaranteed accuracy.")
print("- Predictions outside the training data's observed ranges are rejected.")

print("\n" + "=" * 55)
print("PREDICTION COMPLETED")
print("=" * 55)
