
import os
import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "fertilizerwise",
    "fertilizerwise_train.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "fertilizerwise",
    "best_fertilizerwise_model.joblib"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports", "fertilizerwise")
os.makedirs(REPORT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "Nitrogen", "Phosphorus", "Potassium",
    "pH", "Rainfall", "Temperature"
]

CATEGORICAL_FEATURES = [
    "District_Name", "Soil_color", "Crop"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# LOAD MODEL AND TRAINING DATA
# --------------------------------------------------

print("=" * 60)
print("FERTILIZERWISE - NEW FERTILIZER RECOMMENDATION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("FertilizerWise model not found.")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError("Training dataset not found.")

model = joblib.load(MODEL_PATH)
train_df = pd.read_csv(TRAIN_PATH)

# Establish valid observed input ranges and categories
minimums = train_df[NUMERIC_FEATURES].min()
maximums = train_df[NUMERIC_FEATURES].max()

valid_categories = {
    feature: sorted(train_df[feature].dropna().unique().tolist())
    for feature in CATEGORICAL_FEATURES
}


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

input_values = {}

print("\nEnter the soil, weather and crop details.\n")

for feature in NUMERIC_FEATURES:

    while True:
        try:
            value = float(input(f"Enter {feature}: "))

            if not np.isfinite(value):
                print("Enter a valid finite number.")
                continue

            if value < minimums[feature] or value > maximums[feature]:
                print(
                    f"Value outside observed training range. "
                    f"Allowed: {minimums[feature]:.2f} "
                    f"to {maximums[feature]:.2f}"
                )
                continue

            input_values[feature] = value
            break

        except ValueError:
            print("Invalid input. Please enter a number.")


for feature in CATEGORICAL_FEATURES:

    print(f"\nAvailable {feature} values:")
    print(", ".join(str(x) for x in valid_categories[feature]))

    while True:
        value = input(f"\nEnter {feature}: ").strip()

        if value in valid_categories[feature]:
            input_values[feature] = value
            break

        print("Invalid value. Choose one from the displayed list.")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

input_df = pd.DataFrame(
    [[input_values[feature] for feature in FEATURES]],
    columns=FEATURES
)

prediction = model.predict(input_df)[0]

print("\n" + "=" * 60)
print("FERTILIZERWISE PREDICTION RESULT")
print("=" * 60)

print("\nRecommended fertilizer:", prediction)

confidence = None

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    confidence = float(max(probabilities))

    print("\nTop 5 model probability scores:")

    top_indices = np.argsort(probabilities)[::-1][:5]

    for index in top_indices:
        print(
            f"{classes[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

result = input_df.copy()
result["Recommended_Fertilizer"] = prediction

if confidence is not None:
    result["Highest_Class_Probability_Percent"] = round(
        confidence * 100, 2
    )

output_path = os.path.join(
    REPORT_DIR, "new_fertilizer_prediction.csv"
)

result.to_csv(output_path, index=False)

print("\nPrediction saved to:")
print(output_path)

print("\nNote:")
print("- Recommendations reflect patterns in the training dataset.")
print("- This dataset represents five Western Maharashtra districts.")
print("- Probability scores are not guaranteed correctness.")
print("- Verify fertilizer decisions with reliable agricultural guidance.")

print("\n" + "=" * 60)
print("FERTILIZERWISE PREDICTION COMPLETED")
print("=" * 60)
