import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "soilsense", "soilsense_train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "soilsense", "soilsense_test.csv"
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

TARGET = "Output"


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

print("=" * 60)
print("SOILSENSE - ERROR ANALYSIS")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

model = joblib.load(MODEL_PATH)

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_test = test_df[TARGET]
y_pred = model.predict(X_test)

errors = (y_test.to_numpy() != y_pred)


# --------------------------------------------------
# ANALYSIS 1: CLASS-WISE ERRORS
# --------------------------------------------------

print("\n1. CLASS-WISE ERROR ANALYSIS")

class_report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

class_df = pd.DataFrame(class_report).transpose()
class_df.to_csv(
    os.path.join(REPORT_DIR, "error_by_class.csv")
)

class_summary = []

for class_value in sorted(y_test.unique()):
    mask = y_test.to_numpy() == class_value
    total = int(mask.sum())
    incorrect = int(
        (y_test.to_numpy()[mask] != y_pred[mask]).sum()
    )

    class_summary.append({
        "Class": class_value,
        "Total_Records": total,
        "Incorrect_Predictions": incorrect,
        "Correct_Predictions": total - incorrect,
        "Error_Rate_Percent": round(
            incorrect / total * 100, 2
        ) if total else 0
    })

class_summary_df = pd.DataFrame(class_summary)

class_summary_df.to_csv(
    os.path.join(REPORT_DIR, "class_error_summary.csv"),
    index=False
)

print(class_summary_df.to_string(index=False))


# --------------------------------------------------
# ANALYSIS 2: FEATURE-RANGE ERROR ANALYSIS
# --------------------------------------------------

print("\n2. FEATURE-RANGE ERROR ANALYSIS")

feature_results = []

for feature in FEATURES:

    # Calculate boundaries using training data only
    boundaries = X_train[feature].quantile(
        [0, 1/3, 2/3, 1]
    ).to_numpy()

    boundaries = np.unique(boundaries)

    if len(boundaries) < 3:
        print(f"Skipping {feature}: insufficient unique values.")
        continue

    bins = pd.cut(
        X_test[feature],
        bins=boundaries,
        include_lowest=True,
        duplicates="drop"
    )

    for interval in bins.dropna().unique():

        mask = (bins == interval).to_numpy()

        total = int(mask.sum())

        if total == 0:
            continue

        incorrect = int(errors[mask].sum())

        feature_results.append({
            "Feature": feature,
            "Feature_Range": str(interval),
            "Total_Records": total,
            "Incorrect_Predictions": incorrect,
            "Correct_Predictions": total - incorrect,
            "Error_Rate_Percent": round(
                incorrect / total * 100, 2
            )
        })

feature_df = pd.DataFrame(feature_results)

feature_df.to_csv(
    os.path.join(REPORT_DIR, "error_by_feature_range.csv"),
    index=False
)

print(feature_df.to_string(index=False))


# --------------------------------------------------
# SAVE MISCLASSIFIED RECORDS
# --------------------------------------------------

print("\n3. MISCLASSIFIED RECORDS")

misclassified = test_df.copy()
misclassified["Actual"] = y_test.to_numpy()
misclassified["Predicted"] = y_pred
misclassified["Correct"] = ~errors

misclassified = misclassified[
    misclassified["Correct"] == False
]

misclassified.to_csv(
    os.path.join(REPORT_DIR, "misclassified_records.csv"),
    index=False
)

print("Total incorrect predictions:", len(misclassified))


# --------------------------------------------------
# SAVE CONFUSION MATRIX
# --------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Predicted_{x}" for x in labels]
)

cm_df.to_csv(
    os.path.join(REPORT_DIR, "error_analysis_confusion_matrix.csv")
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

accuracy = (y_test.to_numpy() == y_pred).mean()

summary = f"""
SOILSENSE ERROR ANALYSIS SUMMARY

Total test records: {len(y_test)}
Correct predictions: {len(y_test) - len(misclassified)}
Incorrect predictions: {len(misclassified)}
Test accuracy: {accuracy:.4f}

Analysis 1:
Class-wise error analysis saved.

Analysis 2:
Feature-range error analysis saved.

Misclassified records saved.

Note:
Class labels 0, 1 and 2 are retained as numerical
labels. Their meanings have not been independently verified.
"""

with open(
    os.path.join(REPORT_DIR, "error_analysis_summary.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("=" * 60)
print("SOILSENSE ERROR ANALYSIS COMPLETED")
print("=" * 60)
print("Reports saved in:", REPORT_DIR)