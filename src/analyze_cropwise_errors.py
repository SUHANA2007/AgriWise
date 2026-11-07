
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "cropwise" / "cropwise_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "cropwise" / "cropwise_test.csv"

MODEL_PATH = ROOT / "models" / "cropwise" / "best_cropwise_model.joblib"

REPORT_DIR = ROOT / "reports" / "cropwise"

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

TARGET = "label"


# 2. Load data and model
def load_data():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    model = joblib.load(MODEL_PATH)

    X_test = test[FEATURES]
    y_test = test[TARGET]

    predictions = model.predict(X_test)

    results = test.copy()
    results["Predicted_Crop"] = predictions
    results["Correct"] = (
        results[TARGET] == results["Predicted_Crop"]
    )

    return train, results


# 3. Crop-wise error analysis
def analyze_crop_errors(results):

    crop_report = (
        results.groupby(TARGET)
        .agg(
            Total_Records=("Correct", "size"),
            Correct_Predictions=("Correct", "sum")
        )
        .reset_index()
    )

    crop_report["Errors"] = (
        crop_report["Total_Records"]
        - crop_report["Correct_Predictions"]
    )

    crop_report["Accuracy"] = (
        crop_report["Correct_Predictions"]
        / crop_report["Total_Records"]
    )

    crop_report["Error_Rate"] = (
        crop_report["Errors"]
        / crop_report["Total_Records"]
    )

    crop_report.to_csv(
        REPORT_DIR / "error_by_crop.csv",
        index=False
    )

    print("\n===== ERROR ANALYSIS BY CROP =====")
    print(crop_report.to_string(index=False))


# 4. Feature-range error analysis
def analyze_feature_ranges(train, results):

    all_slices = []

    for feature in FEATURES:

        # Use training-data quantiles to define ranges
        _, bins = pd.qcut(
            train[feature],
            q=3,
            retbins=True,
            duplicates="drop"
        )

        bins[0] = float("-inf")
        bins[-1] = float("inf")

        if len(bins) < 3:
            print(f"Skipping {feature}: insufficient unique ranges.")
            continue

        range_labels = [
            f"Range {i + 1}"
            for i in range(len(bins) - 1)
        ]

        feature_results = results.copy()

        feature_results["Feature_Range"] = pd.cut(
            feature_results[feature],
            bins=bins,
            labels=range_labels,
            include_lowest=True
        )

        grouped = (
            feature_results.groupby(
                "Feature_Range",
                observed=False
            )
            .agg(
                Total_Records=("Correct", "size"),
                Correct_Predictions=("Correct", "sum")
            )
            .reset_index()
        )

        grouped["Feature"] = feature
        grouped["Errors"] = (
            grouped["Total_Records"]
            - grouped["Correct_Predictions"]
        )

        grouped["Accuracy"] = (
            grouped["Correct_Predictions"]
            / grouped["Total_Records"].replace(0, pd.NA)
        )

        grouped["Error_Rate"] = (
            grouped["Errors"]
            / grouped["Total_Records"].replace(0, pd.NA)
        )

        all_slices.append(grouped)

    if all_slices:

        feature_report = pd.concat(
            all_slices,
            ignore_index=True
        )

        feature_report = feature_report[
            [
                "Feature",
                "Feature_Range",
                "Total_Records",
                "Correct_Predictions",
                "Errors",
                "Accuracy",
                "Error_Rate"
            ]
        ]

        feature_report.to_csv(
            REPORT_DIR / "error_by_feature_range.csv",
            index=False
        )

        print("\n===== ERROR ANALYSIS BY FEATURE RANGE =====")
        print(feature_report.to_string(index=False))


# 5. Save incorrectly classified records
def save_misclassified_records(results):

    mistakes = results[
        results["Correct"] == False
    ].copy()

    mistakes.to_csv(
        REPORT_DIR / "misclassified_records.csv",
        index=False
    )

    print("\n===== MISCLASSIFIED RECORDS =====")

    if mistakes.empty:
        print("No misclassified records found.")
    else:
        print(
            mistakes[
                FEATURES + [TARGET, "Predicted_Crop"]
            ].to_string(index=False)
        )

    print("\nTotal test records:", len(results))
    print("Correct predictions:", int(results["Correct"].sum()))
    print("Incorrect predictions:", len(mistakes))
    print(
        "Overall accuracy:",
        round(accuracy_score(
            results[TARGET],
            results["Predicted_Crop"]
        ), 4)
    )


# 6. Main function
def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== CropWise Error Analysis =====")

    train, results = load_data()

    analyze_crop_errors(results)

    analyze_feature_ranges(train, results)

    save_misclassified_records(results)

    print("\n===== Error Analysis Completed =====")


if __name__ == "__main__":
    main()
