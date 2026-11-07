
import pandas as pd
from pathlib import Path

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "yield_df.csv"

print("=" * 55)
print("YIELDWISE DATASET INSPECTION")
print("=" * 55)

try:
    df = pd.read_csv(file_path)

    print("\n1. DATASET SIZE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print(df.isnull().sum())

    print("\n5. DUPLICATE RECORDS")

    print("Including index column:", df.duplicated().sum())

    # Check duplicates after excluding the index column
    if "Unnamed: 0" in df.columns:
        data_without_index = df.drop(columns=["Unnamed: 0"])
        print(
            "Excluding index column:",
            data_without_index.duplicated().sum()
        )

    print("\n6. NUMERICAL COLUMN SUMMARY")
    print(df.describe())

    print("\n7. UNIQUE VALUES")
    for column in df.columns:
        print(f"\n{column}:")
        print("Unique values:", df[column].nunique())

    print("\n8. SAMPLE RECORDS")
    print(df.head())

    print("\n9. DATASET REQUIREMENT CHECK")

    # Expected dataset schema
    expected_columns = [
        "Unnamed: 0",
        "Area",
        "Item",
        "Year",
        "hg/ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]

    missing_columns = [
        col for col in expected_columns if col not in df.columns
    ]

    print(
        "Required schema:",
        "PASS" if not missing_columns else "FAIL"
    )

    if missing_columns:
        print("Missing columns:", missing_columns)

    # Define predictors, target and excluded columns
    target_column = "hg/ha_yield"
    excluded_columns = ["Unnamed: 0", target_column]

    predictor_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    # Identify numerical and categorical predictors
    numeric_predictors = df[predictor_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_predictors = df[predictor_columns].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Requirement checks
    print(
        "500+ records:",
        "PASS" if len(df) >= 500 else "FAIL"
    )

    print(
        "6+ predictors:",
        "PASS" if len(predictor_columns) >= 6 else "FAIL"
    )

    print(
        "Target column available:",
        "PASS" if target_column in df.columns else "FAIL"
    )

    print("\nTarget column:", target_column)
    print("Total predictors:", len(predictor_columns))
    print("Numerical predictors:", numeric_predictors)
    print("Categorical predictors:", categorical_predictors)

    print(
        "\nTarget data type:",
        df[target_column].dtype if target_column in df.columns else "Not found"
    )

    print("\nInspection completed.")

except FileNotFoundError:
    print("ERROR: Dataset file not found.")
    print("Check the filename and data/raw folder.")

except Exception as error:
    print("ERROR:", error)
