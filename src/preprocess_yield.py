
from pathlib import Path
import pandas as pd

# Project paths
RAW_PATH = Path("data/raw/yield_df.csv")
OUTPUT_DIR = Path("data/processed/yieldwise")

# Dataset columns
KEY_COLUMNS = ["Area", "Item", "Year"]

INPUT_COLUMNS = [
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

TARGET_COLUMN = "hg/ha_yield"

FEATURE_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load original dataset
    df = pd.read_csv(RAW_PATH)
    original_count = len(df)

    print("Original records:", original_count)

    # 2. Remove meaningless index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # 3. Clean categorical values
    for col in ["Area", "Item"]:
        df[col] = df[col].astype(str).str.strip()

    # 4. Convert numerical columns
    numeric_columns = [
        "Year",
        *INPUT_COLUMNS,
        TARGET_COLUMN
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Validate required columns and missing values
    required_columns = KEY_COLUMNS + INPUT_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing or invalid values found.")

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("Negative yield values found.")

    # 6. Remove exact duplicate observations
    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=required_columns
    ).copy()

    exact_duplicates_removed = (
        before_deduplication - len(df)
    )

    print("Exact duplicates removed:", exact_duplicates_removed)
    print("Records after deduplication:", len(df))

    # 7. Check whether non-temperature values agree
    consistent_columns = [
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        TARGET_COLUMN
    ]

    grouped = df.groupby(KEY_COLUMNS)

    conflicts = grouped[consistent_columns].nunique()
    bad_groups = conflicts[(conflicts > 1).any(axis=1)]

    if not bad_groups.empty:
        raise ValueError(
            "Conflicting yield, rainfall or pesticide values "
            "found within Area-Crop-Year groups."
        )

    # 8. Aggregate to one record per Area-Crop-Year
    aggregated = (
        df.groupby(KEY_COLUMNS, as_index=False)
        .agg({
            "average_rain_fall_mm_per_year": "first",
            "pesticides_tonnes": "first",
            "avg_temp": "mean",
            TARGET_COLUMN: "first"
        })
    )

    print("Records after aggregation:", len(aggregated))

    # 9. Chronological train-test split
    train_df = aggregated[
        aggregated["Year"] <= 2008
    ].copy()

    test_df = aggregated[
        aggregated["Year"] >= 2009
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test dataset is empty.")

    # 10. Keep only required features and target
    train_df = train_df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    test_df = test_df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    # 11. Save processed datasets
    train_df.to_csv(
        OUTPUT_DIR / "yieldwise_train.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "yieldwise_test.csv",
        index=False
    )

    # 12. Save preprocessing summary
    summary = pd.DataFrame({
        "Metric": [
            "Original records",
            "Exact duplicates removed",
            "Records after deduplication",
            "Records after aggregation",
            "Training records",
            "Testing records",
            "Training year range",
            "Testing year range"
        ],
        "Value": [
            original_count,
            exact_duplicates_removed,
            len(df),
            len(aggregated),
            len(train_df),
            len(test_df),
            f"{train_df['Year'].min()}-{train_df['Year'].max()}",
            f"{test_df['Year'].min()}-{test_df['Year'].max()}"
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False
    )

    print("\nPreprocessing completed successfully!")
    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))
    print("\nTraining years:",
          train_df["Year"].min(), "-",
          train_df["Year"].max())
    print("Testing years:",
          test_df["Year"].min(), "-",
          test_df["Year"].max())

    print("\nSaved files:")
    print(OUTPUT_DIR / "yieldwise_train.csv")
    print(OUTPUT_DIR / "yieldwise_test.csv")
    print(OUTPUT_DIR / "split_summary.csv")


if __name__ == "__main__":
    main()
