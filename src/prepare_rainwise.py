
from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# RAINWISE - NEXT-DAY RAINFALL DATA PREPARATION
# AgriWise - Smart Agriculture Prediction System using ML
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "data" / "raw" / "rainwise" / "rainwise.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "rainwise"

LOG_FILE = PROJECT_DIR / "rainwise_preparation.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Required original columns
REQUIRED_COLUMNS = [
    "city",
    "date",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant"
]

# Weather observations available from the previous day
WEATHER_FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "rain_sum"
]

# Final model input columns
FEATURE_COLUMNS = (
    ["city"]
    + [f"previous_day_{col}" for col in WEATHER_FEATURES]
    + ["target_dayofyear_sin", "target_dayofyear_cos"]
)

TARGET_COLUMN = "next_day_rain_mm"


def prepare_dataset():

    print("=" * 60)
    print("RAINWISE - DATA PREPARATION")
    print("=" * 60)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(RAW_FILE)

    print("\n1. ORIGINAL DATASET")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # --------------------------------------------------------
    # 2. Validate and clean basic data types
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        col for col in WEATHER_FEATURES
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["city"] = df["city"].astype("string").str.strip()

    if df.isnull().any().any():
        raise ValueError(
            "Missing or invalid values found. "
            "Inspect the raw dataset before proceeding."
        )

    if (df["city"] == "").any():
        raise ValueError("Empty city names found.")

    if df["date"].duplicated().any():
        print(
            "Note: Dates can repeat across different cities. "
            "City-date uniqueness will be checked separately."
        )

    if df.duplicated(subset=["city", "date"]).any():
        raise ValueError(
            "Duplicate city-date records found."
        )

    if (df["rain_sum"] < 0).any():
        raise ValueError("Negative rainfall values found.")

    # --------------------------------------------------------
    # 3. Sort records chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["city", "date"]
    ).reset_index(drop=True)

    # Ensure each city's observations are consecutive
    date_gaps = (
        df.groupby("city")["date"]
        .diff()
        .dt.days
    )

    if (date_gaps.dropna() != 1).any():
        raise ValueError(
            "Non-consecutive dates found. "
            "Review missing or irregular dates before training."
        )

    print("\n2. DATA VALIDATION")
    print("Missing values: 0")
    print("Duplicate city-date records: 0")
    print("Negative rainfall values: 0")
    print("Date continuity: Passed")

    # --------------------------------------------------------
    # 4. Create next-day target
    # --------------------------------------------------------

    df["target_date"] = (
        df.groupby("city")["date"].shift(-1)
    )

    df[TARGET_COLUMN] = (
        df.groupby("city")["rain_sum"].shift(-1)
    )

    # The final observation for each city has no next day
    df = df.dropna(
        subset=["target_date", TARGET_COLUMN]
    ).copy()

    # --------------------------------------------------------
    # 5. Create seasonal date features
    # --------------------------------------------------------

    day_of_year = df["target_date"].dt.dayofyear

    days_in_year = np.where(
        df["target_date"].dt.is_leap_year,
        366,
        365
    )

    df["target_dayofyear_sin"] = np.sin(
        2 * np.pi * day_of_year / days_in_year
    )

    df["target_dayofyear_cos"] = np.cos(
        2 * np.pi * day_of_year / days_in_year
    )

    # --------------------------------------------------------
    # 6. Rename previous-day weather features
    # --------------------------------------------------------

    rename_columns = {
        col: f"previous_day_{col}"
        for col in WEATHER_FEATURES
    }

    df = df.rename(columns=rename_columns)

    # Keep metadata separately from model inputs
    metadata_columns = ["date", "target_date"]

    processed = df[
        metadata_columns
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    # --------------------------------------------------------
    # 7. Validate final processed dataset
    # --------------------------------------------------------

    if processed.isnull().any().any():
        raise ValueError(
            "Missing values found in processed dataset."
        )

    if (processed[TARGET_COLUMN] < 0).any():
        raise ValueError(
            "Negative target values found."
        )

    expected_rows = 91320 - 10

    if len(processed) != expected_rows:
        raise ValueError(
            f"Unexpected processed row count: {len(processed)}. "
            f"Expected {expected_rows}."
        )

    # --------------------------------------------------------
    # 8. Chronological train-validation-test split
    # --------------------------------------------------------

    train = processed[
        processed["target_date"].dt.year <= 2020
    ].copy()

    validation = processed[
        processed["target_date"].dt.year.between(2021, 2022)
    ].copy()

    test = processed[
        processed["target_date"].dt.year.between(2023, 2024)
    ].copy()

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError(
            "One or more chronological datasets are empty."
        )

    # Check that all records are assigned
    if len(train) + len(validation) + len(test) != len(processed):
        raise ValueError(
            "Train, validation and test split counts do not match."
        )

    # --------------------------------------------------------
    # 9. Save datasets
    # --------------------------------------------------------

    processed.to_csv(
        OUTPUT_DIR / "rainwise_processed.csv",
        index=False
    )

    train.to_csv(
        OUTPUT_DIR / "rainwise_train.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "rainwise_validation.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "rainwise_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # 10. Display final report
    # --------------------------------------------------------

    print("\n3. PROCESSED DATASET")
    print("Total rows:", len(processed))
    print("Total columns:", len(processed.columns))
    print("Model predictors:", len(FEATURE_COLUMNS))

    print("\n4. DATASET SPLIT")
    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Testing records:", len(test))

    print("\n5. TARGET INFORMATION")
    print("Target:", TARGET_COLUMN)
    print("Minimum rainfall:", processed[TARGET_COLUMN].min())
    print("Maximum rainfall:", processed[TARGET_COLUMN].max())
    print("Mean rainfall:", round(processed[TARGET_COLUMN].mean(), 3))

    print("\n6. SAVED FILES")

    for filename in [
        "rainwise_processed.csv",
        "rainwise_train.csv",
        "rainwise_validation.csv",
        "rainwise_test.csv"
    ]:
        print("Saved:", OUTPUT_DIR / filename)

    print("\n" + "=" * 60)
    print("RAINWISE DATA PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    try:
        prepare_dataset()

    except Exception:
        logging.exception(
            "RainWise data preparation failed."
        )
        raise
