import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# FERTILIZERWISE DATA PREPROCESSING
# ==========================================

RAW_PATH = "data/raw/fertilizer_recommendation.csv"

OUTPUT_DIR = "data/processed/fertilizerwise"

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERICAL_FEATURES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "pH",
    "Rainfall",
    "Temperature"
]

CATEGORICAL_FEATURES = [
    "District_Name",
    "Soil_color",
    "Crop"
]

TARGET = "Fertilizer"

REQUIRED_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET]
)


def main():

    print("=" * 55)
    print("FERTILIZERWISE PREPROCESSING")
    print("=" * 55)

    # 1. Check dataset file
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    # 2. Load dataset
    df = pd.read_csv(RAW_PATH)

    print(f"\nOriginal records: {len(df)}")

    # 3. Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Keep only required features and target
    df = df[REQUIRED_COLUMNS].copy()

    # 5. Clean categorical values
    for col in CATEGORICAL_FEATURES + [TARGET]:
        df[col] = df[col].astype("string").str.strip()

    # 6. Validate numerical columns
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 7. Check missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing or invalid values found after cleaning."
        )

    # 8. Validate target classes
    if df[TARGET].nunique() != 19:
        raise ValueError(
            f"Expected 19 fertilizer classes, "
            f"found {df[TARGET].nunique()}."
        )

    # 9. Check minimum class count
    class_counts = df[TARGET].value_counts()

    print("\nFertilizer classes:", len(class_counts))
    print("Smallest class count:", class_counts.min())

    if class_counts.min() < 2:
        raise ValueError(
            "At least 2 records are required per class "
            "for stratified splitting."
        )

    # 10. Separate features and target
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # 11. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 12. Combine features and target
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    test_df = X_test.copy()
    test_df[TARGET] = y_test

    # 13. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 14. Save processed datasets
    train_path = os.path.join(
        OUTPUT_DIR,
        "fertilizerwise_train.csv"
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "fertilizerwise_test.csv"
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 15. Save class distribution
    distribution = pd.DataFrame({
        "Fertilizer": class_counts.index,
        "Total_Count": class_counts.values,
        "Train_Count": train_df[TARGET].value_counts().reindex(
            class_counts.index
        ).values,
        "Test_Count": test_df[TARGET].value_counts().reindex(
            class_counts.index
        ).values
    })

    distribution_path = os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    )

    distribution.to_csv(
        distribution_path,
        index=False
    )

    # 16. Display results
    print("\n===== PREPROCESSING RESULTS =====")

    print(f"Total records: {len(df)}")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    print(f"\nNumerical features: {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Total predictors: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts().sort_index())

    print("\nTesting class distribution:")
    print(test_df[TARGET].value_counts().sort_index())

    print("\nSaved files:")
    print(train_path)
    print(test_path)
    print(distribution_path)

    print("\n===== PREPROCESSING COMPLETED =====")


if __name__ == "__main__":
    main()