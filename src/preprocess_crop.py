
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "crop_recommendation.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cropwise"


# 2. Features and target
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET = "label"

EXPECTED_COLUMNS = FEATURES + [TARGET]


# 3. Load dataset
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)

    return df


# 4. Validate dataset
def validate_dataset(df):

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match expected schema.")

    if df.isnull().values.any():
        raise ValueError("Missing values detected.")

    if df.duplicated().any():
        raise ValueError("Duplicate records detected.")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not df[TARGET].notna().all():
        raise ValueError("Missing target values detected.")

    if df[TARGET].nunique() < 2:
        raise ValueError("Insufficient target classes.")

    print("\nDataset validation passed.")
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate records:", df.duplicated().sum())
    print("Predictor features:", len(FEATURES))
    print("Target classes:", df[TARGET].nunique())


# 5. Separate features and target
def split_features_target(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


# 6. Split into training and testing data
def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset split completed.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# 7. Save processed datasets
def save_datasets(X_train, X_test, y_train, y_test):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    train_data.to_csv(
        OUTPUT_DIR / "cropwise_train.csv",
        index=False
    )

    test_data.to_csv(
        OUTPUT_DIR / "cropwise_test.csv",
        index=False
    )

    print("\nFiles saved successfully.")
    print("Output folder:", OUTPUT_DIR)


# 8. Main function
def main():

    print("===== CropWise Preprocessing =====")

    df = load_dataset()

    validate_dataset(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    save_datasets(X_train, X_test, y_train, y_test)

    print("\n===== Preprocessing Completed =====")


if __name__ == "__main__":
    main()
