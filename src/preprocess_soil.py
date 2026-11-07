
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "soil_fertility.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "soilsense"


# 2. Features and target
FEATURES = [
    "N",
    "P",
    "K",
    "pH",
    "EC",
    "OC",
    "S",
    "Zn",
    "Fe",
    "Cu",
    "Mn",
    "B"
]

TARGET = "Output"


# 3. Load dataset
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)

    return df


# 4. Validate dataset
def validate_dataset(df):

    expected_columns = FEATURES + [TARGET]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns or column order.\n"
            f"Expected: {expected_columns}\n"
            f"Received: {list(df.columns)}"
        )

    if df.isnull().values.any():
        raise ValueError("Missing values detected.")

    if df.duplicated().any():
        raise ValueError("Duplicate records detected.")

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not df[TARGET].isin([0, 1, 2]).all():
        raise ValueError("Unexpected target class values.")

    if df[TARGET].nunique() < 2:
        raise ValueError("Insufficient target classes.")

    print("\nDataset validation passed.")
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate records:", df.duplicated().sum())
    print("Predictor features:", len(FEATURES))
    print("Target classes:")
    print(df[TARGET].value_counts().sort_index())


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

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# 7. Save processed datasets
def save_datasets(X_train, X_test, y_train, y_test):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    train_data.to_csv(
        OUTPUT_DIR / "soilsense_train.csv",
        index=False
    )

    test_data.to_csv(
        OUTPUT_DIR / "soilsense_test.csv",
        index=False
    )

    print("\nProcessed files saved.")
    print("Output folder:", OUTPUT_DIR)


# 8. Main function
def main():

    print("===== SoilSense Preprocessing =====")

    df = load_dataset()

    validate_dataset(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    save_datasets(X_train, X_test, y_train, y_test)

    print("\n===== SoilSense Preprocessing Completed =====")


if __name__ == "__main__":
    main()
