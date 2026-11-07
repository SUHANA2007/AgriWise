
import pandas as pd
from pathlib import Path

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "crop_recommendation.csv"

print("\n========== CROPWISE DATASET AUDIT ==========\n")

# Check whether the file exists
if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("1. DATASET SIZE")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# 3. Data types
print("\n3. DATA TYPES")
print(df.dtypes)

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# 5. Duplicate records
print("\n5. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 6. Basic statistics
print("\n6. NUMERICAL SUMMARY")
print(df.describe())

# 7. Target column
print("\n7. TARGET COLUMN ANALYSIS")
if "label" in df.columns:
    print("Target column: label")
    print("Number of crop classes:", df["label"].nunique())
    print("\nCrop distribution:")
    print(df["label"].value_counts())
else:
    print("WARNING: Expected target column 'label' not found.")

# 8. Dataset requirement check
print("\n8. REQUIREMENT CHECK")
print("500+ records:", "PASS" if len(df) >= 500 else "FAIL")

if "label" in df.columns:
    predictors = [col for col in df.columns if col != "label"]
    print("6+ predictors:", "PASS" if len(predictors) >= 6 else "FAIL")
    print("Target available: PASS")
else:
    print("Target available: FAIL")

print("\n========== AUDIT COMPLETED ==========\n")
