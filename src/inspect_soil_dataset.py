
import pandas as pd
from pathlib import Path

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "soil_fertility.csv"

print("\n========== SOILSENSE DATASET AUDIT ==========\n")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# 1. Dataset size
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

# 6. Numerical summary
print("\n6. NUMERICAL SUMMARY")
print(df.describe())

# 7. Categorical columns and possible targets
print("\n7. CATEGORICAL COLUMN ANALYSIS")

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    print(f"\nColumn: {column}")
    print("Unique values:", df[column].nunique())
    print(df[column].value_counts(dropna=False))

# 8. Requirement check
print("\n8. INITIAL REQUIREMENT CHECK")

target_column = "Output"

predictor_columns = [
    col for col in df.columns if col != target_column
]

print("500+ records:",
      "PASS" if len(df) >= 500 else "FAIL")

print("6+ predictors:",
      "PASS" if len(predictor_columns) >= 6 else "FAIL")

print("Target column available:",
      "PASS" if target_column in df.columns else "FAIL")

print("Target column:", target_column)
print("Number of predictors:", len(predictor_columns))
print("Predictor columns:", predictor_columns)

print("\n9. OUTPUT CLASS DISTRIBUTION")

print("Unique Output values:")
print(df["Output"].unique())

print("\nNumber of records in each class:")
print(df["Output"].value_counts().sort_index())

print("\n========== AUDIT COMPLETED ==========\n")