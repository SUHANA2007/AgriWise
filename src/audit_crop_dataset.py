
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "crop_recommendation.csv"

df = pd.read_csv(file_path)

print("=" * 55)
print("CROPWISE - FINAL DATASET AUDIT")
print("=" * 55)

# 1. Target class distribution
print("\n1. CROP CLASS DISTRIBUTION")
print(df["label"].value_counts().sort_index())

print("\nNumber of crop classes:", df["label"].nunique())

# 2. Numerical feature ranges
numeric_columns = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

print("\n2. NUMERICAL FEATURE RANGES")
print(df[numeric_columns].describe().round(2))

# 3. Check unusual values using IQR
print("\n3. POSSIBLE OUTLIERS (IQR METHOD)")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[column] < lower) | (df[column] > upper)).sum()

    print(f"{column}: {count} possible outliers")

# 4. Check class balance
class_counts = df["label"].value_counts()

print("\n4. CLASS BALANCE")
print("Largest class:", class_counts.max())
print("Smallest class:", class_counts.min())

if class_counts.max() == class_counts.min():
    print("Result: Perfectly balanced")
else:
    print("Result: Classes have different record counts")

# 5. Final checks
print("\n5. FINAL CHECKS")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Target column:", "label")
print("Predictor count:", len(numeric_columns))

print("\nCropWise audit completed.")
