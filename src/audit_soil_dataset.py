
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "soil_fertility.csv"

df = pd.read_csv(file_path)

print("=" * 55)
print("SOILSENSE - FINAL DATASET AUDIT")
print("=" * 55)

target = "Output"

# 1. Target class distribution
print("\n1. FERTILITY CLASS DISTRIBUTION")
print(df[target].value_counts().sort_index())

print("\nNumber of classes:", df[target].nunique())

# 2. Numerical feature ranges
numeric_columns = [
    col for col in df.select_dtypes(include="number").columns
    if col != target
]

print("\n2. NUMERICAL FEATURE RANGES")
print(df[numeric_columns].describe().round(2))

# 3. Possible outliers using IQR
print("\n3. POSSIBLE OUTLIERS (IQR METHOD)")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[column] < lower) | (df[column] > upper)).sum()

    print(f"{column}: {count} possible outliers")

# 4. Class imbalance analysis
class_counts = df[target].value_counts().sort_index()

print("\n4. CLASS BALANCE")
print("Largest class:", class_counts.max())
print("Smallest class:", class_counts.min())

print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

# 5. Final checks
print("\n5. FINAL CHECKS")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Target column:", target)
print("Predictor count:", len(numeric_columns))

print("\nSoilSense audit completed.")
