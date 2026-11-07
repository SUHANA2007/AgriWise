
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "fertilizer_recommendation.csv"

df = pd.read_csv(file_path)

print("=" * 55)
print("FERTILIZERWISE - FINAL DATASET AUDIT")
print("=" * 55)

target = "Fertilizer"

# 1. Fertilizer class distribution
print("\n1. FERTILIZER CLASS DISTRIBUTION")
class_counts = df[target].value_counts()
print(class_counts)

print("\nNumber of fertilizer classes:", df[target].nunique())

# 2. Class percentages
print("\n2. CLASS PERCENTAGES")
print((class_counts / len(df) * 100).round(2))

# 3. Numerical feature ranges
numeric_columns = [
    "Nitrogen", "Phosphorus", "Potassium",
    "pH", "Rainfall", "Temperature"
]

print("\n3. NUMERICAL FEATURE RANGES")
print(df[numeric_columns].describe().round(2))

# 4. Possible outliers using IQR
print("\n4. POSSIBLE OUTLIERS (IQR METHOD)")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[column] < lower) | (df[column] > upper)).sum()

    print(f"{column}: {count} possible outliers")

# 5. Categorical feature quality
print("\n5. CATEGORICAL FEATURE CHECK")

for column in ["District_Name", "Soil_color", "Crop"]:
    cleaned = df[column].astype(str).str.strip()

    print(f"\n{column}")
    print("Unique values before cleaning:", df[column].nunique())
    print("Unique values after trimming spaces:", cleaned.nunique())
    print("Values:", sorted(cleaned.unique()))

# 6. Class imbalance
print("\n6. CLASS BALANCE")
print("Largest class:", class_counts.max())
print("Smallest class:", class_counts.min())

# 7. Final checks
print("\n7. FINAL CHECKS")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())
print("Target column:", target)
print("Numerical predictors:", len(numeric_columns))

print("\nFertilizerWise audit completed.")
