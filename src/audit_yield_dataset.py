
import pandas as pd

# Load YieldWise dataset
file_path = "data/raw/yield_df.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("YIELDWISE - FINAL DATASET AUDIT")
print("=" * 60)

# 1. Dataset size
print("\n1. DATASET INFORMATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

# 2. Missing and duplicate values
print("\n2. DATA QUALITY")
print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

# 3. Unique values
print("\n3. CATEGORY INFORMATION")
for column in ["Area", "Item", "Year"]:
    print(f"\n{column}: {df[column].nunique()} unique values")
    if column != "Area":
        print(sorted(df[column].dropna().unique()))

# 4. Numerical feature ranges
print("\n4. NUMERICAL FEATURE SUMMARY")
numeric_columns = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]
print(df[numeric_columns].describe().round(2))

# 5. Yield distribution
print("\n5. YIELD DISTRIBUTION")
yield_data = df["hg/ha_yield"]

print("Minimum:", yield_data.min())
print("Maximum:", yield_data.max())
print("Mean:", round(yield_data.mean(), 2))
print("Median:", round(yield_data.median(), 2))
print("Standard deviation:", round(yield_data.std(), 2))
print("Skewness:", round(yield_data.skew(), 2))

# 6. Possible target outliers (IQR method)
q1 = yield_data.quantile(0.25)
q3 = yield_data.quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outliers = yield_data[
    (yield_data < lower_limit) |
    (yield_data > upper_limit)
]

print("\n6. POSSIBLE YIELD OUTLIERS")
print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))
print("Possible outlier records:", len(outliers))

# 7. Records and average yield by crop
print("\n7. CROP-WISE ANALYSIS")
crop_summary = df.groupby("Item")["hg/ha_yield"].agg(
    ["count", "mean", "min", "max"]
).round(2)

print(crop_summary)

# 8. Area-wise record counts
print("\n8. AREA-WISE RECORD COUNTS")
print(df["Area"].value_counts().head(10))

# 9. Predictor selection
print("\n9. SELECTED PREDICTORS")
print([
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
])

print("\nTarget: hg/ha_yield")
print("Note: Unnamed: 0 is an index column and must be excluded.")

print("\nYIELDWISE AUDIT COMPLETED")
