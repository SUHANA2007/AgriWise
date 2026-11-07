# AgriWise – Dataset Documentation

## 1. Project Information

**Project Title:** AgriWise – Smart Agriculture Prediction System using ML  
**Project Type:** Software-only Machine Learning Web Application  
**Platform:** Python and Streamlit  
**Total Datasets:** 4 independent datasets  
**Total Original Records:** 35,835

### Project Overview

AgriWise is a machine learning-based agricultural prediction system consisting of four independent modules. Each module uses its own dataset, preprocessing workflow and machine learning model.

The four modules are:

1. CropWise – Crop Recommendation
2. SoilSense – Soil Fertility Analysis
3. FertilizerWise – Fertilizer Recommendation
4. YieldWise – Crop Yield Prediction

The datasets are not merged. Each module performs a separate prediction task.

---

## 2. CropWise – Crop Recommendation

### 2.1 Dataset Information

**Dataset Name:** Crop Recommendation Dataset  
**Source:** Kaggle – Atharva Ingle  
**Dataset URL:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset  
**Local File:** `data/raw/crop_recommendation.csv`  
**Original Records:** 2,200  
**Columns:** 8  
**Predictor Features:** 7  
**Target Variable:** `label`

### 2.2 Data Dictionary

| Feature | Description | Data Type |
|---|---|---|
| N | Nitrogen content | Numerical |
| P | Phosphorus content | Numerical |
| K | Potassium content | Numerical |
| temperature | Temperature | Numerical |
| humidity | Relative humidity | Numerical |
| ph | Soil pH value | Numerical |
| rainfall | Rainfall | Numerical |
| label | Recommended crop | Target |

Measurement units should be interpreted according to the original dataset documentation.

### 2.3 Machine Learning Task

**Task Type:** Multi-class Classification  
**Purpose:** Predicts a suitable crop based on soil nutrient values and environmental conditions.  
**Target Classes:** 22 crop categories.

### 2.4 Dataset Quality Audit

- Original records: 2,200
- Missing values: 0
- Duplicate records: 0
- Crop classes: 22
- Records per class: 100
- Class distribution: Balanced
- Numerical predictors: 7

Feature ranges identified during the audit:

| Feature | Minimum | Maximum |
|---|---:|---:|
| N | 0 | 140 |
| P | 5 | 145 |
| K | 5 | 205 |
| temperature | 8.83 | 43.68 |
| humidity | 14.26 | 99.98 |
| ph | 3.50 | 9.94 |
| rainfall | 20.21 | 298.56 |

Potential outliers were identified using the IQR method. These values are not automatically considered errors and were not automatically removed.

### 2.5 Dataset Limitations

- The dataset contains a relatively limited number of records.
- It may not represent every geographical region or agricultural condition.
- The dataset description indicates that agricultural and environmental data sources were used; the original data-generation methodology requires further documentation.
- Predictions depend on patterns represented in the available dataset.
- Model performance on real-world agricultural conditions may differ from performance on the dataset.
- The dataset is not specifically validated for Tamil Nadu.

### 2.6 Dataset Licence

**Licence:** Apache License 2.0  
**Licence URL:** https://www.apache.org/licenses/LICENSE-2.0  
**Verification:** The Kaggle dataset page displays Apache 2.0.

### 2.7 Provenance Details

- Download date: Not recorded.
- Dataset version: To be verified.
- Original data provenance: Further verification required.

---

## 3. SoilSense – Soil Fertility Analysis

### 3.1 Dataset Information

**Dataset Name:** Soil Fertility Dataset  
**Source:** Kaggle – Rahul Jaiswal  
**Dataset URL:** https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset  
**Local File:** `data/raw/soil_fertility.csv`  
**Original Records:** 880  
**Columns:** 13  
**Predictor Features:** 12  
**Target Variable:** `Output`

### 3.2 Data Dictionary

| Feature | Description | Data Type |
|---|---|---|
| N | Nitrogen content | Numerical |
| P | Phosphorus content | Numerical |
| K | Potassium content | Numerical |
| pH | Soil pH value | Numerical |
| EC | Electrical Conductivity | Numerical |
| OC | Organic Carbon | Numerical |
| S | Sulphur content | Numerical |
| Zn | Zinc content | Numerical |
| Fe | Iron content | Numerical |
| Cu | Copper content | Numerical |
| Mn | Manganese content | Numerical |
| B | Boron content | Numerical |
| Output | Soil fertility category | Target |

Measurement units require confirmation from the original source.

### 3.3 Machine Learning Task

**Task Type:** Multi-class Classification  
**Purpose:** Classifies soil fertility based on available soil nutrient and chemical measurements.  
**Target Classes:** 3 numerical categories.

The meanings of target classes 0, 1 and 2 have not yet been verified from the original source. They should therefore remain numerical classes in the application documentation.

### 3.4 Dataset Quality Audit

- Original records: 880
- Missing values: 0
- Duplicate records: 0
- Numerical predictors: 12
- Target classes: 3
- Class imbalance: Present

**Target Class Distribution:**

| Target Class | Records | Percentage |
|---|---:|---:|
| Class 0 | 401 | 45.57% |
| Class 1 | 440 | 50.00% |
| Class 2 | 39 | 4.43% |
| **Total** | **880** | **100%** |

Class 2 represents only 4.43% of the dataset.

The test set contains only 8 records from Class 2, of which 5 were misclassified. Therefore, performance for this class is not sufficiently reliable to support strong conclusions.

Potential extreme values were identified during IQR analysis. These require source and measurement-unit verification before deciding whether any preprocessing is necessary.

### 3.5 Dataset Limitations

- The dataset contains fewer records than the other datasets.
- One target class has significantly fewer records.
- Class imbalance may affect classification performance.
- Measurement units and extreme values require source verification.
- The model may perform differently on soil samples outside the dataset distribution.
- Geographical coverage and collection methodology require further verification.

### 3.6 Dataset Licence

**Licence:** Unknown  
**Verification:** The Kaggle dataset page displays Unknown.

Reuse and redistribution permissions must be treated cautiously until the applicable licence or permission is established.

### 3.7 Provenance Details

- Download date: Not recorded.
- Dataset version: To be verified.
- Original data provenance: Further verification required.
- Measurement units: To be confirmed.
- Target class meanings: To be confirmed.

---

## 4. FertilizerWise – Fertilizer Recommendation

### 4.1 Dataset Information

**Dataset Name:** Crop and Fertilizer Dataset for Western Maharashtra  
**Source:** Kaggle – Sanchita Gholap  
**Dataset URL:** https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra  
**Local File:** `data/raw/fertilizer_recommendation.csv`  
**Original Records:** 4,513  
**Columns:** 11  
**Predictor Features:** 9  
**Target Variable:** `Fertilizer`

### 4.2 Data Dictionary

| Feature | Description | Data Type |
|---|---|---|
| District_Name | District name | Categorical |
| Soil_color | Soil colour | Categorical |
| Nitrogen | Nitrogen content | Numerical |
| Phosphorus | Phosphorus content | Numerical |
| Potassium | Potassium content | Numerical |
| pH | Soil pH value | Numerical |
| Rainfall | Rainfall | Numerical |
| Temperature | Temperature | Numerical |
| Crop | Crop type | Categorical |
| Fertilizer | Recommended fertilizer | Target |
| Link | Reference/link field | Excluded |

The `Link` column is excluded from predictive modelling.

### 4.3 Machine Learning Task

**Task Type:** Multi-class Classification  
**Purpose:** Recommends a fertilizer based on soil characteristics, crop type and environmental conditions.  
**Target Classes:** 19 fertilizer categories.

### 4.4 Dataset Quality Audit

- Original records: 4,513
- Missing values: 0
- Duplicate records: 0
- Numerical predictors: 6
- Categorical predictors: 3
- Total predictors: 9
- Fertilizer classes: 19
- Class imbalance: Present
- Geographic coverage: 5 districts in Western Maharashtra

**Geographical Coverage:**

- Kolhapur
- Satara
- Sangli
- Solapur
- Pune

**Class Distribution Observation:**

Urea is the largest category, with 1,364 records (approximately 30.22%).

Some categories contain very few records. The smallest categories, including 18:46:00 NPK and Sulphur, contain only 6 records each.

This affects the reliability of predictions for rare categories.

**Categorical Data Observation:**

The `Soil_color` column contained whitespace inconsistencies. The number of unique values changed from 7 to 6 after trimming whitespace during preprocessing.

**Numerical Feature Ranges:**

| Feature | Minimum | Maximum |
|---|---:|---:|
| Nitrogen | 20 | 150 |
| Phosphorus | 10 | 90 |
| Potassium | 5 | 150 |
| pH | 5.5 | 8.5 |
| Rainfall | 300 | 1700 |
| Temperature | 10 | 40 |

Potential extreme values were identified through IQR analysis. They should not automatically be removed.

### 4.5 Dataset Limitations

- The dataset represents a limited geographical region in Western Maharashtra.
- It covers only five districts.
- Some fertilizer categories have very few records.
- Predictions may not generalize to other geographical regions.
- The source and data collection methodology require further verification.
- Recommendations should be interpreted within the geographical and data limitations of the dataset.
- The dataset is not specifically validated for Tamil Nadu.

### 4.6 Dataset Licence

**Licence:** CC0 1.0 Universal (No Copyright)  
**Licence URL:** https://creativecommons.org/publicdomain/zero/1.0/  
**Verification:** The Kaggle dataset page displays CC0 / No Copyright.

The original dataset source and creator will still be acknowledged in project documentation.

### 4.7 Provenance Details

- Download date: Not recorded.
- Dataset version: To be verified.
- Original data provenance: Further verification required.
- Geographic coverage: Western Maharashtra – five districts.

---

## 5. YieldWise – Crop Yield Prediction

### 5.1 Dataset Information

**Dataset Name:** Crop Yield Prediction Dataset  
**Source:** Kaggle – Patelris  
**Dataset URL:** https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset  
**Local File:** `data/raw/yield_df.csv`  
**Original Records:** 28,242  
**Columns:** 8  
**Predictor Features:** 6  
**Target Variable:** `hg/ha_yield`

### 5.2 Data Dictionary

| Feature | Description | Data Type |
|---|---|---|
| Area | Geographical area/country | Categorical |
| Item | Crop type | Categorical |
| Year | Agricultural year | Numerical |
| average_rain_fall_mm_per_year | Average annual rainfall | Numerical |
| pesticides_tonnes | Pesticides used in tonnes | Numerical |
| avg_temp | Average temperature | Numerical |
| hg/ha_yield | Crop yield | Target |
| Unnamed: 0 | Index-like column | Excluded |

The `Unnamed: 0` column is excluded from predictive modelling.

### 5.3 Machine Learning Task

**Task Type:** Regression  
**Purpose:** Predicts crop yield based on geographical, historical and environmental information.  
**Target Variable:** `hg/ha_yield`.

### 5.4 Dataset Quality Audit

- Original records: 28,242
- Missing values: 0
- Duplicate rows in the original audit: 0
- Predictor features: 6
- Geographical coverage: 101 areas
- Crop types: 10
- Historical period: 1990–2013
- Year 2003: Absent

**Numerical Feature Summary:**

| Feature | Minimum | Maximum |
|---|---:|---:|
| Year | 1990 | 2013 |
| Average rainfall | 51 | 3240 |
| Pesticides | 0.04 | 367778 |
| Average temperature | 1.30 | 30.65 |
| Crop yield | 50 | 501412 |

**Target Variable Observations:**

- Minimum yield: 50 hg/ha
- Maximum yield: 501,412 hg/ha
- Mean yield: Approximately 77,053.33 hg/ha
- Median yield: 38,295 hg/ha
- Standard deviation: 84,956.61
- Skewness: 1.79

The target yield distribution is positively skewed.

The original audit identified 2,059 possible target outlier records using the IQR method. These values should be investigated within crop groups rather than automatically removed, because yield ranges can differ considerably between crops.

### 5.5 Preprocessing and Duplicate Handling

The original dataset includes an index-like column named `Unnamed: 0`.

The original audit reported zero duplicate rows when all eight columns were considered. However, after excluding the index column, 2,310 exact duplicate records were identified.

Repeated records were also found for the same Area–Item–Year combination.

To prepare the dataset:

1. The index-like column was excluded.
2. Exact duplicate records were removed.
3. Repeated Area–Item–Year records were aggregated into one record per combination.
4. Average temperature was averaged for repeated combinations.
5. Yield, rainfall and pesticide values were preserved because no conflicting values were found for those fields within the examined combinations.

This preprocessing produced **13,130 records**.

The aggregation is a modelling decision made to prevent repeated combinations from receiving disproportionate weight. It does not establish that the averaged temperature is an official annual measurement.

### 5.6 Train-Test Preparation

A chronological split was used to respect the historical order of the data.

| Dataset | Year Range | Records |
|---|---|---:|
| Training | 1990–2008 | 10,193 |
| Testing | 2009–2013 | 2,937 |
| **Total** | | **13,130** |

The processed dataset contains no missing values or duplicate Area–Item–Year keys. No key overlap was found between the training and testing sets.

### 5.7 Dataset Limitations

- The dataset contains historical agricultural data.
- It does not necessarily represent current farming conditions.
- Yield values vary considerably between crop types.
- Geographic and time-based patterns may influence model evaluation.
- Random splitting alone may not adequately measure performance on future years or unseen geographical areas.
- Predictions beyond the training period are not validated forecasts.
- The original source provenance and exact applicable licence terms require further verification.

### 5.8 Dataset Licence

**Licence Reference:** World Bank Dataset Terms of Use  
**Terms URL:** https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets

The World Bank's general dataset terms provide CC BY 4.0 as a default, subject to dataset-specific conditions and third-party exceptions.

**Verification Status:** The exact original data provenance and applicability of these terms to the Kaggle-hosted dataset require confirmation.

The project must not assume that every file in the Kaggle dataset is automatically covered by CC BY 4.0 without verifying its original source and applicable metadata.

### 5.9 Provenance Details

- Download date: Not recorded.
- Dataset version: To be verified.
- Original data provenance: Further verification required.
- Historical coverage: 1990–2013.

---

## 6. Overall Dataset Summary

| Module | Original Records | Columns | Predictors | ML Task |
|---|---:|---:|---:|---|
| CropWise | 2,200 | 8 | 7 | Classification |
| SoilSense | 880 | 13 | 12 | Classification |
| FertilizerWise | 4,513 | 11 | 9 | Classification |
| YieldWise | 28,242 | 8 | 6 | Regression |
| **Total** | **35,835** | **40** | | |

Each module uses an independent dataset and a separate machine learning pipeline.

**YieldWise processed records:** 13,130 after duplicate handling and aggregation.

---

## 7. Overall Data Quality Observations

The four datasets were inspected and audited individually.

The following observations were recorded:

1. No missing values were detected in the original datasets.
2. CropWise contains balanced target classes.
3. SoilSense contains significant class imbalance, especially in Class 2.
4. FertilizerWise contains imbalanced fertilizer categories and categorical whitespace inconsistencies.
5. YieldWise contains a positively skewed target distribution.
6. Potential outlier values were identified in multiple datasets through IQR analysis.
7. Some datasets require further verification of measurement units and original data provenance.
8. YieldWise required additional duplicate handling after excluding its index-like column.

**Important:** IQR-based outlier flags do not automatically indicate incorrect data. Values must be investigated using the original dataset context before removal or transformation.

---

## 8. Dataset Licence Summary

| Module | Licence | Current Status |
|---|---|---|
| CropWise | Apache License 2.0 | Displayed on Kaggle page |
| SoilSense | Unknown | Kaggle page displays Unknown |
| FertilizerWise | CC0 1.0 Universal | Displayed on Kaggle page |
| YieldWise | World Bank Dataset Terms of Use | Dataset-specific applicability requires confirmation |

All dataset sources and creators will be acknowledged in the project documentation.

---

## 9. Documentation Pending

The following activities remain:

1. Record actual download dates, if available.
2. Verify original dataset authors and data collection methodology.
3. Confirm measurement units wherever available.
4. Record dataset versions or file hashes where possible.
5. Verify the original meaning of SoilSense target classes.
6. Confirm the exact applicable terms and original data provenance for YieldWise.
7. Investigate unusual values identified during the audits.
8. Verify reuse permissions for SoilSense before redistribution.

---

## 10. Important Project Notes

- Each AgriWise module uses an independent dataset.
- The datasets are not combined or merged.
- Each module has its own preprocessing workflow and machine learning model.
- Classification modules are evaluated using classification metrics.
- YieldWise is evaluated using regression metrics and residual/error analysis.
- Data transformations and preprocessing are fitted using training data only to avoid data leakage.
- Dataset limitations are considered when interpreting model results.
- Predictions are presented as data-driven estimates and not as guaranteed agricultural advice.

---

## 11. Current Development Status

| Activity | Status |
|---|---|
| Dataset collection | Completed |
| Initial dataset inspection | Completed |
| Detailed dataset quality audits | Completed |
| Initial dataset documentation | Prepared; corrections in progress |
| Licence checks | Checked; some terms require further confirmation |
| Download dates | Pending |
| Provenance verification | Pending |
| Data dictionaries | Prepared; source verification pending |
| Data preprocessing | Completed for all four modules |
| Baseline model development | Completed |
| Predictive model training | Completed |
| Model comparison and evaluation | Completed |
| Error analysis | Completed for all four modules |
| New-record prediction scripts | Completed |
| Streamlit web application | Implemented |
| Model integration | Implemented |
| Website reference testing | Completed |
| Input validation testing | Completed |
| Navigation and stability testing | Completed |

### Final Note

This document records the dataset information, data quality audits, preprocessing decisions, limitations and current development status of AgriWise.

The project uses four independent machine learning workflows integrated into one Streamlit web application. Dataset limitations and unresolved provenance details must be considered when interpreting predictions and presenting the project.
