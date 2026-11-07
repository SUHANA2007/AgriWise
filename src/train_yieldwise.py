
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

TRAIN_PATH = Path("data/processed/yieldwise/yieldwise_train.csv")
TEST_PATH = Path("data/processed/yieldwise/yieldwise_test.csv")

MODEL_DIR = Path("models/yieldwise")
REPORT_DIR = Path("reports/yieldwise")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

CATEGORICAL_FEATURES = ["Area", "Item"]

NUMERICAL_FEATURES = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET_COLUMN = "hg/ha_yield"


# --------------------------------------------------
# 3. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

def main():

    print("=" * 60)
    print("YIELDWISE - CROP YIELD PREDICTION")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    print(
        "Training years:",
        train_df["Year"].min(),
        "-",
        train_df["Year"].max()
    )

    print(
        "Testing years:",
        test_df["Year"].min(),
        "-",
        test_df["Year"].max()
    )


    # --------------------------------------------------
    # 5. DEFINE REGRESSION MODELS
    # --------------------------------------------------

    models = {
        "Dummy Regressor": DummyRegressor(
            strategy="mean"
        ),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


    # --------------------------------------------------
    # 6. CREATE CHRONOLOGICAL CROSS-VALIDATION FOLDS
    # --------------------------------------------------

    unique_years = np.sort(train_df["Year"].unique())

    time_splitter = TimeSeriesSplit(n_splits=5)

    temporal_splits = []

    for train_year_idx, val_year_idx in time_splitter.split(
        unique_years
    ):

        train_years = unique_years[train_year_idx]
        val_years = unique_years[val_year_idx]

        train_indices = np.flatnonzero(
            train_df["Year"].isin(train_years).to_numpy()
        )

        val_indices = np.flatnonzero(
            train_df["Year"].isin(val_years).to_numpy()
        )

        temporal_splits.append(
            (train_indices, val_indices)
        )

    print("\nTemporal cross-validation folds:", len(temporal_splits))


    # --------------------------------------------------
    # 7. TRAINING-ONLY CROSS-VALIDATION
    # --------------------------------------------------

    results = []
    fitted_models = {}

    print("\nStarting model training and evaluation...")

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = create_pipeline(model)

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=temporal_splits,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            },
            n_jobs=1
        )

        cv_mae = -cv_results["test_mae"]
        cv_rmse = -cv_results["test_rmse"]
        cv_r2 = cv_results["test_r2"]

        # Fit using the complete training dataset
        pipeline.fit(X_train, y_train)

        # Final evaluation on untouched test data
        predictions = pipeline.predict(X_test)

        test_mae = mean_absolute_error(
            y_test,
            predictions
        )

        test_rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        test_r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "CV MAE Mean": cv_mae.mean(),
            "CV MAE Std": cv_mae.std(),
            "CV RMSE Mean": cv_rmse.mean(),
            "CV R2 Mean": cv_r2.mean(),
            "Test MAE": test_mae,
            "Test RMSE": test_rmse,
            "Test R2": test_r2
        })

        fitted_models[model_name] = pipeline

        print("CV MAE:", round(cv_mae.mean(), 2))
        print("Test MAE:", round(test_mae, 2))
        print("Test RMSE:", round(test_rmse, 2))
        print("Test R2:", round(test_r2, 4))


    # --------------------------------------------------
    # 8. MODEL COMPARISON
    # --------------------------------------------------

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="CV MAE Mean",
        ascending=True
    ).reset_index(drop=True)

    comparison_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df.to_string(index=False))


    # --------------------------------------------------
    # 9. SELECT BEST MODEL USING CV MAE
    # --------------------------------------------------

    best_model_name = comparison_df.iloc[0]["Model"]

    best_model = fitted_models[best_model_name]

    print("\nSelected model:", best_model_name)


    # --------------------------------------------------
    # 10. SAVE BEST MODEL
    # --------------------------------------------------

    model_path = MODEL_DIR / "best_yieldwise_model.joblib"

    joblib.dump(best_model, model_path)

    print("\nBest model saved:", model_path)


    # --------------------------------------------------
    # 11. SAVE TEST PREDICTIONS
    # --------------------------------------------------

    best_predictions = best_model.predict(X_test)

    prediction_df = test_df[
        ["Area", "Item", "Year", TARGET_COLUMN]
    ].copy()

    prediction_df["Predicted_Yield"] = best_predictions

    prediction_df["Residual"] = (
        prediction_df[TARGET_COLUMN]
        - prediction_df["Predicted_Yield"]
    )

    prediction_df["Absolute_Error"] = (
        prediction_df["Residual"].abs()
    )

    prediction_df.to_csv(
        REPORT_DIR / "predictions.csv",
        index=False
    )


    # --------------------------------------------------
    # 12. SAVE FINAL METRICS
    # --------------------------------------------------

    final_mae = mean_absolute_error(
        y_test,
        best_predictions
    )

    final_rmse = np.sqrt(
        mean_squared_error(y_test, best_predictions)
    )

    final_r2 = r2_score(
        y_test,
        best_predictions
    )

    metrics_text = f"""
YIELDWISE - FINAL REGRESSION RESULTS

Selected Model: {best_model_name}

Training Records: {len(train_df)}
Testing Records: {len(test_df)}

Training Period: {train_df['Year'].min()} - {train_df['Year'].max()}
Testing Period: {test_df['Year'].min()} - {test_df['Year'].max()}

Mean Absolute Error (MAE): {final_mae:.2f}
Root Mean Squared Error (RMSE): {final_rmse:.2f}
R2 Score: {final_r2:.4f}

Model selection was performed using chronological
cross-validation on the training dataset.

The test dataset was reserved for final evaluation.
"""

    with open(
        REPORT_DIR / "regression_metrics.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(metrics_text)

    print(metrics_text)


    # --------------------------------------------------
    # 13. ACTUAL VS PREDICTED GRAPH
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        best_predictions,
        alpha=0.5
    )

    minimum = min(y_test.min(), best_predictions.min())
    maximum = max(y_test.max(), best_predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Crop Yield")
    plt.ylabel("Predicted Crop Yield")
    plt.title("YieldWise - Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "actual_vs_predicted.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # 14. RESIDUAL PLOT
    # --------------------------------------------------

    residuals = y_test - best_predictions

    plt.figure(figsize=(8, 6))

    plt.scatter(
        best_predictions,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Crop Yield")
    plt.ylabel("Residual Error")
    plt.title("YieldWise - Residual Analysis")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "residual_plot.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # 15. COMPLETION
    # --------------------------------------------------

    print("\nTraining completed!")

    print("\nGenerated files:")
    print("- Model:", model_path)
    print("- Model comparison: reports/yieldwise/model_comparison.csv")
    print("- Metrics: reports/yieldwise/regression_metrics.txt")
    print("- Predictions: reports/yieldwise/predictions.csv")
    print("- Actual vs predicted graph")
    print("- Residual plot")


if __name__ == "__main__":
    main()
