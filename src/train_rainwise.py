
import logging
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# RAINWISE - MODEL TRAINING AND VALIDATION
# AgriWise - Smart Agriculture Prediction System using ML
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "rainwise"
MODEL_DIR = PROJECT_DIR / "models" / "rainwise"

LOG_FILE = PROJECT_DIR / "rainwise_training.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TARGET = "next_day_rain_mm"

CATEGORICAL_FEATURES = ["city"]

NUMERICAL_FEATURES = [
    "previous_day_temperature_2m_max",
    "previous_day_temperature_2m_min",
    "previous_day_apparent_temperature_max",
    "previous_day_apparent_temperature_min",
    "previous_day_wind_speed_10m_max",
    "previous_day_wind_gusts_10m_max",
    "previous_day_wind_direction_10m_dominant",
    "previous_day_rain_sum",
    "target_dayofyear_sin",
    "target_dayofyear_cos"
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_data():

    train_path = DATA_DIR / "rainwise_train.csv"
    validation_path = DATA_DIR / "rainwise_validation.csv"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df)
    ]:

        missing = [
            col for col in FEATURES + [TARGET]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[FEATURES + [TARGET]].isnull().any().any():
            raise ValueError(
                f"Missing values found in {name} dataset."
            )

        if (df[TARGET] < 0).any():
            raise ValueError(
                f"Negative rainfall values found in {name} dataset."
            )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    return X_train, y_train, X_validation, y_validation


def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numeric_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)

    # Rainfall cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    r2 = r2_score(
        y_validation,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def train_models():

    print("=" * 60)
    print("RAINWISE - MODEL TRAINING")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    X_train, y_train, X_validation, y_validation = load_data()

    print("\n1. DATASET INFORMATION")
    print("Training records:", len(X_train))
    print("Validation records:", len(X_validation))
    print("Number of features:", len(FEATURES))

    print("\n2. MODEL FEATURES")
    for feature in FEATURES:
        print("-", feature)

    # --------------------------------------------------------
    # 2. Define candidate models
    # --------------------------------------------------------

    candidates = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=5,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    # --------------------------------------------------------
    # 3. Train and evaluate each model
    # --------------------------------------------------------

    for name, estimator in candidates.items():

        print("\n" + "-" * 60)
        print("Training:", name)
        print("-" * 60)

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", estimator)
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            pipeline,
            X_validation,
            y_validation
        )

        print("MAE:", round(metrics["MAE"], 4))
        print("RMSE:", round(metrics["RMSE"], 4))
        print("R2:", round(metrics["R2"], 4))

        # Save every candidate model
        safe_name = name.lower().replace(" ", "_")

        model_path = MODEL_DIR / f"rainwise_{safe_name}.pkl"

        joblib.dump(
            pipeline,
            model_path
        )

        trained_models[name] = pipeline

        results.append({
            "Model": name,
            **metrics
        })

        print("Saved:", model_path)

    # --------------------------------------------------------
    # 4. Compare validation results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE",
        ascending=True
    ).reset_index(drop=True)

    results_path = MODEL_DIR / "rainwise_model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("3. VALIDATION MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 5. Select best model using validation MAE
    # --------------------------------------------------------

    best_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_name]

    best_model_path = MODEL_DIR / "rainwise_best_model.pkl"

    joblib.dump(
        best_model,
        best_model_path
    )

    # Save model information
    metadata = {
        "project": "AgriWise",
        "module": "RainWise",
        "prediction": "Next-day rainfall in mm",
        "target": TARGET,
        "selected_model": best_name,
        "selection_metric": "Validation MAE",
        "features": FEATURES,
        "random_state": 42
    }

    with open(
        MODEL_DIR / "rainwise_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n4. SELECTED MODEL")
    print("Best model:", best_name)
    print(
        "Validation MAE:",
        round(results_df.iloc[0]["MAE"], 4)
    )

    print("\nSaved best model:", best_model_path)
    print("Saved comparison:", results_path)

    print("\n" + "=" * 60)
    print("RAINWISE MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    try:
        train_models()

    except Exception:

        logging.exception(
            "RainWise model training failed."
        )

        raise
