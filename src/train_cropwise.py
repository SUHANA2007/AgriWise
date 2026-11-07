
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "cropwise" / "cropwise_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "cropwise" / "cropwise_test.csv"

MODEL_DIR = ROOT / "models" / "cropwise"
REPORT_DIR = ROOT / "reports" / "cropwise"

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

TARGET = "label"

RANDOM_STATE = 42


# 2. Load data
def load_data():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of crop classes:", y_train.nunique())

    return X_train, X_test, y_train, y_test


# 3. Define candidate models
def get_models():

    return {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# 4. Cross-validation and model training
def train_models(X_train, X_test, y_train, y_test):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro"
    }

    results = []
    trained_models = {}

    for name, model in get_models().items():

        print(f"\nEvaluating {name} using 5-fold CV...")

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        mean_f1 = cv_results["test_f1"].mean()
        std_f1 = cv_results["test_f1"].std()

        print("Mean CV Macro F1:", round(mean_f1, 4))
        print("CV Macro F1 Std:", round(std_f1, 4))

        # Fit model using training data only
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        # Test data is used only for final evaluation
        predictions = fitted_model.predict(X_test)

        results.append({
            "Model": name,
            "CV Accuracy": cv_results["test_accuracy"].mean(),
            "CV Precision": cv_results["test_precision"].mean(),
            "CV Recall": cv_results["test_recall"].mean(),
            "CV Macro F1": mean_f1,
            "CV Macro F1 Std": std_f1,
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Test Precision": precision_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ),
            "Test Recall": recall_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ),
            "Test Macro F1": f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            )
        })

        trained_models[name] = fitted_model

    return pd.DataFrame(results), trained_models


# 5. Save model and evaluation reports
def save_results(results, trained_models, X_test, y_test):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Select model using CV Macro F1, NOT test performance
    results = results.sort_values(
        by=["CV Macro F1", "CV Accuracy"],
        ascending=False
    ).reset_index(drop=True)

    results.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n===== MODEL COMPARISON =====")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_cropwise_model.joblib"
    )

    print("\nSelected model:", best_name)
    print("Selection basis: Mean 5-fold CV Macro F1")

    # Final test evaluation
    predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Confusion matrix
    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    matrix_df.to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    plt.figure(figsize=(14, 11))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"CropWise Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Crop")
    plt.ylabel("Actual Crop")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nFinal Test Accuracy:",
          round(accuracy_score(y_test, predictions), 4))

    print("Final Test Macro F1:",
          round(f1_score(
              y_test, predictions,
              average="macro",
              zero_division=0
          ), 4))

    print("\nModel and evaluation reports saved.")


# 6. Main function
def main():

    print("===== CropWise Model Training =====")

    X_train, X_test, y_train, y_test = load_data()

    results, trained_models = train_models(
        X_train, X_test, y_train, y_test
    )

    save_results(
        results, trained_models, X_test, y_test
    )

    print("\n===== Training Completed =====")


if __name__ == "__main__":
    main()
