
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# ======================================================
# 1. PROJECT PATHS
# ======================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "soilsense" / "soilsense_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "soilsense" / "soilsense_test.csv"

MODEL_DIR = ROOT / "models" / "soilsense"
REPORT_DIR = ROOT / "reports" / "soilsense"

FEATURES = [
    "N", "P", "K", "pH", "EC", "OC",
    "S", "Zn", "Fe", "Cu", "Mn", "B"
]

TARGET = "Output"
RANDOM_STATE = 42

# ======================================================
# 2. LOAD DATA
# ======================================================

def load_data():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# ======================================================
# 3. DEFINE MODELS
# ======================================================

def get_models():

    return {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )
    }


# ======================================================
# 4. CROSS-VALIDATION ON TRAINING DATA ONLY
# ======================================================

def train_models(X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    trained_models = {}

    for name, model in get_models().items():

        print(f"\nCross-validating {name}...")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro"
            },
            n_jobs=1
        )

        mean_f1 = scores["test_macro_f1"].mean()
        std_f1 = scores["test_macro_f1"].std()

        mean_accuracy = scores["test_accuracy"].mean()

        results.append({
            "Model": name,
            "CV Accuracy": mean_accuracy,
            "CV Macro F1": mean_f1,
            "CV Macro F1 Std": std_f1
        })

        trained_models[name] = model

        print("CV Accuracy:", round(mean_accuracy, 4))
        print("CV Macro F1:", round(mean_f1, 4))
        print("CV F1 Std:", round(std_f1, 4))

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV Macro F1",
        ascending=False
    ).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    print("\nSelected model using CV Macro F1:", best_name)

    # Train selected model on the complete training dataset
    best_model.fit(X_train, y_train)

    return results_df, best_name, best_model


# ======================================================
# 5. FINAL TEST EVALUATION
# ======================================================

def evaluate_model(best_model, best_name, X_test, y_test):

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n===== FINAL TEST RESULTS =====")
    print("Selected model:", best_name)
    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1 Score:", round(f1, 4))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[0, 1, 2],
        columns=[0, 1, 2]
    )

    matrix_df.to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[0, 1, 2],
        yticklabels=[0, 1, 2]
    )

    plt.title(f"SoilSense Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Save final test metrics
    with open(
        REPORT_DIR / "test_metrics.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Selected Model: {best_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {precision:.4f}\n")
        file.write(f"Macro Recall: {recall:.4f}\n")
        file.write(f"Macro F1 Score: {f1:.4f}\n")


# ======================================================
# 6. SAVE MODEL AND COMPARISON
# ======================================================

def save_results(results, best_model, best_name):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results["Selected"] = results["Model"] == best_name

    results.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "best_soilsense_model.joblib"
    )

    with open(
        REPORT_DIR / "selected_model.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Selected Model: {best_name}\n")
        file.write("Selection Metric: 5-fold CV Macro F1\n")
        file.write(f"Random State: {RANDOM_STATE}\n")

    print("\n===== CROSS-VALIDATION COMPARISON =====")
    print(results.to_string(index=False))

    print("\nSaved model:", best_name)
    print("Model and reports saved successfully.")


# ======================================================
# 7. MAIN FUNCTION
# ======================================================

def main():

    print("=" * 55)
    print("SOILSENSE - SOIL FERTILITY MODEL TRAINING")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_data()

    results, best_name, best_model = train_models(
        X_train,
        y_train
    )

    save_results(
        results,
        best_model,
        best_name
    )

    evaluate_model(
        best_model,
        best_name,
        X_test,
        y_test
    )

    print("\n===== SOILSENSE TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()
