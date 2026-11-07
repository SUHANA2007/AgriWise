import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

# ==========================================
# FERTILIZERWISE MODEL TRAINING
# ==========================================

TRAIN_PATH = "data/processed/fertilizerwise/fertilizerwise_train.csv"
TEST_PATH = "data/processed/fertilizerwise/fertilizerwise_test.csv"

MODEL_DIR = "models/fertilizerwise"
REPORT_DIR = "reports/fertilizerwise"

NUMERICAL_FEATURES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "pH",
    "Rainfall",
    "Temperature"
]

CATEGORICAL_FEATURES = [
    "District_Name",
    "Soil_color",
    "Crop"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "Fertilizer"

RANDOM_STATE = 42


def create_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", NUMERICAL_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline


def main():

    print("=" * 55)
    print("FERTILIZERWISE MODEL TRAINING")
    print("=" * 55)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Load processed datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Fertilizer classes:", y_train.nunique())

    # 2. Define models
    models = {
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

    # 3. Configure cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )
    }

    comparison_results = []
    trained_models = {}

    # 4. Cross-validation and model comparison
    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        pipeline = create_pipeline(model)

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        mean_cv_accuracy = cv_results[
            "test_accuracy"
        ].mean()

        mean_cv_f1 = cv_results[
            "test_macro_f1"
        ].mean()

        std_cv_f1 = cv_results[
            "test_macro_f1"
        ].std()

        print("Mean CV Accuracy:", round(mean_cv_accuracy, 4))
        print("Mean CV Macro F1:", round(mean_cv_f1, 4))

        # 5. Fit model on complete training data
        pipeline.fit(X_train, y_train)

        # 6. Evaluate on held-out test data
        predictions = pipeline.predict(X_test)

        test_accuracy = accuracy_score(
            y_test,
            predictions
        )

        test_precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        print("Test Accuracy:", round(test_accuracy, 4))
        print("Test Macro F1:", round(test_f1, 4))

        comparison_results.append({
            "Model": name,
            "CV Accuracy": mean_cv_accuracy,
            "CV Macro F1": mean_cv_f1,
            "CV F1 Std": std_cv_f1,
            "Test Accuracy": test_accuracy,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test Macro F1": test_f1
        })

        trained_models[name] = pipeline

    # 7. Select model using CV Macro F1 only
    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df.sort_values(
        by=["CV Macro F1", "CV Accuracy"],
        ascending=False
    )

    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df.to_string(index=False))

    print("\nSelected model:", best_model_name)

    # 8. Save selected model
    model_path = os.path.join(
        MODEL_DIR,
        "best_fertilizerwise_model.joblib"
    )

    joblib.dump(best_model, model_path)

    # 9. Save comparison report
    comparison_path = os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # 10. Generate final classification report
    final_predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )

    report_path = os.path.join(
        REPORT_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # 11. Generate confusion matrix
    class_labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        final_predictions,
        labels=class_labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    cm_path = os.path.join(
        REPORT_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # 12. Save confusion matrix image
    plt.figure(figsize=(16, 13))

    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.title("FertilizerWise - Confusion Matrix")
    plt.colorbar()

    positions = range(len(class_labels))

    plt.xticks(
        positions,
        class_labels,
        rotation=90
    )

    plt.yticks(
        positions,
        class_labels
    )

    plt.xlabel("Predicted Fertilizer")
    plt.ylabel("Actual Fertilizer")
    plt.tight_layout()

    image_path = os.path.join(
        REPORT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n===== SAVED FILES =====")
    print(model_path)
    print(comparison_path)
    print(report_path)
    print(cm_path)
    print(image_path)

    print("\n===== FERTILIZERWISE TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()