
import joblib
import pandas as pd
from pathlib import Path

# =======================================================
# CROPWISE - CROP RECOMMENDATION
# =======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cropwise" / "best_cropwise_model.joblib"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "cropwise" / "cropwise_train.csv"


def main():

    print("=" * 55)
    print("CROPWISE - CROP RECOMMENDATION")
    print("=" * 55)

    # Check required files
    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found.")
        return

    if not TRAIN_PATH.exists():
        print("ERROR: Training dataset not found.")
        return

    # Load model and training data
    model = joblib.load(MODEL_PATH)
    train_data = pd.read_csv(TRAIN_PATH)

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    # Get valid input ranges from training data
    print("\nEnter soil and weather details.")

    user_input = {}

    for feature in features:

        minimum = train_data[feature].min()
        maximum = train_data[feature].max()

        while True:
            try:
                value = float(input(f"Enter {feature}: "))

                if value < minimum or value > maximum:
                    print(
                        f"Please enter a value between "
                        f"{minimum:.2f} and {maximum:.2f}."
                    )
                    continue

                user_input[feature] = value
                break

            except ValueError:
                print("Invalid input. Please enter a number.")

    # Convert input into DataFrame
    input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Highest probability
    confidence = max(probabilities) * 100

    # Find top 5 predictions
    top_indices = probabilities.argsort()[::-1][:5]

    # Display results
    print("\n" + "=" * 55)
    print("CROPWISE PREDICTION RESULT")
    print("=" * 55)

    print(f"\nRecommended Crop: {prediction}")
    print(f"Model Prediction Confidence: {confidence:.2f}%")

    print("\nTOP 5 CROP PREDICTIONS")
    print("-" * 40)

    for rank, index in enumerate(top_indices, start=1):

        crop = model.classes_[index]
        probability = probabilities[index] * 100

        print(f"{rank}. {crop}: {probability:.2f}%")

    print("\n" + "=" * 55)

    print("Note:")
    print(
        "These predictions are based on historical "
        "dataset patterns."
    )
    print(
        "Model probabilities are not guarantees of "
        "agricultural suitability."
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
