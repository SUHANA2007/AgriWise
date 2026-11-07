
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/yieldwise/best_yieldwise_model.joblib")
TRAIN_PATH = Path("data/processed/yieldwise/yieldwise_train.csv")


def get_number(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))

            if not minimum <= value <= maximum:
                print(
                    f"Please enter a value between "
                    f"{minimum:.2f} and {maximum:.2f}."
                )
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


def main():

    print("=" * 55)
    print("YIELDWISE - NEW CROP YIELD PREDICTION")
    print("=" * 55)

    # Load trained model and training data
    model = joblib.load(MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    # Allowed categories
    areas = sorted(train_df["Area"].unique())
    crops = sorted(train_df["Item"].unique())

    print("\nAvailable areas:")
    print(", ".join(areas))

    area = input("\nEnter Area/Country: ").strip()

    if area not in areas:
        print("Invalid area. Choose an area from the available list.")
        return

    print("\nAvailable crops:")
    print(", ".join(crops))

    crop = input("\nEnter Crop: ").strip()

    if crop not in crops:
        print("Invalid crop. Choose a crop from the available list.")
        return

    # Numeric input validation
    year = get_number(
        "\nEnter Year: ",
        train_df["Year"].min(),
        train_df["Year"].max()
    )

    rainfall = get_number(
        "Enter Average Rainfall (mm/year): ",
        train_df["average_rain_fall_mm_per_year"].min(),
        train_df["average_rain_fall_mm_per_year"].max()
    )

    pesticides = get_number(
        "Enter Pesticides (tonnes): ",
        train_df["pesticides_tonnes"].min(),
        train_df["pesticides_tonnes"].max()
    )

    temperature = get_number(
        "Enter Average Temperature (°C): ",
        train_df["avg_temp"].min(),
        train_df["avg_temp"].max()
    )

    # Prepare input record
    input_data = pd.DataFrame([{
        "Area": area,
        "Item": crop,
        "Year": int(year),
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }])

    # Predict yield
    prediction = model.predict(input_data)[0]

    print("\n" + "=" * 55)
    print("PREDICTION RESULT")
    print("=" * 55)

    print("Area:", area)
    print("Crop:", crop)
    print("Year:", int(year))

    print(
        f"\nPredicted Crop Yield: {prediction:,.2f} hg/ha"
    )

    print("\nNote:")
    print(
        "This is an ML-based estimate using historical dataset "
        "patterns, not a guaranteed actual yield."
    )


if __name__ == "__main__":
    main()
