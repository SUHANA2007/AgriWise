
import calendar
import math
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# RAINWISE - NEXT-DAY RAINFALL PREDICTION
# AgriWise - Smart Agriculture Prediction System using ML
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "rainwise" / "rainwise_best_model.pkl"

FEATURES = [
    "city",
    "previous_day_temperature_2m_max",
    "previous_day_temperature_2m_min",
    "previous_day_apparent_temperature_max",
    "previous_day_apparent_temperature_min",
    "previous_day_wind_speed_10m_max",
    "previous_day_wind_gusts_10m_max",
    "previous_day_wind_direction_10m_dominant",
    "previous_day_rain_sum",
    "target_dayofyear_sin",
    "target_dayofyear_cos",
]

CITIES = [
    "Ahmedabad",
    "Bangalore",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Jaipur",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Pune",
]


def get_float(prompt, minimum=None, maximum=None):

    while True:
        try:
            value = float(input(prompt))

            if not math.isfinite(value):
                print("Please enter a valid finite number.")
                continue

            if minimum is not None and value < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must not exceed {maximum}.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_city():

    print("\nAvailable cities:")

    for city in CITIES:
        print("-", city)

    while True:
        city = input("\nEnter city: ").strip()

        for available_city in CITIES:
            if city.lower() == available_city.lower():
                return available_city

        print("Invalid city. Please select a city from the list.")


def get_observation_date():

    while True:
        date_text = input(
            "\nEnter observation date (YYYY-MM-DD): "
        ).strip()

        try:
            return datetime.strptime(
                date_text, "%Y-%m-%d"
            ).date()

        except ValueError:
            print("Invalid date. Use YYYY-MM-DD format.")


def main():

    print("=" * 60)
    print("RAINWISE - NEXT-DAY RAINFALL PREDICTION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("\nEnter the current day's weather information.")

    # --------------------------------------------------------
    # 1. BASIC INPUTS
    # --------------------------------------------------------

    city = get_city()

    observation_date = get_observation_date()
    target_date = observation_date + timedelta(days=1)

    # --------------------------------------------------------
    # 2. TEMPERATURE INPUTS
    # --------------------------------------------------------

    print("\nTEMPERATURE DETAILS")

    temp_max = get_float(
        "Maximum temperature (C): ", -90, 65
    )

    temp_min = get_float(
        "Minimum temperature (C): ", -90, 65
    )

    if temp_max < temp_min:
        raise ValueError(
            "Maximum temperature cannot be below minimum temperature."
        )

    apparent_max = get_float(
        "Maximum apparent temperature (C): ", -90, 65
    )

    apparent_min = get_float(
        "Minimum apparent temperature (C): ", -90, 65
    )

    if apparent_max < apparent_min:
        raise ValueError(
            "Maximum apparent temperature cannot be below minimum."
        )

    # --------------------------------------------------------
    # 3. WIND INPUTS
    # --------------------------------------------------------

    print("\nWIND DETAILS")

    wind_speed = get_float(
        "Maximum wind speed (km/h): ", 0
    )

    wind_gust = get_float(
        "Maximum wind gust (km/h): ", 0
    )

    wind_direction = get_float(
        "Dominant wind direction (degrees, 0-360): ", 0, 360
    )

    # --------------------------------------------------------
    # 4. CURRENT-DAY RAINFALL
    # --------------------------------------------------------

    print("\nRAINFALL DETAILS")

    rain_sum = get_float(
        "Current-day rainfall (mm): ", 0
    )

    # --------------------------------------------------------
    # 5. SEASONAL FEATURE CALCULATION
    # --------------------------------------------------------

    day_of_year = target_date.timetuple().tm_yday

    days_in_year = (
        366 if calendar.isleap(target_date.year) else 365
    )

    target_dayofyear_sin = np.sin(
        2 * np.pi * day_of_year / days_in_year
    )

    target_dayofyear_cos = np.cos(
        2 * np.pi * day_of_year / days_in_year
    )

    # --------------------------------------------------------
    # 6. PREPARE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            city,
            temp_max,
            temp_min,
            apparent_max,
            apparent_min,
            wind_speed,
            wind_gust,
            wind_direction,
            rain_sum,
            target_dayofyear_sin,
            target_dayofyear_cos,
        ]],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # 7. PREDICTION
    # --------------------------------------------------------

    prediction = float(model.predict(input_data)[0])

    prediction = max(prediction, 0.0)

    # --------------------------------------------------------
    # 8. DISPLAY RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAINWISE - PREDICTION RESULT")
    print("=" * 60)

    print("City:", city)
    print("Observation date:", observation_date)
    print("Prediction date:", target_date)

    print(
        "\nPredicted next-day rainfall:",
        f"{prediction:.2f} mm"
    )

    print("=" * 60)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nPrediction failed:", error)
