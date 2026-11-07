
from pathlib import Path
import pandas as pd
import numpy as np

REPORT_DIR = Path("reports/yieldwise")

PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"


def calculate_error_slices(df, group_column):

    results = []

    for group_name, group in df.groupby(group_column):

        actual = group["hg/ha_yield"]
        predicted = group["Predicted_Yield"]

        errors = actual - predicted

        results.append({
            group_column: group_name,
            "Records": len(group),
            "MAE": errors.abs().mean(),
            "RMSE": np.sqrt((errors ** 2).mean()),
            "Mean Actual Yield": actual.mean()
        })

    return pd.DataFrame(results).sort_values(
        "MAE",
        ascending=False
    )


def main():

    df = pd.read_csv(PREDICTIONS_PATH)

    print("=" * 60)
    print("YIELDWISE - ERROR SLICE ANALYSIS")
    print("=" * 60)

    df["Absolute_Error"] = (
        df["hg/ha_yield"] - df["Predicted_Yield"]
    ).abs()

    # 1. Error analysis by crop
    crop_results = calculate_error_slices(
        df,
        "Item"
    )

    crop_results.to_csv(
        REPORT_DIR / "error_by_crop.csv",
        index=False
    )

    print("\nERROR BY CROP")
    print(crop_results.to_string(index=False))


    # 2. Error analysis by area
    area_results = calculate_error_slices(
        df,
        "Area"
    )

    area_results.to_csv(
        REPORT_DIR / "error_by_area.csv",
        index=False
    )

    print("\nTOP 15 AREAS WITH HIGHEST MAE")
    print(area_results.head(15).to_string(index=False))


    # 3. Error analysis by year
    year_results = calculate_error_slices(
        df,
        "Year"
    )

    year_results.to_csv(
        REPORT_DIR / "error_by_year.csv",
        index=False
    )

    print("\nERROR BY YEAR")
    print(year_results.to_string(index=False))


    # 4. Identify largest individual prediction errors
    worst_predictions = df.nlargest(
        20,
        "Absolute_Error"
    )

    worst_predictions.to_csv(
        REPORT_DIR / "worst_predictions.csv",
        index=False
    )

    print("\n20 LARGEST PREDICTION ERRORS")
    print(
        worst_predictions[
            [
                "Area",
                "Item",
                "Year",
                "hg/ha_yield",
                "Predicted_Yield",
                "Absolute_Error"
            ]
        ].to_string(index=False)
    )


    print("\nError analysis completed!")

    print("\nGenerated reports:")
    print("- error_by_crop.csv")
    print("- error_by_area.csv")
    print("- error_by_year.csv")
    print("- worst_predictions.csv")


if __name__ == "__main__":
    main()
