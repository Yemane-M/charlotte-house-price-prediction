import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

TEST_PATH = (
    "data/processed/"
    "test_grouped_spatial.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_grouped_spatial_tuned.joblib"
)

OUTPUT_PATH = (
    "data/interim/"
    "xgboost_tuned_test_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "sale_year",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "landusefulldescription",
    "neighborhood",
    "latitude",
    "longitude",
    "distance_to_uptown_miles",
]

TARGET = "saleprice"


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("Loading test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Test rows: {len(test_df):,}")


print("\nLoading tuned model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTIONS
# ============================================================

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


print("Generating tuned predictions...")

predicted_log = model.predict(
    X_test
)

predicted_price = np.expm1(
    predicted_log
)

predicted_price = np.maximum(
    predicted_price,
    0,
)


# ============================================================
# ERROR COLUMNS
# ============================================================

results = test_df.copy()

results["predicted_price"] = predicted_price

results["error"] = (
    results["predicted_price"]
    - results[TARGET]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percent_error"] = (
    results["absolute_error"]
    / results[TARGET]
    * 100
)


# ============================================================
# OVERALL RESULTS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predicted_price,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_price,
    )
)

r2 = r2_score(
    y_test,
    predicted_price,
)


print("\n" + "=" * 70)
print("TUNED MODEL OVERALL PERFORMANCE")
print("=" * 70)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# PRICE BANDS
# ============================================================

bins = [
    0,
    250_000,
    500_000,
    750_000,
    1_000_000,
    2_000_000,
    5_000_000,
    np.inf,
]

labels = [
    "<= $250K",
    "$250K-$500K",
    "$500K-$750K",
    "$750K-$1M",
    "$1M-$2M",
    "$2M-$5M",
    "> $5M",
]


results["price_band"] = pd.cut(
    results[TARGET],
    bins=bins,
    labels=labels,
    include_lowest=True,
)


# ============================================================
# BAND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE BY PRICE BAND")
print("=" * 70)


for label in labels:

    band = results[
        results["price_band"] == label
    ].copy()

    if len(band) == 0:
        continue

    band_actual = band[TARGET]

    band_predicted = band[
        "predicted_price"
    ]

    band_mae = mean_absolute_error(
        band_actual,
        band_predicted,
    )

    band_rmse = np.sqrt(
        mean_squared_error(
            band_actual,
            band_predicted,
        )
    )

    # R² can be misleading within narrow price bands,
    # but we print it for diagnostic comparison.
    if len(band) > 1:
        band_r2 = r2_score(
            band_actual,
            band_predicted,
        )
    else:
        band_r2 = np.nan

    mean_percent_error = (
        band["percent_error"].mean()
    )

    median_percent_error = (
        band["percent_error"].median()
    )

    mean_signed_error = (
        band["error"].mean()
    )

    print("\n" + "-" * 70)

    print(f"Price band: {label}")

    print(f"Records: {len(band):,}")

    print(
        f"MAE : ${band_mae:,.2f}"
    )

    print(
        f"RMSE: ${band_rmse:,.2f}"
    )

    print(
        f"R²  : {band_r2:.4f}"
    )

    print(
        "Mean absolute % error  : "
        f"{mean_percent_error:.2f}%"
    )

    print(
        "Median absolute % error: "
        f"{median_percent_error:.2f}%"
    )

    print(
        "Mean signed error      : "
        f"${mean_signed_error:,.2f}"
    )


# ============================================================
# BIGGEST UNDERPREDICTIONS
# ============================================================

display_columns = [
    "parcelid",
    "saleprice",
    "predicted_price",
    "error",
    "absolute_error",
    "neighborhood",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "property_age",
    "distance_to_uptown_miles",
]


print("\n" + "=" * 70)
print("15 LARGEST UNDERPREDICTIONS")
print("=" * 70)

underpredictions = (
    results
    .sort_values("error")
    .head(15)
)

print(
    underpredictions[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# BIGGEST OVERPREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("15 LARGEST OVERPREDICTIONS")
print("=" * 70)

overpredictions = (
    results
    .sort_values(
        "error",
        ascending=False,
    )
    .head(15)
)

print(
    overpredictions[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nPredictions saved to:"
)

print(
    OUTPUT_PATH
)