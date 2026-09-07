import os

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
    "test_temporal.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_temporal.joblib"
)

PREDICTIONS_PATH = (
    "data/interim/"
    "xgboost_temporal_test_predictions.csv"
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
# LOAD TEMPORAL TEST DATA
# ============================================================

print("Loading temporal test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Testing rows   : {len(test_df):,}")

print(
    f"Unique parcels : "
    f"{test_df['parcelid'].nunique():,}"
)

print(
    f"Sale year range: "
    f"{test_df['sale_year'].min()} "
    f"to "
    f"{test_df['sale_year'].max()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading temporal XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREPARE TEST FEATURES
# ============================================================

X_test = test_df[
    FEATURES
].copy()

y_test = test_df[
    TARGET
].copy()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating future-sale predictions...")

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

results["absolute_percent_error"] = (
    results["absolute_error"]
    / results[TARGET]
    * 100
)


# ============================================================
# OVERALL METRICS
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

mean_signed_error = (
    results["error"].mean()
)

median_absolute_error = (
    results["absolute_error"].median()
)

median_absolute_percent_error = (
    results[
        "absolute_percent_error"
    ].median()
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TEST RESULTS: 2025-2026")
print("=" * 70)

print(
    f"MAE                 : "
    f"${mae:,.2f}"
)

print(
    f"RMSE                : "
    f"${rmse:,.2f}"
)

print(
    f"R²                  : "
    f"{r2:.4f}"
)

print(
    f"Median absolute error: "
    f"${median_absolute_error:,.2f}"
)

print(
    f"Median absolute % error: "
    f"{median_absolute_percent_error:.2f}%"
)

print(
    f"Mean signed error   : "
    f"${mean_signed_error:,.2f}"
)


# ============================================================
# ACTUAL VS PREDICTED SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED PRICE SUMMARY")
print("=" * 70)

print(
    f"Actual mean     : "
    f"${y_test.mean():,.2f}"
)

print(
    f"Predicted mean  : "
    f"${predicted_price.mean():,.2f}"
)

print(
    f"Actual median   : "
    f"${y_test.median():,.2f}"
)

print(
    f"Predicted median: "
    f"${np.median(predicted_price):,.2f}"
)


# ============================================================
# PERFORMANCE BY YEAR
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE BY SALE YEAR")
print("=" * 70)

for year in sorted(
    results["sale_year"].unique()
):

    year_df = results[
        results["sale_year"] == year
    ].copy()

    actual = year_df[TARGET]
    predicted = year_df[
        "predicted_price"
    ]

    year_mae = mean_absolute_error(
        actual,
        predicted,
    )

    year_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    year_r2 = r2_score(
        actual,
        predicted,
    )

    year_signed_error = (
        year_df["error"].mean()
    )

    year_median_ape = (
        year_df[
            "absolute_percent_error"
        ].median()
    )

    print("\n" + "-" * 70)

    print(
        f"Sale year: {year}"
    )

    print(
        f"Records: {len(year_df):,}"
    )

    print(
        f"Actual mean    : "
        f"${actual.mean():,.2f}"
    )

    print(
        f"Predicted mean : "
        f"${predicted.mean():,.2f}"
    )

    print(
        f"MAE            : "
        f"${year_mae:,.2f}"
    )

    print(
        f"RMSE           : "
        f"${year_rmse:,.2f}"
    )

    print(
        f"R²             : "
        f"{year_r2:.4f}"
    )

    print(
        f"Median abs % error: "
        f"{year_median_ape:.2f}%"
    )

    print(
        f"Mean signed error : "
        f"${year_signed_error:,.2f}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    "data/interim",
    exist_ok=True,
)

results.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("TEMPORAL EVALUATION COMPLETED")
print("=" * 70)

print("\nPredictions saved to:")

print(
    PREDICTIONS_PATH
)