import os

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TEST_PATH = (
    "data/processed/"
    "test_grouped_coordinates.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_grouped_coordinates.joblib"
)

OUTPUT_PATH = (
    "data/interim/"
    "xgboost_grouped_coordinates_test_predictions.csv"
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
]

TARGET = "saleprice"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading coordinate grouped test data...")

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Testing rows: {len(test_df):,}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading coordinate XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREPARE FEATURES
# ============================================================

X_test = test_df[FEATURES].copy()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

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
# CREATE RESULTS
# ============================================================

results = test_df.copy()

results["predicted_price"] = predicted_price

# Positive error = model underpredicted
# Negative error = model overpredicted

results["error"] = (
    results[TARGET]
    - results["predicted_price"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results[TARGET]
    * 100
)

results["percentage_error"] = (
    results["percentage_error"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


# ============================================================
# VERIFY
# ============================================================

print("\nPrediction summary:")

print(
    results[
        [
            TARGET,
            "predicted_price",
            "absolute_error",
        ]
    ]
    .describe()
)

print(
    f"\nPrediction rows: "
    f"{len(results):,}"
)

if len(results) != len(test_df):

    raise ValueError(
        "Prediction row count does not match test data."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/interim",
    exist_ok=True,
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 60)
print("COORDINATE PREDICTIONS SAVED")
print("=" * 60)

print("\nOutput:")

print(OUTPUT_PATH)