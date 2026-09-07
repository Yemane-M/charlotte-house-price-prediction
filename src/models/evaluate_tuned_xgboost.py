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
# LOAD TEST DATA
# ============================================================

print("Loading grouped spatial test data...")

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Testing rows: {len(test_df):,}")


# ============================================================
# LOAD TUNED MODEL
# ============================================================

print("\nLoading tuned XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print("Tuned model loaded successfully.")


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# PREDICT
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
# METRICS
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


# ============================================================
# CURRENT UNTUNED SPATIAL BASELINE
# ============================================================

BASELINE_MAE = 73_266.06
BASELINE_RMSE = 192_164.54
BASELINE_R2 = 0.8343


mae_change = (
    mae - BASELINE_MAE
)

rmse_change = (
    rmse - BASELINE_RMSE
)

r2_change = (
    r2 - BASELINE_R2
)


mae_percent = (
    mae_change
    / BASELINE_MAE
    * 100
)

rmse_percent = (
    rmse_change
    / BASELINE_RMSE
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TUNED XGBOOST TEST RESULTS")
print("=" * 60)

print(
    f"MAE : ${mae:,.2f}"
)

print(
    f"RMSE: ${rmse:,.2f}"
)

print(
    f"R²  : {r2:.4f}"
)


print("\n" + "=" * 60)
print("COMPARISON WITH UNTUNED SPATIAL MODEL")
print("=" * 60)

print(
    f"MAE change : "
    f"${mae_change:,.2f} "
    f"({mae_percent:+.2f}%)"
)

print(
    f"RMSE change: "
    f"${rmse_change:,.2f} "
    f"({rmse_percent:+.2f}%)"
)

print(
    f"R² change  : "
    f"{r2_change:+.4f}"
)


# ============================================================
# SIMPLE DECISION
# ============================================================

print("\n" + "=" * 60)
print("TUNING DECISION")
print("=" * 60)

if (
    mae < BASELINE_MAE
    and rmse < BASELINE_RMSE
    and r2 > BASELINE_R2
):
    print(
        "Tuned model improved all three metrics."
    )

elif (
    mae > BASELINE_MAE
    and rmse > BASELINE_RMSE
    and r2 < BASELINE_R2
):
    print(
        "Tuned model worsened all three metrics."
    )

else:
    print(
        "Tuning produced mixed results."
    )


# ============================================================
# PREDICTION SUMMARY
# ============================================================

print("\nPrediction summary:")

summary = pd.DataFrame(
    {
        "actual_price": y_test,
        "predicted_price": predicted_price,
    }
)

print(
    summary.describe()
)