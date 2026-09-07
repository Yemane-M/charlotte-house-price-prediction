import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_PATH = "data/processed/test_grouped_coordinates.csv"
MODEL_PATH = "model/xgboost_grouped_coordinates.joblib"

PREDICTIONS_OUTPUT = "data/interim/xgboost_grouped_test_predictions.csv"
PRICE_RANGE_OUTPUT = "data/interim/xgboost_grouped_performance_by_price_range.csv"
FEATURE_IMPORTANCE_OUTPUT = "data/interim/xgboost_grouped_feature_importance.csv"

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
# LOAD DATA
# ============================================================

print("Loading grouped test data...")

test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Testing rows: {len(test_df):,}")

print("\nLoading grouped XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()

print("\nGenerating predictions...")

# The model was trained using log1p(saleprice)
predicted_log = model.predict(X_test)

# Convert predictions back to dollar values
predicted_price = np.expm1(predicted_log)

# Prevent negative predictions
predicted_price = np.maximum(predicted_price, 0)


# ============================================================
# OVERALL MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, predicted_price)
rmse = np.sqrt(mean_squared_error(y_test, predicted_price))
r2 = r2_score(y_test, predicted_price)

print("\n" + "=" * 60)
print("GROUP-AWARE XGBOOST ERROR ANALYSIS")
print("=" * 60)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

results = test_df.copy()

results["predicted_price"] = predicted_price

results["absolute_error"] = (
    results[TARGET] - results["predicted_price"]
).abs()

results["error"] = (
    results[TARGET] - results["predicted_price"]
)

results["percentage_error"] = (
    results["absolute_error"] / results[TARGET]
) * 100

results["percentage_error"] = (
    results["percentage_error"].replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# PERFORMANCE BY PRICE RANGE
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE BY PRICE RANGE")
print("=" * 60)

price_bins = [
    0,
    250_000,
    500_000,
    750_000,
    1_000_000,
    2_000_000,
    5_000_000,
    np.inf,
]

price_labels = [
    "<= $250K",
    "$250K - $500K",
    "$500K - $750K",
    "$750K - $1M",
    "$1M - $2M",
    "$2M - $5M",
    "> $5M",
]

results["price_range"] = pd.cut(
    results[TARGET],
    bins=price_bins,
    labels=price_labels,
    include_lowest=True,
)

range_results = []

for price_range, group in results.groupby(
    "price_range",
    observed=False
):

    if len(group) == 0:
        continue

    range_mae = mean_absolute_error(
        group[TARGET],
        group["predicted_price"]
    )

    range_rmse = np.sqrt(
        mean_squared_error(
            group[TARGET],
            group["predicted_price"]
        )
    )

    range_r2 = r2_score(
        group[TARGET],
        group["predicted_price"]
    ) if len(group) > 1 else np.nan

    range_mape = group["percentage_error"].mean()

    print(f"\n{price_range}")
    print(f"Count       : {len(group):,}")
    print(f"MAE         : ${range_mae:,.2f}")
    print(f"RMSE        : ${range_rmse:,.2f}")
    print(f"R²          : {range_r2:.4f}")
    print(f"Mean % Error: {range_mape:.2f}%")

    range_results.append(
        {
            "price_range": str(price_range),
            "count": len(group),
            "MAE": range_mae,
            "RMSE": range_rmse,
            "R2": range_r2,
            "mean_percentage_error": range_mape,
        }
    )

price_range_df = pd.DataFrame(range_results)


# ============================================================
# LARGEST OVERPREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("LARGEST OVERPREDICTIONS")
print("=" * 60)

largest_over = results.sort_values(
    "error",
    ascending=True
).head(10)

for _, row in largest_over.iterrows():

    print(
        f"Actual: ${row[TARGET]:,.0f} | "
        f"Predicted: ${row['predicted_price']:,.0f} | "
        f"Error: ${row['error']:,.0f} | "
        f"Neighborhood: {row['neighborhood']}"
    )


# ============================================================
# LARGEST UNDERPREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("LARGEST UNDERPREDICTIONS")
print("=" * 60)

largest_under = results.sort_values(
    "error",
    ascending=False
).head(10)

for _, row in largest_under.iterrows():

    print(
        f"Actual: ${row[TARGET]:,.0f} | "
        f"Predicted: ${row['predicted_price']:,.0f} | "
        f"Error: ${row['error']:,.0f} | "
        f"Neighborhood: {row['neighborhood']}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance_df = pd.DataFrame()

# ------------------------------------------------------------
# Case 1: Saved model is a Pipeline
# ------------------------------------------------------------

if hasattr(model, "named_steps"):

    print("\nModel is a Pipeline.")

    # Find the preprocessing step
    preprocessor = None
    xgb_model = None

    for step_name, step in model.named_steps.items():

        if hasattr(step, "transformers_"):
            preprocessor = step

        if hasattr(step, "feature_importances_"):
            xgb_model = step

    if preprocessor is not None and xgb_model is not None:

        print("Preprocessor found.")
        print("XGBoost model found.")

        # Get transformed feature names
        try:

            transformed_features = (
                preprocessor.get_feature_names_out()
            )

            feature_importances = (
                xgb_model.feature_importances_
            )

            print(
                f"Original features   : {len(FEATURES):,}"
            )

            print(
                f"Transformed features: "
                f"{len(transformed_features):,}"
            )

            print(
                f"Importance values    : "
                f"{len(feature_importances):,}"
            )

            if len(transformed_features) == len(
                feature_importances
            ):

                importance_df = pd.DataFrame(
                    {
                        "feature": transformed_features,
                        "importance": feature_importances,
                    }
                )

                importance_df = importance_df.sort_values(
                    "importance",
                    ascending=False
                )

            else:

                print(
                    "\nWARNING: Feature-name count does "
                    "not match importance count."
                )

        except Exception as e:

            print(
                f"\nCould not extract transformed "
                f"feature names: {e}"
            )


# ------------------------------------------------------------
# Case 2: Saved model is a direct XGBoost model
# ------------------------------------------------------------

elif hasattr(model, "feature_importances_"):

    print("\nModel is a direct XGBoost model.")

    feature_importances = model.feature_importances_

    if len(feature_importances) == len(FEATURES):

        importance_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": feature_importances,
            }
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

    else:

        print(
            "\nWARNING: Number of feature importance "
            "values does not match original features."
        )


# ------------------------------------------------------------
# Display feature importance
# ------------------------------------------------------------

if not importance_df.empty:

    print("\nTop 30 transformed features:\n")

    for _, row in importance_df.head(30).iterrows():

        print(
            f"{row['feature']:<60} "
            f"{row['importance']:.6f}"
        )

else:

    print(
        "\nFeature importance could not be extracted."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "data/interim",
    exist_ok=True
)

results.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

price_range_df.to_csv(
    PRICE_RANGE_OUTPUT,
    index=False
)

if not importance_df.empty:

    importance_df.to_csv(
        FEATURE_IMPORTANCE_OUTPUT,
        index=False
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print("\nFiles saved:")

print(
    f"Predictions: "
    f"{PREDICTIONS_OUTPUT}"
)

print(
    f"Price ranges: "
    f"{PRICE_RANGE_OUTPUT}"
)

if not importance_df.empty:

    print(
        f"Feature importance: "
        f"{FEATURE_IMPORTANCE_OUTPUT}"
    )