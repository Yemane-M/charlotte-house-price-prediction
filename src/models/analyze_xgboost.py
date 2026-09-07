from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Configuration
# =========================================================

TARGET = "saleprice"

NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "landusefulldescription",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================================================
# Load data
# =========================================================

print("Loading training and testing data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")


# =========================================================
# Target transformation
# =========================================================

y_train_log = np.log1p(y_train)


# =========================================================
# Preprocessing
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ]
)


# =========================================================
# XGBoost model
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


# =========================================================
# Complete pipeline
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            xgb_model,
        ),
    ]
)


# =========================================================
# Train model
# =========================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train_log,
)


# =========================================================
# Generate predictions
# =========================================================

print("Generating predictions...")

predictions_log = model.predict(X_test)

predictions = np.expm1(predictions_log)

predictions = np.maximum(
    predictions,
    0,
)


# =========================================================
# Overall metrics
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)


print("\n")
print("=" * 60)
print("OVERALL XGBOOST PERFORMANCE")
print("=" * 60)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# =========================================================
# Create prediction DataFrame
# =========================================================

analysis_df = test_df.copy()

analysis_df["predicted_price"] = predictions

analysis_df["error"] = (
    analysis_df[TARGET]
    - analysis_df["predicted_price"]
)

analysis_df["absolute_error"] = (
    analysis_df["error"]
    .abs()
)

# Avoid division by zero
analysis_df["percentage_error"] = np.where(
    analysis_df[TARGET] != 0,
    (
        analysis_df["absolute_error"]
        / analysis_df[TARGET]
        * 100
    ),
    np.nan,
)


# =========================================================
# Error summary
# =========================================================

print("\n")
print("=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(
    analysis_df[
        [
            TARGET,
            "predicted_price",
            "error",
            "absolute_error",
            "percentage_error",
        ]
    ].describe()
)


# =========================================================
# Performance by price range
# =========================================================

print("\n")
print("=" * 60)
print("PERFORMANCE BY ACTUAL SALE PRICE")
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
    "<=$250K",
    "$250K-$500K",
    "$500K-$750K",
    "$750K-$1M",
    "$1M-$2M",
    "$2M-$5M",
    ">$5M",
]

analysis_df["price_range"] = pd.cut(
    analysis_df[TARGET],
    bins=price_bins,
    labels=price_labels,
    include_lowest=True,
)


price_performance = (
    analysis_df
    .groupby(
        "price_range",
        observed=True,
    )
    .agg(
        records=(TARGET, "count"),
        MAE=("absolute_error", "mean"),
        median_absolute_error=(
            "absolute_error",
            "median",
        ),
        RMSE=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2)),
        ),
        mean_percentage_error=(
            "percentage_error",
            "mean",
        ),
    )
    .reset_index()
)

print(
    price_performance.to_string(
        index=False,
        formatters={
            "MAE": "${:,.2f}".format,
            "median_absolute_error": "${:,.2f}".format,
            "RMSE": "${:,.2f}".format,
            "mean_percentage_error": "{:.2f}%".format,
        },
    )
)


# =========================================================
# Largest overpredictions
# =========================================================
#
# error < 0 means:
#
# predicted price > actual price
# =========================================================

print("\n")
print("=" * 60)
print("LARGEST OVERPREDICTIONS")
print("=" * 60)

overpredictions = (
    analysis_df
    .sort_values(
        "error",
        ascending=True,
    )
    .head(15)
)

print(
    overpredictions[
        [
            TARGET,
            "predicted_price",
            "error",
            "absolute_error",
            "sale_year",
            "gisacres",
            "heatedarea",
            "finisharea",
            "bedrooms",
            "bathroom_equivalents",
            "property_age",
            "landusefulldescription",
        ]
    ].to_string(index=False)
)


# =========================================================
# Largest underpredictions
# =========================================================
#
# error > 0 means:
#
# actual price > predicted price
# =========================================================

print("\n")
print("=" * 60)
print("LARGEST UNDERPREDICTIONS")
print("=" * 60)

underpredictions = (
    analysis_df
    .sort_values(
        "error",
        ascending=False,
    )
    .head(15)
)

print(
    underpredictions[
        [
            TARGET,
            "predicted_price",
            "error",
            "absolute_error",
            "sale_year",
            "gisacres",
            "heatedarea",
            "finisharea",
            "bedrooms",
            "bathroom_equivalents",
            "property_age",
            "landusefulldescription",
        ]
    ].to_string(index=False)
)


# =========================================================
# Feature importance
# =========================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_xgb = model.named_steps["model"]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_values = fitted_xgb.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance_values,
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    importance_df.head(20).to_string(
        index=False,
        formatters={
            "importance": "{:.6f}".format,
        },
    )
)


# =========================================================
# Save analysis results
# =========================================================

predictions_file = (
    OUTPUT_DIR
    / "xgboost_test_predictions.csv"
)

performance_file = (
    OUTPUT_DIR
    / "xgboost_performance_by_price_range.csv"
)

importance_file = (
    OUTPUT_DIR
    / "xgboost_feature_importance.csv"
)

analysis_df.to_csv(
    predictions_file,
    index=False,
)

price_performance.to_csv(
    performance_file,
    index=False,
)

importance_df.to_csv(
    importance_file,
    index=False,
)


# =========================================================
# Save trained model
# =========================================================

model_file = (
    MODEL_DIR
    / "xgboost_baseline.joblib"
)

joblib.dump(
    model,
    model_file,
)


# =========================================================
# Final output
# =========================================================

print("\n")
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print(f"Predictions      : {predictions_file}")
print(f"Price performance: {performance_file}")
print(f"Feature importance: {importance_file}")
print(f"Model             : {model_file}")

print("\nXGBoost error analysis completed successfully.")