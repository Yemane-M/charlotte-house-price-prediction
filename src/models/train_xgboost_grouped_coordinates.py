import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = (
    "data/processed/"
    "train_grouped_coordinates.csv"
)

TEST_PATH = (
    "data/processed/"
    "test_grouped_coordinates.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_grouped_coordinates.joblib"
)


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
# FEATURE TYPES
# ============================================================

numeric_features = [
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
    "latitude",
    "longitude",
]

categorical_features = [
    "landusefulldescription",
    "neighborhood",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading coordinate grouped training and testing data...")

train_df = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")


# ============================================================
# PREPARE X AND Y
# ============================================================

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
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
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# XGBOOST MODEL
# ============================================================

# IMPORTANT:
# Keep these parameters identical to the previous grouped
# XGBoost experiment. The only experimental change should
# be the addition of latitude and longitude.

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


# ============================================================
# PIPELINE
# ============================================================

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


# ============================================================
# LOG-TRANSFORM TARGET
# ============================================================

y_train_log = np.log1p(y_train)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining grouped XGBoost model with coordinates...")

model.fit(
    X_train,
    y_train_log,
)


# ============================================================
# PREDICT
# ============================================================

print("Generating predictions...")

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
# EVALUATE
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


print("\n" + "=" * 60)
print("GROUP-AWARE XGBOOST + COORDINATES RESULTS")
print("=" * 60)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# COMPARE WITH CURRENT BASELINE
# ============================================================

BASELINE_MAE = 101_336.82
BASELINE_RMSE = 235_548.68
BASELINE_R2 = 0.7510


print("\n" + "=" * 60)
print("COMPARISON WITH NEIGHBORHOOD-ONLY BASELINE")
print("=" * 60)

print(
    f"MAE change : "
    f"${mae - BASELINE_MAE:,.2f}"
)

print(
    f"RMSE change: "
    f"${rmse - BASELINE_RMSE:,.2f}"
)

print(
    f"R² change  : "
    f"{r2 - BASELINE_R2:+.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True,
)

joblib.dump(
    model,
    MODEL_PATH,
)


print("\nModel saved:")

print(MODEL_PATH)

print(
    "\nGrouped XGBoost coordinate "
    "training completed successfully."
)