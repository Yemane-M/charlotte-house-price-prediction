import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = (
    "data/processed/"
    "train_temporal.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_temporal.joblib"
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
    "distance_to_uptown_miles",
]

categorical_features = [
    "landusefulldescription",
    "neighborhood",
]


# ============================================================
# LOAD TEMPORAL TRAINING DATA
# ============================================================

print("Loading temporal training dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

print(f"Training rows: {len(train_df):,}")

print(
    f"Unique parcels: "
    f"{train_df['parcelid'].nunique():,}"
)

print(
    f"Sale year range: "
    f"{train_df['sale_year'].min()} "
    f"to "
    f"{train_df['sale_year'].max()}"
)


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_train = train_df[
    FEATURES
].copy()

y_train = train_df[
    TARGET
].copy()


# ============================================================
# LOG TRANSFORM TARGET
# ============================================================

y_train_log = np.log1p(
    y_train
)


# ============================================================
# NUMERIC PREPROCESSING
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


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

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


# ============================================================
# PREPROCESSOR
# ============================================================

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
# XGBOOST
#
# Winning parameters from our group-aware tuning experiment.
# We are NOT tuning against the temporal test set.
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=1.0,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# COMPLETE PIPELINE
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
# TRAIN
# ============================================================

print("\n" + "=" * 65)
print("TRAINING TEMPORAL XGBOOST MODEL")
print("=" * 65)

print("\nHyperparameters:")

print("n_estimators     : 900")
print("learning_rate    : 0.05")
print("max_depth        : 8")
print("min_child_weight : 3")
print("subsample        : 1.0")
print("colsample_bytree : 0.8")
print("reg_alpha        : 0.5")
print("reg_lambda       : 1")

print("\nTraining...")

model.fit(
    X_train,
    y_train_log,
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "model",
    exist_ok=True,
)

joblib.dump(
    model,
    MODEL_PATH,
)


print("\n" + "=" * 65)
print("TEMPORAL MODEL TRAINING COMPLETED")
print("=" * 65)

print("\nModel saved to:")

print(
    MODEL_PATH
)

print(
    "\nThe 2025-2026 temporal test set "
    "has NOT been used during training."
)