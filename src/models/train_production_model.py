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

DATA_PATH = (
    "data/processed/"
    "house_price_modeling_spatial.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_production.joblib"
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
# LOAD FULL MODELING DATASET
# ============================================================

print("Loading full spatial modeling dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(f"Rows          : {len(df):,}")
print(
    f"Unique parcels: "
    f"{df['parcelid'].nunique():,}"
)

print(
    f"Sale year range: "
    f"{df['sale_year'].min()} "
    f"to "
    f"{df['sale_year'].max()}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [TARGET]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# LOG TRANSFORM TARGET
# ============================================================

y_log = np.log1p(
    y
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
# FINAL SELECTED XGBOOST CONFIGURATION
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
# COMPLETE PRODUCTION PIPELINE
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
# TRAIN PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING PRODUCTION XGBOOST MODEL")
print("=" * 65)

print("\nTraining data:")

print(
    f"Transactions: {len(df):,}"
)

print(
    f"Unique parcels: "
    f"{df['parcelid'].nunique():,}"
)

print(
    f"Features: {len(FEATURES)}"
)

print("\nSelected hyperparameters:")

print("n_estimators     : 900")
print("learning_rate    : 0.05")
print("max_depth        : 8")
print("min_child_weight : 3")
print("subsample        : 1.0")
print("colsample_bytree : 0.8")
print("reg_alpha        : 0.5")
print("reg_lambda       : 1")

print("\nTraining production model...")

model.fit(
    X,
    y_log,
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


print("\n" + "=" * 65)
print("PRODUCTION MODEL TRAINING COMPLETED")
print("=" * 65)

print("\nProduction model saved to:")

print(
    MODEL_PATH
)

print(
    "\nThis model was trained on all "
    "available modeling transactions."
)