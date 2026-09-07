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
    "train_grouped_spatial.csv"
)

TEST_PATH = (
    "data/processed/"
    "test_grouped_spatial.csv"
)

MODEL_PATH = (
    "model/"
    "xgboost_grouped_spatial.joblib"
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
# LOAD DATA
# ============================================================

print("Loading grouped spatial datasets...")

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
# PREPARE DATA
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
# XGBOOST
# ============================================================

# Keep these parameters IDENTICAL to the coordinate model.
# The only experimental change is:
#
# + distance_to_uptown_miles

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
# LOG TARGET
# ============================================================

y_train_log = np.log1p(
    y_train
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTraining grouped XGBoost "
    "spatial model..."
)

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


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GROUP-AWARE XGBOOST + SPATIAL RESULTS")
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


# ============================================================
# COMPARE WITH COORDINATE BASELINE
# ============================================================

BASELINE_MAE = 75_111.01
BASELINE_RMSE = 192_810.86
BASELINE_R2 = 0.8332


mae_change = (
    mae - BASELINE_MAE
)

rmse_change = (
    rmse - BASELINE_RMSE
)

r2_change = (
    r2 - BASELINE_R2
)


mae_percent_change = (
    mae_change
    / BASELINE_MAE
    * 100
)

rmse_percent_change = (
    rmse_change
    / BASELINE_RMSE
    * 100
)


print("\n" + "=" * 60)
print("COMPARISON WITH COORDINATE BASELINE")
print("=" * 60)

print(
    f"MAE change : "
    f"${mae_change:,.2f} "
    f"({mae_percent_change:+.2f}%)"
)

print(
    f"RMSE change: "
    f"${rmse_change:,.2f} "
    f"({rmse_percent_change:+.2f}%)"
)

print(
    f"R² change  : "
    f"{r2_change:+.4f}"
)


# ============================================================
# DECISION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

if (
    mae < BASELINE_MAE
    and rmse < BASELINE_RMSE
    and r2 > BASELINE_R2
):
    print(
        "Distance-to-Uptown improved all "
        "three evaluation metrics."
    )

elif (
    mae > BASELINE_MAE
    and rmse > BASELINE_RMSE
    and r2 < BASELINE_R2
):
    print(
        "Distance-to-Uptown worsened all "
        "three evaluation metrics."
    )

else:
    print(
        "Results are mixed. Review the "
        "metric changes before deciding "
        "whether to keep the feature."
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
    "\nGrouped spatial XGBoost "
    "training completed successfully."
)