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


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train_grouped.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_grouped.csv"

MODEL_FILE = PROJECT_ROOT / "model" / "xgboost_grouped.joblib"


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
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
]

TARGET = "saleprice"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading grouped training and testing data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ---------------------------------------------------------
# Identify feature types
# ---------------------------------------------------------
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
]

categorical_features = [
    "landusefulldescription",
     "neighborhood",
]


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
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
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ---------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------
model = XGBRegressor(
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


# ---------------------------------------------------------
# Create pipeline
# ---------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
print("\nTraining grouped XGBoost model...")

y_train_log = np.log1p(y_train)

pipeline.fit(X_train, y_train_log)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
print("Generating predictions...")

predictions_log = pipeline.predict(X_test)

predictions = np.expm1(predictions_log)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("GROUP-AWARE XGBOOST RESULTS")
print("=" * 60)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
joblib.dump(pipeline, MODEL_FILE)

print("\nModel saved:")
print(MODEL_FILE)

print("\nGrouped XGBoost training completed successfully.")