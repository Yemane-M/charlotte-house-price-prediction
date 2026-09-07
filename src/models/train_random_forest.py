from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]

X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df[TARGET]


# ---------------------------------------------------------
# Log-transform target
# ---------------------------------------------------------

y_train_log = np.log1p(y_train)


# ---------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------
#
# Random Forest does not require scaling.
# We only need to handle missing values.
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)


# ---------------------------------------------------------
# Categorical preprocessing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Combined preprocessing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            random_forest,
        ),
    ]
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train_log,
)


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

print("Generating predictions...")

predictions_log = model.predict(X_test)

predictions = np.expm1(predictions_log)

# Protect against tiny negative numerical values
predictions = np.maximum(
    predictions,
    0,
)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")

print("\nRandom Forest training completed successfully.")