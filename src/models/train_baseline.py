from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# Transform target
# ---------------------------------------------------------
#
# log1p(x) = log(1 + x)
#
# This reduces the influence of extremely expensive
# properties during training.
# ---------------------------------------------------------

y_train_log = np.log1p(y_train)


# ---------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
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
# Function to evaluate model
# ---------------------------------------------------------

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions_log = model.predict(X_test)

    # Convert predictions back to dollar values
    predictions = np.expm1(predictions_log)

    # Prevent tiny numerical negative values
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"MAE : ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ---------------------------------------------------------
# Dummy baseline
# ---------------------------------------------------------
#
# Predicts the median log-price for every property.
# This gives us a minimum benchmark that a useful model
# should beat.
# ---------------------------------------------------------

dummy_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DummyRegressor(strategy="median"),
        ),
    ]
)


# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LinearRegression(),
        ),
    ]
)


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------

results = []

results.append(
    evaluate_model(
        "Dummy Regressor",
        dummy_model,
        X_train,
        y_train_log,
        X_test,
        y_test,
    )
)

results.append(
    evaluate_model(
        "Linear Regression",
        linear_model,
        X_train,
        y_train_log,
        X_test,
        y_test,
    )
)


# ---------------------------------------------------------
# Compare results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "${:,.2f}".format,
            "RMSE": "${:,.2f}".format,
            "R2": "{:.4f}".format,
        },
    )
)


print("\nBaseline modeling completed successfully.")