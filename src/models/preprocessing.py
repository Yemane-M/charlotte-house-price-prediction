from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)


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
# Load training and testing data
# ---------------------------------------------------------

print("Loading training and testing data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]

X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df[TARGET]


# ---------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------
#
# Missing numeric values are replaced with the median
# calculated from the TRAINING data.
#
# StandardScaler is included because some future models
# benefit from standardized features.
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
#
# Missing categorical values would be replaced with the
# most frequent category.
#
# OneHotEncoder converts the seven land-use categories
# into numerical columns.
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
# Combine preprocessing pipelines
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
# FIT ONLY ON TRAINING DATA
# ---------------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)


# ---------------------------------------------------------
# Transform test data
# ---------------------------------------------------------
#
# IMPORTANT:
# We use transform(), NOT fit_transform().
#
# The test data must not influence the preprocessing.
# ---------------------------------------------------------

X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\nPREPROCESSING RESULTS")

print(
    f"Original training features: "
    f"{X_train.shape[1]}"
)

print(
    f"Processed training features: "
    f"{X_train_processed.shape[1]}"
)

print(
    f"Original testing features: "
    f"{X_test.shape[1]}"
)

print(
    f"Processed testing features: "
    f"{X_test_processed.shape[1]}"
)


# ---------------------------------------------------------
# Verify missing values
# ---------------------------------------------------------

print("\nMISSING VALUES AFTER PREPROCESSING")

print(
    f"Training missing values: "
    f"{pd.isna(X_train_processed).sum()}"
)

print(
    f"Testing missing values: "
    f"{pd.isna(X_test_processed).sum()}"
)


# ---------------------------------------------------------
# Verify target
# ---------------------------------------------------------

print("\nTARGET")

print(f"Training target rows: {len(y_train):,}")
print(f"Testing target rows : {len(y_test):,}")

print("\nPreprocessing completed successfully.")