from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "house_price_modeling.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET = "saleprice"

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
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df):,}")


# ---------------------------------------------------------
# Select features and target
# ---------------------------------------------------------

model_df = df[FEATURES + [TARGET]].copy()

print("\nSelected modeling columns:")
for column in model_df.columns:
    print(f"  {column}")


# ---------------------------------------------------------
# Separate X and y
# ---------------------------------------------------------

X = model_df[FEATURES].copy()
y = model_df[TARGET].copy()


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

print("\nTARGET CHECK")
print(f"Target: {TARGET}")
print(f"Target missing: {y.isna().sum():,}")
print(f"Target dtype: {y.dtype}")

print("\nFEATURE CHECK")

for column in FEATURES:
    print(
        f"{column:30} "
        f"dtype={X[column].dtype} "
        f"missing={X[column].isna().sum():,}"
    )


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------
# 80% training
# 20% testing
#
# random_state makes the split reproducible.
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ---------------------------------------------------------
# Verify split
# ---------------------------------------------------------

print("\nTRAIN / TEST SPLIT")

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")

print(
    f"Training percentage: "
    f"{len(X_train) / len(X) * 100:.1f}%"
)

print(
    f"Testing percentage : "
    f"{len(X_test) / len(X) * 100:.1f}%"
)


# ---------------------------------------------------------
# Target distribution
# ---------------------------------------------------------

print("\nTARGET DISTRIBUTION")

print("Training:")
print(y_train.describe())

print("\nTesting:")
print(y_test.describe())


# ---------------------------------------------------------
# Save split datasets
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


train_output = OUTPUT_DIR / "train.csv"
test_output = OUTPUT_DIR / "test.csv"


train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test


train_df.to_csv(
    train_output,
    index=False,
)

test_df.to_csv(
    test_output,
    index=False,
)


# ---------------------------------------------------------
# Final confirmation
# ---------------------------------------------------------

print("\nFILES CREATED")

print(f"Training data: {train_output}")
print(f"Testing data : {test_output}")

print("\nData preparation completed successfully.")