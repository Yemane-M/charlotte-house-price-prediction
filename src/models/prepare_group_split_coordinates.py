import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "house_price_modeling_coordinates.csv"
)

TRAIN_OUTPUT = (
    "data/processed/"
    "train_grouped_coordinates.csv"
)

TEST_OUTPUT = (
    "data/processed/"
    "test_grouped_coordinates.csv"
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
GROUP_COLUMN = "parcelid"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading coordinate-enriched modeling dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(f"Total rows     : {len(df):,}")
print(
    f"Unique parcels : "
    f"{df[GROUP_COLUMN].nunique():,}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [TARGET, GROUP_COLUMN]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

print("\nCreating group-aware train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_index, test_index = next(
    splitter.split(
        df,
        groups=df[GROUP_COLUMN],
    )
)

train_df = (
    df.iloc[train_index]
    .copy()
    .reset_index(drop=True)
)

test_df = (
    df.iloc[test_index]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# VERIFY PARCEL SEPARATION
# ============================================================

train_parcels = set(
    train_df[GROUP_COLUMN]
)

test_parcels = set(
    test_df[GROUP_COLUMN]
)

overlap = (
    train_parcels
    .intersection(test_parcels)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GROUP-AWARE COORDINATE SPLIT")
print("=" * 60)

print(
    f"Training rows   : "
    f"{len(train_df):,}"
)

print(
    f"Testing rows    : "
    f"{len(test_df):,}"
)

print(
    f"Training parcels: "
    f"{len(train_parcels):,}"
)

print(
    f"Testing parcels : "
    f"{len(test_parcels):,}"
)

print(
    f"Parcel overlap  : "
    f"{len(overlap):,}"
)

print(
    f"\nTraining %: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    f"Testing % : "
    f"{len(test_df) / len(df) * 100:.2f}%"
)


# ============================================================
# VERIFY COORDINATES
# ============================================================

print("\nCoordinate missing values:")

print("\nTraining:")

print(
    train_df[
        ["latitude", "longitude"]
    ]
    .isna()
    .sum()
)

print("\nTesting:")

print(
    test_df[
        ["latitude", "longitude"]
    ]
    .isna()
    .sum()
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTraining target:")

print(
    train_df[TARGET]
    .describe()
)

print("\nTesting target:")

print(
    test_df[TARGET]
    .describe()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

test_df.to_csv(
    TEST_OUTPUT,
    index=False,
)


print("\n" + "=" * 60)
print("COORDINATE GROUP SPLIT COMPLETED")
print("=" * 60)

print("\nFiles saved:")

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)