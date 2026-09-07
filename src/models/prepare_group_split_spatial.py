import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "house_price_modeling_spatial.csv"
)

TRAIN_OUTPUT = (
    "data/processed/"
    "train_grouped_spatial.csv"
)

TEST_OUTPUT = (
    "data/processed/"
    "test_grouped_spatial.csv"
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
    "distance_to_uptown_miles",
]

TARGET = "saleprice"
GROUP_COLUMN = "parcelid"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading spatial modeling dataset...")

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
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

print(
    "\nCreating group-aware spatial "
    "train/test split..."
)

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

overlap = train_parcels.intersection(
    test_parcels
)


# ============================================================
# VERIFY SPATIAL FEATURE
# ============================================================

print("\n" + "=" * 60)
print("GROUP-AWARE SPATIAL SPLIT")
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


print("\nDistance missing values:")

print(
    "Training: "
    f"{train_df['distance_to_uptown_miles'].isna().sum():,}"
)

print(
    "Testing : "
    f"{test_df['distance_to_uptown_miles'].isna().sum():,}"
)


# ============================================================
# COMPARE WITH COORDINATE SPLIT
# ============================================================

COORDINATE_TRAIN_PATH = (
    "data/processed/"
    "train_grouped_coordinates.csv"
)

COORDINATE_TEST_PATH = (
    "data/processed/"
    "test_grouped_coordinates.csv"
)


if (
    os.path.exists(COORDINATE_TRAIN_PATH)
    and os.path.exists(COORDINATE_TEST_PATH)
):

    print(
        "\nComparing parcel sets with "
        "previous coordinate experiment..."
    )

    old_train = pd.read_csv(
        COORDINATE_TRAIN_PATH,
        usecols=[GROUP_COLUMN],
    )

    old_test = pd.read_csv(
        COORDINATE_TEST_PATH,
        usecols=[GROUP_COLUMN],
    )

    old_train_parcels = set(
        old_train[GROUP_COLUMN]
    )

    old_test_parcels = set(
        old_test[GROUP_COLUMN]
    )

    print(
        "Training parcel sets identical: "
        f"{train_parcels == old_train_parcels}"
    )

    print(
        "Testing parcel sets identical : "
        f"{test_parcels == old_test_parcels}"
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
print("SPATIAL GROUP SPLIT COMPLETED")
print("=" * 60)

print("\nFiles saved:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)