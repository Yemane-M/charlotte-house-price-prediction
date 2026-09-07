import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "house_price_modeling_spatial.csv"
)

TRAIN_OUTPUT = (
    "data/processed/"
    "train_temporal.csv"
)

TEST_OUTPUT = (
    "data/processed/"
    "test_temporal.csv"
)

GROUP_COLUMN = "parcelid"
YEAR_COLUMN = "sale_year"

TRAIN_END_YEAR = 2024
TEST_START_YEAR = 2025


# ============================================================
# LOAD DATA
# ============================================================

print("Loading spatial modeling dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(f"Total rows    : {len(df):,}")
print(
    f"Unique parcels: "
    f"{df[GROUP_COLUMN].nunique():,}"
)

print(
    f"Sale year range: "
    f"{df[YEAR_COLUMN].min()} "
    f"to "
    f"{df[YEAR_COLUMN].max()}"
)


# ============================================================
# SHOW YEAR DISTRIBUTION
# ============================================================

print("\nTransactions by sale year:")

year_counts = (
    df[YEAR_COLUMN]
    .value_counts()
    .sort_index()
)

print(year_counts.to_string())


# ============================================================
# INITIAL TEMPORAL SPLIT
# ============================================================

historical_df = df[
    df[YEAR_COLUMN] <= TRAIN_END_YEAR
].copy()

test_df = df[
    df[YEAR_COLUMN] >= TEST_START_YEAR
].copy()


print("\n" + "=" * 65)
print("INITIAL TEMPORAL SPLIT")
print("=" * 65)

print(
    f"Historical rows through {TRAIN_END_YEAR}: "
    f"{len(historical_df):,}"
)

print(
    f"Test rows from {TEST_START_YEAR}: "
    f"{len(test_df):,}"
)


# ============================================================
# FIND TEST PARCELS
# ============================================================

test_parcels = set(
    test_df[GROUP_COLUMN]
)

historical_parcels = set(
    historical_df[GROUP_COLUMN]
)

initial_overlap = (
    historical_parcels
    .intersection(test_parcels)
)


print(
    "\nParcels appearing in both historical "
    "and future periods:"
)

print(
    f"{len(initial_overlap):,}"
)


# ============================================================
# REMOVE FUTURE TEST PARCELS FROM TRAINING
# ============================================================

train_df = historical_df[
    ~historical_df[GROUP_COLUMN].isin(
        test_parcels
    )
].copy()


# ============================================================
# FINAL PARCEL CHECK
# ============================================================

train_parcels = set(
    train_df[GROUP_COLUMN]
)

test_parcels = set(
    test_df[GROUP_COLUMN]
)

final_overlap = (
    train_parcels
    .intersection(test_parcels)
)


# ============================================================
# VALIDATE TEMPORAL ORDER
# ============================================================

train_max_year = (
    train_df[YEAR_COLUMN].max()
)

test_min_year = (
    test_df[YEAR_COLUMN].min()
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STRICT TEMPORAL + PARCEL-AWARE SPLIT")
print("=" * 65)

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
    f"{len(final_overlap):,}"
)

print(
    f"Latest train year: "
    f"{train_max_year}"
)

print(
    f"Earliest test year: "
    f"{test_min_year}"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(final_overlap) != 0:
    raise ValueError(
        "Parcel leakage detected between "
        "temporal train and test sets."
    )


if train_max_year >= test_min_year:
    raise ValueError(
        "Temporal leakage detected."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTraining target summary:")

print(
    train_df["saleprice"].describe()
)


print("\nTesting target summary:")

print(
    test_df["saleprice"].describe()
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


print("\n" + "=" * 65)
print("TEMPORAL SPLIT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFiles saved:")

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)