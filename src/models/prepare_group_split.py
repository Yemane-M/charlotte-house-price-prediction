from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "house_price_modeling.csv"
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train_grouped.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_grouped.csv"


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
GROUP = "parcelid"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")
print(f"Unique parcels: {df[GROUP].nunique():,}")


# ---------------------------------------------------------
# Keep required columns
# ---------------------------------------------------------
required_columns = FEATURES + [TARGET, GROUP]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

model_df = df[required_columns].copy()


# ---------------------------------------------------------
# Group-aware train/test split
# ---------------------------------------------------------
print("\nCreating group-aware train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        model_df,
        groups=model_df[GROUP],
    )
)

train_df = model_df.iloc[train_idx].copy()
test_df = model_df.iloc[test_idx].copy()


# ---------------------------------------------------------
# Verify no parcel overlap
# ---------------------------------------------------------
train_parcels = set(train_df[GROUP])
test_parcels = set(test_df[GROUP])

overlap = train_parcels.intersection(test_parcels)


# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("GROUP-AWARE SPLIT RESULTS")
print("=" * 60)

print(f"Training rows       : {len(train_df):,}")
print(f"Testing rows        : {len(test_df):,}")

print(f"Training parcels    : {train_df[GROUP].nunique():,}")
print(f"Testing parcels     : {test_df[GROUP].nunique():,}")

print(f"Parcel overlap      : {len(overlap)}")

print("\nTraining percentage:")
print(f"{len(train_df) / len(model_df) * 100:.2f}%")

print("\nTesting percentage:")
print(f"{len(test_df) / len(model_df) * 100:.2f}%")

print("\nTarget distribution:")

print("\nTraining:")
print(train_df[TARGET].describe())

print("\nTesting:")
print(test_df[TARGET].describe())

print("\nFiles created:")
print(f"Training: {TRAIN_FILE}")
print(f"Testing : {TEST_FILE}")

if len(overlap) == 0:
    print("\nSUCCESS: No parcel appears in both training and testing.")
else:
    raise ValueError(
        f"ERROR: {len(overlap)} parcels appear in both datasets."
    )