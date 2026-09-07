import os

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODELING_DATA_PATH = "data/processed/house_price_modeling.csv"
CAMA_DATA_PATH = "data/raw/mecklenburg/Cama_Table.csv"

OUTPUT_PATH = (
    "data/processed/"
    "house_price_modeling_coordinates.csv"
)


# ============================================================
# LOAD MODELING DATA
# ============================================================

print("Loading modeling dataset...")

model_df = pd.read_csv(
    MODELING_DATA_PATH,
    low_memory=False,
)

print(f"Modeling rows   : {len(model_df):,}")
print(f"Modeling columns: {len(model_df.columns):,}")


# ============================================================
# LOAD COORDINATES FROM CAMA
# ============================================================

print("\nLoading coordinates from CAMA...")

coordinates = pd.read_csv(
    CAMA_DATA_PATH,
    usecols=[
        "parcelid",
        "xcoord",
        "ycoord",
    ],
    low_memory=False,
)

print(f"CAMA rows loaded: {len(coordinates):,}")


# ============================================================
# CLEAN PARCEL IDs
# ============================================================

model_df["parcelid"] = (
    model_df["parcelid"]
    .astype(str)
    .str.strip()
)

coordinates["parcelid"] = (
    coordinates["parcelid"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT COORDINATES TO NUMERIC
# ============================================================

coordinates["xcoord"] = pd.to_numeric(
    coordinates["xcoord"],
    errors="coerce",
)

coordinates["ycoord"] = pd.to_numeric(
    coordinates["ycoord"],
    errors="coerce",
)


# ============================================================
# VERIFY COORDINATE CONSISTENCY
# ============================================================

print("\nChecking coordinate consistency...")

coord_counts = (
    coordinates
    .dropna(subset=["xcoord", "ycoord"])
    .groupby("parcelid")
    .agg(
        x_values=("xcoord", "nunique"),
        y_values=("ycoord", "nunique"),
    )
)

conflicting = coord_counts[
    (coord_counts["x_values"] > 1)
    | (coord_counts["y_values"] > 1)
]

print(
    "Parcels with conflicting coordinates: "
    f"{len(conflicting):,}"
)

if len(conflicting) > 0:
    raise ValueError(
        "Conflicting parcel coordinates were found."
    )


# ============================================================
# CREATE ONE COORDINATE RECORD PER PARCEL
# ============================================================

parcel_coordinates = (
    coordinates[
        [
            "parcelid",
            "xcoord",
            "ycoord",
        ]
    ]
    .drop_duplicates(
        subset=["parcelid"],
        keep="first",
    )
    .copy()
)

parcel_coordinates = parcel_coordinates.rename(
    columns={
        "xcoord": "latitude",
        "ycoord": "longitude",
    }
)

print(
    "Unique coordinate parcel mappings: "
    f"{len(parcel_coordinates):,}"
)


# ============================================================
# MERGE INTO MODELING DATASET
# ============================================================

print("\nAdding coordinates to modeling dataset...")

enriched_df = model_df.merge(
    parcel_coordinates,
    on="parcelid",
    how="left",
    validate="many_to_one",
)


# ============================================================
# VERIFY ROW COUNT
# ============================================================

print(
    f"Rows before merge: {len(model_df):,}"
)

print(
    f"Rows after merge : {len(enriched_df):,}"
)

if len(model_df) != len(enriched_df):
    raise ValueError(
        "Row count changed after coordinate merge."
    )


# ============================================================
# VERIFY MISSING VALUES
# ============================================================

missing_latitude = (
    enriched_df["latitude"].isna().sum()
)

missing_longitude = (
    enriched_df["longitude"].isna().sum()
)

print("\n" + "=" * 60)
print("COORDINATE FEATURE SUMMARY")
print("=" * 60)

print(
    f"Missing latitude : {missing_latitude:,}"
)

print(
    f"Missing longitude: {missing_longitude:,}"
)

print("\nCoordinate ranges:")

print(
    enriched_df[
        [
            "latitude",
            "longitude",
        ]
    ].describe()
)


# ============================================================
# SAMPLE
# ============================================================

print("\nSample modeling records:")

sample_columns = [
    "parcelid",
    "saleprice",
    "neighborhood",
    "latitude",
    "longitude",
]

print(
    enriched_df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

enriched_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("COORDINATE FEATURES ADDED SUCCESSFULLY")
print("=" * 60)

print("\nOutput saved to:")

print(OUTPUT_PATH)