import os

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "house_price_modeling_coordinates.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "house_price_modeling_spatial.csv"
)


# Approximate center of Uptown Charlotte
UPTOWN_LATITUDE = 35.2271
UPTOWN_LONGITUDE = -80.8431

EARTH_RADIUS_MILES = 3958.8


# ============================================================
# LOAD DATA
# ============================================================

print("Loading coordinate modeling dataset...")

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
)

print(f"Rows   : {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# VERIFY COORDINATES
# ============================================================

required_columns = [
    "latitude",
    "longitude",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\nMissing coordinates:")

print(
    df[
        ["latitude", "longitude"]
    ]
    .isna()
    .sum()
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

print("\nCalculating distance to Uptown Charlotte...")


latitude_radians = np.radians(
    df["latitude"]
)

longitude_radians = np.radians(
    df["longitude"]
)

uptown_latitude_radians = np.radians(
    UPTOWN_LATITUDE
)

uptown_longitude_radians = np.radians(
    UPTOWN_LONGITUDE
)


delta_latitude = (
    latitude_radians
    - uptown_latitude_radians
)

delta_longitude = (
    longitude_radians
    - uptown_longitude_radians
)


a = (
    np.sin(delta_latitude / 2) ** 2
    +
    np.cos(uptown_latitude_radians)
    * np.cos(latitude_radians)
    * np.sin(delta_longitude / 2) ** 2
)


c = 2 * np.arctan2(
    np.sqrt(a),
    np.sqrt(1 - a),
)


df["distance_to_uptown_miles"] = (
    EARTH_RADIUS_MILES * c
)


# ============================================================
# VALIDATION
# ============================================================

missing_distance = (
    df["distance_to_uptown_miles"]
    .isna()
    .sum()
)

negative_distance = (
    df["distance_to_uptown_miles"]
    < 0
).sum()


print("\n" + "=" * 60)
print("UPTOWN DISTANCE FEATURE SUMMARY")
print("=" * 60)

print(
    f"Missing distance : "
    f"{missing_distance:,}"
)

print(
    f"Negative distance: "
    f"{negative_distance:,}"
)


print("\nDistance distribution:")

print(
    df["distance_to_uptown_miles"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


# ============================================================
# SAMPLE CLOSEST PROPERTIES
# ============================================================

sample_columns = [
    "parcelid",
    "saleprice",
    "neighborhood",
    "latitude",
    "longitude",
    "distance_to_uptown_miles",
]


print("\nClosest 10 records to Uptown:")

print(
    df[
        sample_columns
    ]
    .sort_values(
        "distance_to_uptown_miles"
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAMPLE FARTHEST PROPERTIES
# ============================================================

print("\nFarthest 10 records from Uptown:")

print(
    df[
        sample_columns
    ]
    .sort_values(
        "distance_to_uptown_miles",
        ascending=False,
    )
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

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 60)
print("UPTOWN DISTANCE FEATURE ADDED SUCCESSFULLY")
print("=" * 60)

print(f"\nRows   : {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nOutput saved to:")

print(OUTPUT_PATH)