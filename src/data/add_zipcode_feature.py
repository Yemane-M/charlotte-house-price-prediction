import os

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODELING_DATA_PATH = "data/processed/house_price_modeling.csv"
CAMA_DATA_PATH = "data/raw/mecklenburg/Cama_Table.csv"

OUTPUT_PATH = "data/processed/house_price_modeling_location.csv"


# ============================================================
# LOAD MODELING DATA
# ============================================================

print("Loading modeling dataset...")

model_df = pd.read_csv(MODELING_DATA_PATH)

print(f"Modeling rows: {len(model_df):,}")
print(f"Modeling columns: {len(model_df.columns):,}")


# ============================================================
# LOAD ZIP CODE FROM CAMA
# ============================================================

print("\nLoading parcelid and zipcode from CAMA...")

cama_zip = pd.read_csv(
    CAMA_DATA_PATH,
    usecols=[
        "parcelid",
        "zipcode",
    ],
    low_memory=False,
)

print(f"CAMA rows loaded: {len(cama_zip):,}")


# ============================================================
# CLEAN PARCEL ID
# ============================================================

model_df["parcelid"] = (
    model_df["parcelid"]
    .astype(str)
    .str.strip()
)

cama_zip["parcelid"] = (
    cama_zip["parcelid"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CLEAN ZIP CODE
# ============================================================

cama_zip["zipcode"] = (
    cama_zip["zipcode"]
    .astype("string")
    .str.strip()
)

# Convert obvious text versions of missing values to NA
cama_zip["zipcode"] = cama_zip["zipcode"].replace(
    {
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
    }
)


# ============================================================
# ANALYZE ZIP CODE DUPLICATES BY PARCEL
# ============================================================

print("\nChecking ZIP codes per parcel...")

zipcode_counts = (
    cama_zip
    .dropna(subset=["zipcode"])
    .groupby("parcelid")["zipcode"]
    .nunique()
)

multiple_zip_parcels = zipcode_counts[
    zipcode_counts > 1
]

print(
    f"Parcels with more than one ZIP code: "
    f"{len(multiple_zip_parcels):,}"
)


# ============================================================
# CREATE ONE ZIP CODE PER PARCEL
# ============================================================

# For most parcels, ZIP code should be identical across CAMA rows.
# We keep the most frequently occurring non-null ZIP code
# for each parcel rather than blindly dropping duplicates.

def choose_zipcode(series):
    non_null = series.dropna()

    if non_null.empty:
        return pd.NA

    mode_values = non_null.mode()

    if not mode_values.empty:
        return mode_values.iloc[0]

    return non_null.iloc[0]


parcel_zip = (
    cama_zip
    .groupby("parcelid", as_index=False)
    .agg(
        zipcode=("zipcode", choose_zipcode)
    )
)

print(
    f"Unique CAMA parcels with ZIP mapping: "
    f"{len(parcel_zip):,}"
)


# ============================================================
# MERGE ZIP CODE INTO MODELING DATASET
# ============================================================

print("\nAdding zipcode to modeling dataset...")

enriched_df = model_df.merge(
    parcel_zip,
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

if len(enriched_df) != len(model_df):
    raise ValueError(
        "Row count changed after ZIP code merge."
    )


# ============================================================
# ZIP CODE SUMMARY
# ============================================================

missing_zip = enriched_df["zipcode"].isna().sum()
unique_zip = enriched_df["zipcode"].nunique(dropna=True)

print("\n" + "=" * 60)
print("ZIP CODE SUMMARY")
print("=" * 60)

print(
    f"Missing ZIP codes: "
    f"{missing_zip:,} "
    f"({missing_zip / len(enriched_df) * 100:.2f}%)"
)

print(
    f"Unique ZIP codes: {unique_zip:,}"
)

print("\nTop 20 ZIP codes:")

print(
    enriched_df["zipcode"]
    .value_counts(dropna=False)
    .head(20)
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

print("\n" + "=" * 60)
print("ZIP CODE FEATURE ADDED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput saved to:")
print(OUTPUT_PATH)