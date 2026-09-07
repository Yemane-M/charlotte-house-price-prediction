import pandas as pd


DATA_PATH = "data/processed/house_price_modeling_location.csv"


print("Loading location-enriched dataset...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"zipcode": "string"},
    low_memory=False,
)

print(f"Rows: {len(df):,}")


# ============================================================
# BASIC ZIPCODE QUALITY
# ============================================================

zipcode = df["zipcode"].str.strip()

print("\n" + "=" * 60)
print("ZIPCODE FORMAT ANALYSIS")
print("=" * 60)

print(f"Unique raw ZIP values: {zipcode.nunique(dropna=True):,}")
print(f"Missing: {zipcode.isna().sum():,}")


# Length distribution
print("\nZIP value length distribution:")

print(
    zipcode
    .dropna()
    .str.len()
    .value_counts()
    .sort_index()
)


# Numeric/non-numeric
five_digit = zipcode.str.fullmatch(r"\d{5}", na=False)

print(f"\nValid 5-digit format: {five_digit.sum():,}")
print(f"Other format       : {(~five_digit & zipcode.notna()).sum():,}")


# ============================================================
# MOST COMMON VALUES
# ============================================================

print("\n" + "=" * 60)
print("TOP 50 ZIP VALUES")
print("=" * 60)

print(
    zipcode
    .value_counts()
    .head(50)
    .to_string()
)


# ============================================================
# SUSPICIOUS HIGH-CARDINALITY VALUES
# ============================================================

counts = zipcode.value_counts()

print("\n" + "=" * 60)
print("ZIP FREQUENCY DISTRIBUTION")
print("=" * 60)

print(f"ZIP values occurring once : {(counts == 1).sum():,}")
print(f"ZIP values occurring <= 2 : {(counts <= 2).sum():,}")
print(f"ZIP values occurring <= 5 : {(counts <= 5).sum():,}")
print(f"ZIP values occurring <=10 : {(counts <= 10).sum():,}")
print(f"ZIP values occurring >100 : {(counts > 100).sum():,}")


# ============================================================
# SAMPLE RARE VALUES
# ============================================================

rare_values = counts[counts <= 2].index

rare_rows = df[
    df["zipcode"].astype("string").isin(rare_values)
].copy()

columns_to_show = [
    col
    for col in [
        "parcelid",
        "saleprice",
        "neighborhood",
        "landusefulldescription",
        "zipcode",
    ]
    if col in rare_rows.columns
]

print("\n" + "=" * 60)
print("SAMPLE RARE ZIP VALUES")
print("=" * 60)

print(
    rare_rows[columns_to_show]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# CHECK CHARLOTTE-LIKE ZIP PREFIXES
# ============================================================

# This is diagnostic only.
# Do NOT delete anything based on this check.

starts_28 = zipcode.str.startswith("28", na=False)

print("\n" + "=" * 60)
print("PREFIX DIAGNOSTIC")
print("=" * 60)

print(
    f"Rows beginning with '28': "
    f"{starts_28.sum():,} "
    f"({starts_28.mean() * 100:.2f}%)"
)

print(
    f"Rows not beginning with '28': "
    f"{((~starts_28) & zipcode.notna()).sum():,}"
)


print("\nMost common ZIP values not beginning with 28:")

print(
    zipcode[
        (~starts_28) & zipcode.notna()
    ]
    .value_counts()
    .head(30)
    .to_string()
)


print("\nZIP quality analysis completed.")