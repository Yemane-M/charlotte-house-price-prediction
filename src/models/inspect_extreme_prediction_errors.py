import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = (
    "data/interim/"
    "xgboost_grouped_coordinates_test_predictions.csv"
)

TOP_N = 20


# ============================================================
# LOAD
# ============================================================

print("Loading coordinate-model predictions...")

df = pd.read_csv(
    PREDICTIONS_PATH,
    low_memory=False,
)

print(f"Rows: {len(df):,}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required = [
    "parcelid",
    "saleprice",
    "predicted_price",
    "error",
    "absolute_error",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# COLUMNS TO INSPECT
# ============================================================

possible_columns = [
    "parcelid",
    "transferid",
    "propertyid",
    "saledate",
    "sale_year",
    "saleprice",
    "predicted_price",
    "error",
    "absolute_error",
    "percentage_error",
    "landusefulldescription",
    "neighborhood",
    "latitude",
    "longitude",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "finattic",
    "property_age",
    "yearbuilt",
    "effyearblt",
]

columns = [
    column
    for column in possible_columns
    if column in df.columns
]


# ============================================================
# LARGEST UNDERPREDICTIONS
# ============================================================

under = (
    df.sort_values(
        "error",
        ascending=False,
    )
    .head(TOP_N)
)

print("\n" + "=" * 80)
print("TOP 20 UNDERPREDICTIONS")
print("=" * 80)

print(
    under[columns]
    .to_string(index=False)
)


# ============================================================
# LARGEST OVERPREDICTIONS
# ============================================================

over = (
    df.sort_values(
        "error",
        ascending=True,
    )
    .head(TOP_N)
)

print("\n" + "=" * 80)
print("TOP 20 OVERPREDICTIONS")
print("=" * 80)

print(
    over[columns]
    .to_string(index=False)
)


# ============================================================
# SALES >= $2 MILLION
# ============================================================

luxury = df[
    df["saleprice"] >= 2_000_000
].copy()

print("\n" + "=" * 80)
print("SALES >= $2 MILLION")
print("=" * 80)

print(f"Transactions: {len(luxury):,}")
print(
    f"Unique parcels: "
    f"{luxury['parcelid'].nunique():,}"
)

print("\nSale price distribution:")

print(
    luxury["saleprice"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


# ============================================================
# VERY HIGH SALES
# ============================================================

very_high = df[
    df["saleprice"] >= 5_000_000
].copy()

print("\n" + "=" * 80)
print("SALES >= $5 MILLION")
print("=" * 80)

print(f"Transactions: {len(very_high):,}")
print(
    f"Unique parcels: "
    f"{very_high['parcelid'].nunique():,}"
)

print("\nProperty characteristics:")

summary_columns = [
    "saleprice",
    "predicted_price",
    "gisacres",
    "heatedarea",
    "finisharea",
    "bedrooms",
    "bathroom_equivalents",
    "fireplaces",
    "fingarage",
    "property_age",
]

summary_columns = [
    column
    for column in summary_columns
    if column in very_high.columns
]

print(
    very_high[summary_columns]
    .describe()
)


print("\nExtreme prediction inspection completed.")