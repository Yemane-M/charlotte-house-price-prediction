import pandas as pd


PREDICTIONS_PATH = (
    "data/interim/"
    "xgboost_grouped_coordinates_test_predictions.csv"
)


print("Loading predictions...")

df = pd.read_csv(
    PREDICTIONS_PATH,
    low_memory=False,
)


# ============================================================
# FILTER LUXURY TRANSACTIONS
# ============================================================

luxury = df[
    df["saleprice"] >= 5_000_000
].copy()


luxury["error"] = (
    luxury["saleprice"]
    - luxury["predicted_price"]
)

luxury["absolute_error"] = (
    luxury["error"].abs()
)

luxury["percent_error"] = (
    luxury["absolute_error"]
    / luxury["saleprice"]
    * 100
)


# ============================================================
# SELECT USEFUL COLUMNS
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
    "percent_error",
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
]

columns = [
    column
    for column in possible_columns
    if column in luxury.columns
]


# ============================================================
# SORT BY SALE PRICE
# ============================================================

by_price = luxury.sort_values(
    "saleprice",
    ascending=False,
)


print("\n" + "=" * 120)
print("ALL TEST SALES >= $5 MILLION")
print("=" * 120)

print(
    by_price[columns]
    .to_string(index=False)
)


# ============================================================
# WORST UNDERPREDICTIONS
# ============================================================

underpredictions = luxury.sort_values(
    "error",
    ascending=False,
)


print("\n" + "=" * 120)
print("LUXURY SALES SORTED BY UNDERPREDICTION")
print("=" * 120)

print(
    underpredictions[columns]
    .to_string(index=False)
)


# ============================================================
# SIMPLE SUMMARY
# ============================================================

print("\n" + "=" * 120)
print("LUXURY MODEL SUMMARY")
print("=" * 120)

print(
    f"Transactions        : {len(luxury):,}"
)

print(
    f"Unique parcels      : "
    f"{luxury['parcelid'].nunique():,}"
)

print(
    f"Mean actual price   : "
    f"${luxury['saleprice'].mean():,.0f}"
)

print(
    f"Mean predicted price: "
    f"${luxury['predicted_price'].mean():,.0f}"
)

print(
    f"Mean absolute error : "
    f"${luxury['absolute_error'].mean():,.0f}"
)

print(
    f"Median absolute error: "
    f"${luxury['absolute_error'].median():,.0f}"
)

print(
    f"Mean percent error  : "
    f"{luxury['percent_error'].mean():.2f}%"
)

print(
    f"Median percent error: "
    f"{luxury['percent_error'].median():.2f}%"
)

print(
    f"Underpredictions    : "
    f"{(luxury['error'] > 0).sum():,}"
)

print(
    f"Overpredictions     : "
    f"{(luxury['error'] < 0).sum():,}"
)