import pandas as pd

from src.models.predict_house_price import predict_house_price


# ============================================================
# PATH
# ============================================================

DATA_PATH = (
    "data/processed/"
    "house_price_modeling_spatial.csv"
)


# ============================================================
# LOAD ONE REAL RECORD
# ============================================================

print("Loading modeling dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(f"Rows available: {len(df):,}")


# ============================================================
# PICK ONE REASONABLE PROPERTY
# ============================================================

sample_df = df[
    (df["saleprice"] >= 300_000)
    & (df["saleprice"] <= 700_000)
    & (df["heatedarea"] > 0)
    & (df["bedrooms"] > 0)
    & (df["bathroom_equivalents"] > 0)
    & (df["property_age"].notna())
].copy()


if sample_df.empty:
    raise ValueError(
        "No suitable sample property was found."
    )


sample = sample_df.iloc[0]


# ============================================================
# RECONSTRUCT YEAR BUILT
# ============================================================

sale_year = int(
    sample["sale_year"]
)

property_age = int(
    sample["property_age"]
)

year_built = (
    sale_year
    - property_age
)


# ============================================================
# SHOW ACTUAL PROPERTY
# ============================================================

print("\n" + "=" * 70)
print("REAL PROPERTY SAMPLE")
print("=" * 70)

print(
    f"Parcel ID              : "
    f"{sample['parcelid']}"
)

print(
    f"Actual sale price       : "
    f"${sample['saleprice']:,.2f}"
)

print(
    f"Sale year               : "
    f"{sale_year}"
)

print(
    f"Year built              : "
    f"{year_built}"
)

print(
    f"Acreage                 : "
    f"{sample['gisacres']}"
)

print(
    f"Heated area             : "
    f"{sample['heatedarea']}"
)

print(
    f"Finished area           : "
    f"{sample['finisharea']}"
)

print(
    f"Bedrooms                : "
    f"{sample['bedrooms']}"
)

print(
    f"Bathroom equivalents    : "
    f"{sample['bathroom_equivalents']}"
)

print(
    f"Fireplaces              : "
    f"{sample['fireplaces']}"
)

print(
    f"Finished garage         : "
    f"{sample['fingarage']}"
)

print(
    f"Finished attic          : "
    f"{sample['finattic']}"
)

print(
    f"Land use                : "
    f"{sample['landusefulldescription']}"
)

print(
    f"Neighborhood            : "
    f"{sample['neighborhood']}"
)

print(
    f"Latitude                : "
    f"{sample['latitude']}"
)

print(
    f"Longitude               : "
    f"{sample['longitude']}"
)

print(
    f"Stored Uptown distance  : "
    f"{sample['distance_to_uptown_miles']:.4f} miles"
)


# ============================================================
# MAKE PRODUCTION PREDICTION
# ============================================================

print("\nGenerating production prediction...")

result = predict_house_price(
    sale_year=sale_year,
    year_built=year_built,
    gisacres=sample["gisacres"],
    heatedarea=sample["heatedarea"],
    finisharea=sample["finisharea"],
    bedrooms=sample["bedrooms"],
    bathroom_equivalents=
        sample["bathroom_equivalents"],
    fireplaces=sample["fireplaces"],
    fingarage=sample["fingarage"],
    finattic=sample["finattic"],
    landusefulldescription=
        sample["landusefulldescription"],
    neighborhood=
        sample["neighborhood"],
    latitude=sample["latitude"],
    longitude=sample["longitude"],
)


# ============================================================
# RESULTS
# ============================================================

predicted_price = result[
    "predicted_price"
]

actual_price = float(
    sample["saleprice"]
)

absolute_error = abs(
    predicted_price
    - actual_price
)

percent_error = (
    absolute_error
    / actual_price
    * 100
)


print("\n" + "=" * 70)
print("PRODUCTION PREDICTION RESULT")
print("=" * 70)

print(
    f"Actual sale price      : "
    f"${actual_price:,.2f}"
)

print(
    f"Predicted price        : "
    f"${predicted_price:,.2f}"
)

print(
    f"Absolute error         : "
    f"${absolute_error:,.2f}"
)

print(
    f"Absolute % error       : "
    f"{percent_error:.2f}%"
)

print(
    f"Calculated property age: "
    f"{result['property_age']}"
)

print(
    f"Calculated Uptown dist : "
    f"{result['distance_to_uptown_miles']} miles"
)


# ============================================================
# DISTANCE CONSISTENCY CHECK
# ============================================================

stored_distance = float(
    sample["distance_to_uptown_miles"]
)

calculated_distance = float(
    result[
        "distance_to_uptown_miles"
    ]
)

distance_difference = abs(
    stored_distance
    - calculated_distance
)


print("\n" + "=" * 70)
print("FEATURE ENGINEERING CHECK")
print("=" * 70)

print(
    f"Stored distance     : "
    f"{stored_distance:.4f}"
)

print(
    f"Calculated distance : "
    f"{calculated_distance:.4f}"
)

print(
    f"Difference          : "
    f"{distance_difference:.4f} miles"
)

print(
    "\nProduction inference test completed."
)