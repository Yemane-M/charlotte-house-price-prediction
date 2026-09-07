from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "house_price_modeling.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ---------------------------------------------------------
# Find location-related columns
# ---------------------------------------------------------
location_keywords = [
    "neighborhood",
    "zip",
    "coord",
    "address",
    "city",
    "state",
]

location_columns = [
    column
    for column in df.columns
    if any(
        keyword in column.lower()
        for keyword in location_keywords
    )
]


print("\n" + "=" * 60)
print("LOCATION-RELATED COLUMNS")
print("=" * 60)

if location_columns:
    for column in location_columns:
        print(f"- {column}")
else:
    print("No location-related columns found.")


# ---------------------------------------------------------
# Neighborhood analysis
# ---------------------------------------------------------
if "neighborhood" in df.columns:

    print("\n" + "=" * 60)
    print("NEIGHBORHOOD ANALYSIS")
    print("=" * 60)

    missing = df["neighborhood"].isna().sum()
    unique = df["neighborhood"].nunique()

    print(f"Unique neighborhoods : {unique:,}")
    print(f"Missing              : {missing:,}")
    print(
        f"Missing %            : "
        f"{missing / len(df) * 100:.2f}%"
    )

    print("\nTop 30 neighborhoods:")

    print(
        df["neighborhood"]
        .value_counts(dropna=False)
        .head(30)
    )

    print("\nNeighborhood price analysis:")

    neighborhood_price = (
        df.groupby("neighborhood")["saleprice"]
        .agg(
            records="count",
            median_price="median",
            mean_price="mean",
        )
        .sort_values(
            "records",
            ascending=False,
        )
    )

    print(
        neighborhood_price.head(30)
    )


print("\nLocation feature analysis completed successfully.")