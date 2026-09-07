from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/mecklenburg")

SALES_FILE = DATA_DIR / "Parcel_Sales_Table.csv"
CAMA_FILE = DATA_DIR / "Cama_Table.csv"


def main() -> None:
    print("=" * 80)
    print("MECKLENBURG COUNTY DATA RELATIONSHIP ANALYSIS")
    print("=" * 80)

    print("\nReading Sales data...")

    sales = pd.read_csv(
        SALES_FILE,
        usecols=[
            "parcelid",
            "transferid",
            "propertyid",
            "saleprice",
            "saledate",
            "landusefulldescription",
            "landuse",
            "deeddescription",
            "salesvalidity",
        ],
        low_memory=False,
    )

    print(f"Sales rows: {len(sales):,}")
    print(f"Unique sales parcels: {sales['parcelid'].nunique():,}")
    print(f"Unique transfer IDs: {sales['transferid'].nunique():,}")

    print("\nReading CAMA data...")

    cama = pd.read_csv(
        CAMA_FILE,
        usecols=[
            "parcelid",
            "saleprice",
            "saledate",
            "validsale",
            "landuse_description",
            "heatedarea",
            "bedrooms",
            "fullbath",
            "halfbath",
            "threequabath",
            "yearbuilt",
            "gisacres",
            "zipcode",
            "xcoord",
            "ycoord",
            "grade",
        ],
        low_memory=False,
    )

    print(f"CAMA rows: {len(cama):,}")
    print(f"Unique CAMA parcels: {cama['parcelid'].nunique():,}")

    # ------------------------------------------------------------------
    # Parcel overlap
    # ------------------------------------------------------------------

    sales_parcels = set(sales["parcelid"].dropna())
    cama_parcels = set(cama["parcelid"].dropna())

    common_parcels = sales_parcels.intersection(cama_parcels)

    print("\n" + "-" * 80)
    print("PARCEL OVERLAP")
    print("-" * 80)

    print(f"Sales parcels:       {len(sales_parcels):,}")
    print(f"CAMA parcels:        {len(cama_parcels):,}")
    print(f"Common parcels:      {len(common_parcels):,}")

    sales_match_pct = len(common_parcels) / len(sales_parcels) * 100

    print(f"Sales parcels found in CAMA: {sales_match_pct:.2f}%")

    # ------------------------------------------------------------------
    # CAMA duplicate parcels
    # ------------------------------------------------------------------

    cama_duplicate_count = cama["parcelid"].duplicated().sum()

    print("\n" + "-" * 80)
    print("CAMA DUPLICATES")
    print("-" * 80)

    print(f"CAMA duplicate parcel rows: {cama_duplicate_count:,}")

    # ------------------------------------------------------------------
    # Sales per parcel
    # ------------------------------------------------------------------

    sales_per_parcel = sales.groupby("parcelid").size()

    print("\n" + "-" * 80)
    print("SALES PER PARCEL")
    print("-" * 80)

    print(f"Parcels with 1 sale:  {(sales_per_parcel == 1).sum():,}")
    print(f"Parcels with 2 sales: {(sales_per_parcel == 2).sum():,}")
    print(f"Parcels with 3+ sales: {(sales_per_parcel >= 3).sum():,}")

    print(
        f"Maximum sales for one parcel: "
        f"{sales_per_parcel.max():,}"
    )

    # ------------------------------------------------------------------
    # CAMA sale information
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("CAMA SALE INFORMATION")
    print("-" * 80)

    print(
        f"CAMA rows with sale price: "
        f"{cama['saleprice'].notna().sum():,}"
    )

    print(
        f"CAMA rows with sale price > 0: "
        f"{(cama['saleprice'] > 0).sum():,}"
    )

    print(
        f"CAMA rows with valid sale flag: "
        f"{cama['validsale'].notna().sum():,}"
    )

    print("\nCAMA valid-sale values:")
    print(cama["validsale"].value_counts(dropna=False).head(20))

    # ------------------------------------------------------------------
    # Sales validity
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SALES VALIDITY")
    print("-" * 80)

    print(sales["salesvalidity"].value_counts(dropna=False).head(20))

    # ------------------------------------------------------------------
    # Land use
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TOP SALES PROPERTY TYPES")
    print("-" * 80)

    print(
        sales["landusefulldescription"]
        .value_counts(dropna=False)
        .head(30)
    )

    # ------------------------------------------------------------------
    # Sale prices
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SALE PRICE SUMMARY")
    print("-" * 80)

    print(sales["saleprice"].describe())

    print(
        f"\nZero-price sales: "
        f"{(sales['saleprice'] == 0).sum():,}"
    )

    print(
        f"Missing sale prices: "
        f"{sales['saleprice'].isna().sum():,}"
    )

    # ------------------------------------------------------------------
    # Sample matching records
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SAMPLE MATCHING RECORDS")
    print("-" * 80)

    sample_parcels = list(common_parcels)[:5]

    print(
        sales[
            sales["parcelid"].isin(sample_parcels)
        ][
            [
                "parcelid",
                "transferid",
                "saleprice",
                "saledate",
                "landusefulldescription",
                "salesvalidity",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()